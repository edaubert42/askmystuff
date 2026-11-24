"""Backend `gemini`: the Gemini API over HTTPS (demo, reference run) and the embeddings."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from askmystuff.llm.completion import Completion, LLMError
from askmystuff.llm.prices import cost_usd, price_of

API = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH = 100  # the API's limit per batchEmbedContents call
TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
TIMEOUT_S = 300
RETRY_DELAY_S = 2.0


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise HttpError(error.code, error.read().decode(errors="replace")[:500]) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise LLMError(f"Gemini API unreachable: {error}") from error


def _call(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to the API, with one retry on HTTP 429 and 5xx."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GOOGLE_API_KEY is not set")
    for attempt in (1, 2):
        try:
            return _post(f"{API}/{path}", {"x-goog-api-key": key}, body)
        except HttpError as error:
            retryable = error.status == 429 or error.status >= 500
            if retryable and attempt == 1:
                _sleep(RETRY_DELAY_S)
                continue
            raise LLMError(f"Gemini API failed with {error}") from error
    raise AssertionError("unreachable")  # pragma: no cover


def complete(system: str, prompt: str, model: str | None = None) -> Completion:
    model = model or DEFAULT_MODEL
    price_of(model)  # fail before spending anything on an unpriced model
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    start = time.perf_counter()
    reply = _call(f"models/{model}:generateContent", body)
    latency_s = time.perf_counter() - start
    try:
        parts: list[dict[str, Any]] = reply["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
        usage: dict[str, int] = reply["usageMetadata"]
        input_tokens = usage["promptTokenCount"]
        # Thinking tokens are billed as output.
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError(f"Gemini API returned no answer: {json.dumps(reply)[:300]}") from error
    return Completion(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(model, input_tokens, output_tokens),
        latency_s=latency_s,
        model=model,
    )


def embed(texts: list[str], task: str) -> list[list[float]]:
    if task not in TASK_TYPES:
        raise LLMError(f"unknown embedding task {task!r}: use one of {sorted(TASK_TYPES)}")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[start : start + EMBEDDING_BATCH]
        body = {
            "requests": [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": TASK_TYPES[task],
                    "outputDimensionality": EMBEDDING_DIMENSIONS,
                }
                for text in batch
            ]
        }
        reply = _call(f"models/{EMBEDDING_MODEL}:batchEmbedContents", body)
        try:
            got = [[float(x) for x in e["values"]] for e in reply["embeddings"]]
        except (KeyError, TypeError) as error:
            raise LLMError(
                f"Gemini API returned no embeddings: {json.dumps(reply)[:300]}"
            ) from error
        if len(got) != len(batch):
            raise LLMError(f"Gemini API returned {len(got)} vectors for {len(batch)} texts")
        vectors.extend(got)
    return vectors
