"""Backend `gemini`: the Gemini API over HTTPS (demo, reference run) and the embeddings."""

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
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


def _post_stream(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """POST and yield the JSON of each server-sent event (`data: {...}` lines)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    yield json.loads(line.removeprefix("data:"))
    except urllib.error.HTTPError as error:
        raise HttpError(error.code, error.read().decode(errors="replace")[:500]) from error
    except (OSError, http.client.HTTPException) as error:
        raise LLMError(f"Gemini API unreachable: {error}") from error
    except ValueError as error:
        raise LLMError(f"Gemini API streamed an unexpected JSON: {error}") from error


def _key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GOOGLE_API_KEY is not set")
    return key


def _retryable(error: HttpError) -> bool:
    return error.status == 429 or error.status >= 500


def _call(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to the API, with one retry on HTTP 429 and 5xx."""
    key = _key()
    for attempt in (1, 2):
        try:
            return _post(f"{API}/{path}", {"x-goog-api-key": key}, body)
        except HttpError as error:
            if _retryable(error) and attempt == 1:
                _sleep(RETRY_DELAY_S)
                continue
            raise LLMError(f"Gemini API failed with {error}") from error
    raise AssertionError("unreachable")  # pragma: no cover


def _call_stream(path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Like `_call`, yielding the events. Retries only a call refused before its first event:
    after that, a retry would replay words the caller has already received."""
    key = _key()
    for attempt in (1, 2):
        received = False
        try:
            for event in _post_stream(f"{API}/{path}", {"x-goog-api-key": key}, body):
                received = True
                yield event
            return
        except HttpError as error:
            if _retryable(error) and attempt == 1 and not received:
                _sleep(RETRY_DELAY_S)
                continue
            raise LLMError(f"Gemini API failed with {error}") from error


def _body(system: str, prompt: str) -> dict[str, Any]:
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }


def _text_of(reply: dict[str, Any]) -> str:
    parts: list[dict[str, Any]] = reply["candidates"][0]["content"]["parts"]
    return "".join(part.get("text", "") for part in parts)


def _tokens(usage: dict[str, int]) -> tuple[int, int]:
    # Thinking tokens are billed as output.
    output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return usage["promptTokenCount"], output


def complete(system: str, prompt: str, model: str | None = None) -> Completion:
    model = model or DEFAULT_MODEL
    price_of(model)  # fail before spending anything on an unpriced model
    start = time.perf_counter()
    reply = _call(f"models/{model}:generateContent", _body(system, prompt))
    latency_s = time.perf_counter() - start
    try:
        text = _text_of(reply)
        input_tokens, output_tokens = _tokens(reply["usageMetadata"])
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


def stream(system: str, prompt: str, model: str | None = None) -> Iterator[str | Completion]:
    model = model or DEFAULT_MODEL
    price_of(model)  # fail before spending anything on an unpriced model
    start = time.perf_counter()
    first_token_s: float | None = None
    written = ""
    usage: dict[str, int] | None = None
    for event in _call_stream(
        f"models/{model}:streamGenerateContent?alt=sse", _body(system, prompt)
    ):
        try:
            delta = _text_of(event)
        except (KeyError, IndexError, TypeError):
            delta = ""  # an event with no text, such as one that only carries the usage
        if delta:
            if first_token_s is None:
                first_token_s = time.perf_counter() - start
            written += delta
            yield delta
        usage = event.get("usageMetadata", usage)  # each event repeats it: the last one counts
    latency_s = time.perf_counter() - start
    try:
        if usage is None:
            raise KeyError("usageMetadata")
        input_tokens, output_tokens = _tokens(usage)
    except (KeyError, TypeError) as error:
        raise LLMError(f"Gemini API streamed no usage: {error!r}") from error
    yield Completion(
        text=written,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(model, input_tokens, output_tokens),
        latency_s=latency_s,
        model=model,
        first_token_s=first_token_s,
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
