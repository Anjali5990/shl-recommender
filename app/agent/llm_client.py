"""
Thin wrapper around Groq's chat completions API (OpenAI-compatible schema).
Kept dependency-free (httpx only) rather than pulling in the groq SDK, since
we only need one endpoint.

Swappable: to use a different provider (Gemini, OpenRouter), only this file
needs to change -- agent.py just calls `complete_json(system, user)`.
"""
import json
import os
import time

import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "9"))
MAX_RATE_LIMIT_RETRIES = 1


class LLMError(Exception):
    pass


def _post(payload: dict, headers: dict) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        return client.post(GROQ_API_URL, json=payload, headers=headers)


def complete_json(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """Calls the LLM and returns a parsed JSON dict. Raises LLMError on
    failure (network, non-200, or unparseable JSON) so the caller can decide
    how to fall back -- this function never silently returns bad data.

    Retries on 429 (rate limit) with backoff honoring Retry-After when
    present, since Groq's free tier has a fairly low requests-per-minute
    cap and a burst of turns (e.g. our own eval script replaying traces
    back-to-back) can trip it even though a real, human-paced conversation
    rarely would."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set")

    payload = {
        "model": DEFAULT_MODEL,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            resp = _post(payload, headers)
        except httpx.RequestError as e:
            raise LLMError(f"request failed: {e}") from e

        if resp.status_code == 429:
            last_error = f"rate limited (attempt {attempt + 1})"
            if attempt < MAX_RATE_LIMIT_RETRIES:
                retry_after = resp.headers.get("retry-after")
                wait = min(float(retry_after), 3) if retry_after else 1
                time.sleep(min(wait, 8))
                continue
            raise LLMError(f"LLM returned 429 after {MAX_RATE_LIMIT_RETRIES} retries")

        if resp.status_code != 200:
            raise LLMError(f"LLM returned {resp.status_code}: {resp.text[:500]}")

        try:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"could not parse LLM response: {e}") from e

    raise LLMError(last_error or "unknown error")
