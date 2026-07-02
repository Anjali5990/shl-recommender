"""
Thin wrapper around Groq's chat completions API (OpenAI compatible schema).
Kept dependency free (httpx only) rather than pulling in the groq SDK, since
we only need one endpoint.
Swappable: to use a different provider (Gemini, OpenRouter), only this file
needs to change, agent.py just calls complete_json(system, user).
"""
import json
import os
import time
import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "8"))
MAX_TOTAL_SECONDS = float(os.environ.get("LLM_MAX_TOTAL_SECONDS", "25"))


class LLMError(Exception):
    pass


def _post(payload, headers):
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        return client.post(GROQ_API_URL, json=payload, headers=headers)


def complete_json(system_prompt, user_prompt, temperature=0.2):
    """
    Calls the LLM and returns a parsed JSON dict. Raises LLMError on
    failure so the caller can decide how to fall back.

    Retries on 429 using the actual retry after value Groq sends back,
    instead of a short fixed wait. The whole function stops once
    MAX_TOTAL_SECONDS has passed, so a single call never risks going past
    the assignment's 30 second per call limit.
    """
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
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    start_time = time.monotonic()
    last_error = None
    attempt = 0

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= MAX_TOTAL_SECONDS:
            raise LLMError(last_error or "timed out before a successful response")

        attempt += 1
        try:
            resp = _post(payload, headers)
        except httpx.RequestError as e:
            raise LLMError("request failed: " + str(e)) from e

        if resp.status_code == 429:
            last_error = "rate limited on attempt " + str(attempt)
            retry_after_header = resp.headers.get("retry-after")
            if retry_after_header:
                wait_seconds = float(retry_after_header)
            else:
                wait_seconds = 2.0

            time_left = MAX_TOTAL_SECONDS - (time.monotonic() - start_time)
            if wait_seconds >= time_left:
                raise LLMError(last_error)

            time.sleep(wait_seconds)
            continue

        if resp.status_code != 200:
            raise LLMError("LLM returned " + str(resp.status_code) + ": " + resp.text[:500])

        try:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError("could not parse LLM response: " + str(e)) from e
