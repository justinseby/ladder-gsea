"""
_llm.py — internal LLM router.

annotation.py and validation.py never call any provider directly.
They only ever call:

    response = call_llm(config, system_prompt, user_prompt, max_tokens)

This module resolves which provider to use and handles all
provider-specific SDK differences in one place.
"""

from __future__ import annotations
import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ladder.config import LADDERConfig


# ── Public entry point ────────────────────────────────────────────────────────

def call_llm(
    config,
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int = 4000,
) -> str:
    """
    Send a system + user prompt to the configured LLM provider.
    Returns the response as a plain string.

    Raises RuntimeError on any API failure so callers get a clean message.
    """
    provider = config.llm_provider

    if provider == "openai":
        return _call_openai(config, system_prompt, user_prompt, max_tokens)
    elif provider == "deepseek":
        return _call_deepseek(config, system_prompt, user_prompt, max_tokens)
    elif provider == "anthropic":
        return _call_anthropic(config, system_prompt, user_prompt, max_tokens)
    else:
        # Should never reach here — config.py validates the provider
        raise ValueError(f"Unknown llm_provider: '{provider}'")


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(config, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Call OpenAI chat completions — requires: pip install ladder-gsea[openai]"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not found. "
            "Install it with: pip install ladder-gsea[openai]"
        )

    client = OpenAI(api_key=config.llm_api_key)

    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e


# ── DeepSeek ──────────────────────────────────────────────────────────────────

def _call_deepseek(config, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """
    Call DeepSeek chat completions.
    DeepSeek uses an OpenAI-compatible REST API — no extra SDK needed,
    just requests (already a core dependency).
    """
    url  = "https://api.deepseek.com/v1/chat/completions"
    hdrs = {
        "Authorization": f"Bearer {config.llm_api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": config.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens":  max_tokens,
    }

    try:
        r = requests.post(url, headers=hdrs, json=body, timeout=180)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError(
            f"DeepSeek API returned {r.status_code}: {r.text[:300]}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("DeepSeek API request timed out after 180s.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"DeepSeek network error: {e}") from e


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _call_anthropic(config, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Call Anthropic Claude — requires: pip install ladder-gsea[anthropic]"""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not found. "
            "Install it with: pip install ladder-gsea[anthropic]"
        )

    client = anthropic.Anthropic(api_key=config.llm_api_key)

    try:
        message = client.messages.create(
            model=config.llm_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return message.content[0].text

    except Exception as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e