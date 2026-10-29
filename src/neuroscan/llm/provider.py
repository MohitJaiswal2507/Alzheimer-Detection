"""
NeuroScan AI — LLM Provider Client
Handles communication with the configured Google Gemini API via HTTP requests.
Never transmits MRI images or personal health data.
"""
import os
import requests
from ..config import DEFAULT_LLM_PROVIDER, DEFAULT_GEMINI_MODEL


def call_llm(
    prompt: str,
    system_prompt: str = "",
    provider: str = DEFAULT_LLM_PROVIDER,
    timeout: int = 35,
) -> str:
    """
    Sends a text-only prompt to the configured LLM provider and returns the generated text.

    Args:
        prompt: The user-level instructional text containing structured classifier results.
        system_prompt: Optional system-level instructions or role constraints.
        provider: Provider name (defaults to "gemini").
        timeout: Request timeout in seconds.

    Returns:
        The generated explanation string.

    Raises:
        ValueError: If required API keys are missing or provider is unsupported.
        RuntimeError: If the API request fails, times out, or returns an invalid payload.
    """
    selected_provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).lower().strip()

    if selected_provider != "gemini":
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. "
            "NeuroScan AI is configured for 'gemini'."
        )

    return _call_gemini(prompt, system_prompt, timeout)


def _call_gemini(prompt: str, system_prompt: str, timeout: int) -> str:
    """Calls Google Gemini generateContent REST endpoint using requests."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "Gemini API key is not configured. "
            "Please set GEMINI_API_KEY in your environment or .env file."
        )

    # Sanitize model name (strips accidental prefixes, quotes, models/ path, whitespace)
    model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip().strip("'\"")
    if model.startswith("GEMINI_MODEL="):
        model = model.split("=", 1)[1].strip().strip("'\"")
    if model.startswith("models/"):
        model = model[len("models/"):]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Combine system prompt with user prompt for clean Gemini compatibility
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    payload = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key.strip(),
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Gemini API request timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error communicating with Gemini API: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error (HTTP {response.status_code}): {response.text}"
        )

    try:
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned an empty response (no candidates).")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts or not parts[0].get("text"):
            raise RuntimeError("Gemini API candidate contained no text content.")

        return parts[0]["text"].strip()
    except (ValueError, KeyError, IndexError) as e:
        raise RuntimeError(f"Failed to parse Gemini API JSON response: {e}")
