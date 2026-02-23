import os

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from babel_scribe.types import ScribeError

TRANSIENT_ERRORS = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)

api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    reraise=True,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"

# ISO 639-1 codes for languages supported by Sarvam AI
INDIAN_LANGUAGES: frozenset[str] = frozenset(
    {
        "as",  # Assamese
        "bn",  # Bengali
        "brx",  # Bodo
        "doi",  # Dogri
        "gu",  # Gujarati
        "hi",  # Hindi
        "kn",  # Kannada
        "kok",  # Konkani
        "ks",  # Kashmiri
        "mai",  # Maithili
        "ml",  # Malayalam
        "mni",  # Manipuri
        "mr",  # Marathi
        "ne",  # Nepali
        "or",  # Odia
        "pa",  # Punjabi
        "sa",  # Sanskrit
        "sat",  # Santali
        "sd",  # Sindhi
        "ta",  # Tamil
        "te",  # Telugu
        "ur",  # Urdu
    }
)


def normalize_language_code(code: str) -> str:
    """Strip region/script subtags for routing purposes. e.g. 'hi-IN' -> 'hi', 'pt-BR' -> 'pt'."""
    return code.split("-")[0].lower()


def is_indian_language(code: str) -> bool:
    """Check if a language code refers to an Indian language supported by Sarvam."""
    return normalize_language_code(code) in INDIAN_LANGUAGES


def to_sarvam_language_code(code: str) -> str:
    """Convert an ISO 639 code to the Sarvam AI language code format (e.g. 'hi' → 'hi-IN').

    Odia needs special handling: ISO 639-1 uses 'or' but Sarvam uses 'od'.
    """
    base = normalize_language_code(code)
    if base == "en":
        return "en-IN"
    if base == "or":
        return "od-IN"
    return f"{base}-IN"


def get_api_key(env_var: str) -> str:
    """Read an API key from the environment, raising ScribeError if missing."""
    key = os.environ.get(env_var, "")
    if not key:
        raise ScribeError(f"Missing API key: set the {env_var} environment variable")
    return key
