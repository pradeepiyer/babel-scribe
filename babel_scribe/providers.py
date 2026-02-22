import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

TRANSIENT_ERRORS = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)

api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    reraise=True,
)

# (base_url, api_key_env)
PROVIDERS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "sarvam": ("", "SARVAM_API_KEY"),
}


def parse_model(model: str) -> tuple[str, str, str]:
    """Split 'groq/whisper-large-v3-turbo' into (base_url, api_key_env, model_name)."""
    prefix, sep, model_name = model.partition("/")
    if not sep or prefix not in PROVIDERS:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown provider in model '{model}'. Known providers: {known}")
    base_url, api_key_env = PROVIDERS[prefix]
    return base_url, api_key_env, model_name
