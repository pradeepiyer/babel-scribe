from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from babel_scribe.errors import ScribeError

TRANSIENT_ERRORS = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)

api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    reraise=True,
)


@contextmanager
def handle_api_errors(error_class: type[ScribeError]) -> Iterator[None]:
    try:
        yield
    except TRANSIENT_ERRORS:
        raise
    except openai.OpenAIError as e:
        raise error_class(str(e)) from e


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str


PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "sarvam": ProviderConfig(
        base_url="",
        api_key_env="SARVAM_API_KEY",
    ),
}


def parse_model(model: str) -> tuple[ProviderConfig, str]:
    """Split 'groq/whisper-large-v3-turbo' into (ProviderConfig, 'whisper-large-v3-turbo')."""
    prefix, sep, model_name = model.partition("/")
    if not sep or prefix not in PROVIDERS:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown provider in model '{model}'. Known providers: {known}")
    return PROVIDERS[prefix], model_name
