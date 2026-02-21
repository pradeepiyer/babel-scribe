from dataclasses import dataclass


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
