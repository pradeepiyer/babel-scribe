import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".babel-scribe"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_TARGET_LANGUAGE = "en"
DEFAULT_CONCURRENCY = 5
DEFAULT_TRANSCRIPTION_MODEL = "groq/whisper-large-v3-turbo"
DEFAULT_TRANSLATION_MODEL = "groq/llama-3.3-70b-versatile"


@dataclass(frozen=True)
class Config:
    target_language: str = DEFAULT_TARGET_LANGUAGE
    concurrency: int = DEFAULT_CONCURRENCY
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    translation_model: str = DEFAULT_TRANSLATION_MODEL


def load_config(path: Path = CONFIG_FILE) -> Config:
    """Load configuration from TOML file, falling back to defaults for missing values."""
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    defaults = data.get("defaults", {})
    models = data.get("models", {})

    return Config(
        target_language=defaults.get("target_language", DEFAULT_TARGET_LANGUAGE),
        concurrency=defaults.get("concurrency", DEFAULT_CONCURRENCY),
        transcription_model=models.get("transcription", DEFAULT_TRANSCRIPTION_MODEL),
        translation_model=models.get("translation", DEFAULT_TRANSLATION_MODEL),
    )
