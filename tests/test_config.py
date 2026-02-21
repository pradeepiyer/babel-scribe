from pathlib import Path

from babel_scribe.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_JOB_TIMEOUT,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    Config,
    load_config,
)


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_config(tmp_path / "nonexistent.toml")
    assert config == Config()
    assert config.target_language == DEFAULT_TARGET_LANGUAGE
    assert config.concurrency == DEFAULT_CONCURRENCY
    assert config.job_timeout == DEFAULT_JOB_TIMEOUT
    assert config.transcription_model == DEFAULT_TRANSCRIPTION_MODEL
    assert config.translation_model == DEFAULT_TRANSLATION_MODEL


def test_loads_all_values(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("""\
[defaults]
target_language = "es"
concurrency = 10
job_timeout = 3600

[models]
transcription = "openai/whisper-1"
translation = "openai/gpt-4o"
""")

    config = load_config(config_file)
    assert config.target_language == "es"
    assert config.concurrency == 10
    assert config.job_timeout == 3600
    assert config.transcription_model == "openai/whisper-1"
    assert config.translation_model == "openai/gpt-4o"


def test_partial_config_uses_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("""\
[defaults]
target_language = "fr"
""")

    config = load_config(config_file)
    assert config.target_language == "fr"
    assert config.concurrency == DEFAULT_CONCURRENCY
    assert config.job_timeout == DEFAULT_JOB_TIMEOUT
    assert config.transcription_model == DEFAULT_TRANSCRIPTION_MODEL
    assert config.translation_model == DEFAULT_TRANSLATION_MODEL


def test_empty_config_uses_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    config = load_config(config_file)
    assert config == Config()
