import pytest

from babel_scribe.providers import (
    INDIAN_LANGUAGES,
    get_api_key,
    is_indian_language,
    normalize_language_code,
)
from babel_scribe.types import ScribeError


def test_indian_language_count() -> None:
    assert len(INDIAN_LANGUAGES) == 22


@pytest.mark.parametrize("code", ["hi", "bn", "ta", "te", "ml", "kn", "mr", "gu", "pa", "ur"])
def test_is_indian_language_recognized(code: str) -> None:
    assert is_indian_language(code)


@pytest.mark.parametrize("code", ["en", "es", "fr", "de", "zh", "ja", "ko", "pt"])
def test_is_indian_language_rejects_non_indian(code: str) -> None:
    assert not is_indian_language(code)


@pytest.mark.parametrize("code", ["HI", "Ta", "BN"])
def test_is_indian_language_case_insensitive(code: str) -> None:
    assert is_indian_language(code)


@pytest.mark.parametrize("code", ["hi-IN", "ta-Latn", "bn-BD"])
def test_is_indian_language_strips_region(code: str) -> None:
    assert is_indian_language(code)


def test_normalize_language_code_strips_subtag() -> None:
    assert normalize_language_code("pt-BR") == "pt"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("hi-IN") == "hi"


def test_normalize_language_code_lowercases() -> None:
    assert normalize_language_code("HI") == "hi"
    assert normalize_language_code("EN") == "en"


def test_normalize_language_code_preserves_base() -> None:
    assert normalize_language_code("pt") == "pt"
    assert normalize_language_code("hi") == "hi"


def test_get_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "my-secret")
    assert get_api_key("TEST_KEY") == "my-secret"


def test_get_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ScribeError, match=r"Missing API key.*TEST_KEY"):
        get_api_key("TEST_KEY")


def test_get_api_key_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "")
    with pytest.raises(ScribeError, match=r"Missing API key.*TEST_KEY"):
        get_api_key("TEST_KEY")
