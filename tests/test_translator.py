import os

import pytest

from babel_scribe.translator import ChatTranslator, create_translator
from babel_scribe.types import ScribeError

pytestmark = pytest.mark.integration


@pytest.fixture
def translator() -> ChatTranslator:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return ChatTranslator(
        model="gpt-5-mini",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
    )


@pytest.mark.usefixtures("require_api_key")
async def test_translate_spanish_to_english(translator: ChatTranslator) -> None:
    result = await translator.translate("Hola, mundo", "es", "en")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "hello" in result.lower()


@pytest.mark.usefixtures("require_api_key")
async def test_translate_preserves_meaning(translator: ChatTranslator) -> None:
    result = await translator.translate("Bonjour le monde", "fr", "en")
    assert isinstance(result, str)
    assert "hello" in result.lower() or "good" in result.lower()


def test_create_translator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    t = create_translator()
    assert isinstance(t, ChatTranslator)


def test_create_translator_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ScribeError, match="Missing API key"):
        create_translator()
