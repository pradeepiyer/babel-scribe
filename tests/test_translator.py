import os

import pytest

from babel_scribe.translator import ChatTranslator, create_translator

pytestmark = pytest.mark.integration


@pytest.fixture
def translator() -> ChatTranslator:
    api_key = os.environ.get("GROQ_API_KEY", "")
    return ChatTranslator(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
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


def test_create_translator_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    t = create_translator("groq/llama-3.3-70b-versatile")
    assert isinstance(t, ChatTranslator)


def test_create_translator_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        create_translator("unknown/model")
