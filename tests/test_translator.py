import os

import pytest

from babel_scribe.translator import LitellmTranslator

pytestmark = pytest.mark.integration


@pytest.fixture
def translator() -> LitellmTranslator:
    return LitellmTranslator()


@pytest.fixture
def require_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


@pytest.mark.usefixtures("require_api_key")
async def test_translate_spanish_to_english(translator: LitellmTranslator) -> None:
    result = await translator.translate("Hola, mundo", "es", "en")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "hello" in result.lower()


@pytest.mark.usefixtures("require_api_key")
async def test_translate_preserves_meaning(translator: LitellmTranslator) -> None:
    result = await translator.translate("Bonjour le monde", "fr", "en")
    assert isinstance(result, str)
    assert "hello" in result.lower() or "good" in result.lower()
