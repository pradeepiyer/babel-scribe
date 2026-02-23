import os
from unittest.mock import MagicMock, patch

import pytest

from babel_scribe.translator import (
    ChainedTranslator,
    ChatTranslator,
    SarvamTranslator,
    create_translator,
)
from babel_scribe.types import ScribeError

from .conftest import FakeTranslator

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


# --- Unit tests (no API keys required) ---


class TestCreateTranslator:
    def test_default_returns_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        t = create_translator("es", "fr")
        assert isinstance(t, ChatTranslator)

    def test_indian_to_english_returns_sarvam(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SARVAM_API_KEY", "test-key")
        t = create_translator("hi", "en")
        assert isinstance(t, SarvamTranslator)

    def test_english_to_indian_returns_sarvam(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SARVAM_API_KEY", "test-key")
        t = create_translator("en", "ta")
        assert isinstance(t, SarvamTranslator)

    def test_indian_to_non_english_returns_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SARVAM_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        t = create_translator("hi", "fr")
        assert isinstance(t, ChainedTranslator)
        assert isinstance(t.first, SarvamTranslator)
        assert isinstance(t.second, ChatTranslator)

    def test_non_english_to_indian_returns_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SARVAM_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        t = create_translator("fr", "hi")
        assert isinstance(t, ChainedTranslator)
        assert isinstance(t.first, ChatTranslator)
        assert isinstance(t.second, SarvamTranslator)

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(ScribeError, match="Missing API key"):
            create_translator("es", "fr")


class TestSarvamTranslator:
    async def test_translate_calls_sarvam_api(self) -> None:
        mock_response = MagicMock()
        mock_response.translated_text = "hello world"

        mock_text = MagicMock()
        mock_text.translate.return_value = mock_response

        mock_client = MagicMock()
        mock_client.text = mock_text

        with patch("babel_scribe.translator.SarvamTranslator._translate_sync", return_value="hello world") as mock_sync:
            translator = SarvamTranslator(api_key="test-key")
            result = await translator.translate("नमस्ते दुनिया", "hi", "en")

        assert result == "hello world"
        mock_sync.assert_called_once_with("नमस्ते दुनिया", "hi-IN", "en-IN")

    async def test_translate_maps_odia_code(self) -> None:
        with patch("babel_scribe.translator.SarvamTranslator._translate_sync", return_value="hello") as mock_sync:
            translator = SarvamTranslator(api_key="test-key")
            await translator.translate("text", "or", "en")

        mock_sync.assert_called_once_with("text", "od-IN", "en-IN")

    async def test_translate_wraps_exceptions(self) -> None:
        with patch(
            "babel_scribe.translator.SarvamTranslator._translate_sync",
            side_effect=RuntimeError("API failed"),
        ):
            translator = SarvamTranslator(api_key="test-key")
            with pytest.raises(ScribeError, match="API failed"):
                await translator.translate("text", "hi", "en")


class TestChainedTranslator:
    async def test_chains_two_translators(self) -> None:
        first = FakeTranslator(translated_text="intermediate english")
        second = FakeTranslator(translated_text="final french")

        chained = ChainedTranslator(first, second, intermediate="en")
        result = await chained.translate("हिंदी टेक्स्ट", "hi", "fr")

        assert result == "final french"
        assert first.calls == [("हिंदी टेक्स्ट", "hi", "en")]
        assert second.calls == [("intermediate english", "en", "fr")]

    async def test_passes_intermediate_language(self) -> None:
        first = FakeTranslator(translated_text="english text")
        second = FakeTranslator(translated_text="texte français")

        chained = ChainedTranslator(first, second, intermediate="en")
        await chained.translate("texto español", "es", "fr")

        # First translator: source→intermediate
        assert first.calls[0][2] == "en"
        # Second translator: intermediate→target
        assert second.calls[0][1] == "en"
