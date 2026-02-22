from pathlib import Path

import pytest

from babel_scribe.pipeline import scribe
from babel_scribe.types import ScribeError, Segment

from .conftest import FakeTranscriber, FakeTranslator


async def test_scribe_transcribes_and_translates(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    transcriber = FakeTranscriber(text="hola mundo", source_language="es")
    translator = FakeTranslator(translated_text="hello world")

    result = await scribe(audio, transcriber, translator, target_language="en")

    assert result.transcription.text == "hola mundo"
    assert result.translation is not None
    assert result.translation.text == "hello world"
    assert result.translation.source_language == "es"
    assert result.translation.target_language == "en"


async def test_scribe_skips_translation_when_same_language(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    transcriber = FakeTranscriber(text="hello world", source_language="en")
    translator = FakeTranslator()

    result = await scribe(audio, transcriber, translator, target_language="en")

    assert result.transcription.text == "hello world"
    assert result.translation is None
    assert translator.call_count == 0


async def test_scribe_skips_translation_case_insensitive(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    transcriber = FakeTranscriber(text="hello", source_language="EN")
    translator = FakeTranslator()

    result = await scribe(audio, transcriber, translator, target_language="en")

    assert result.translation is None
    assert translator.call_count == 0


async def test_scribe_passes_language_and_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    segments = [Segment(text="hello", start=0.0, end=1.0)]
    transcriber = FakeTranscriber(text="hello", source_language="en", segments=segments)
    translator = FakeTranslator()

    result = await scribe(audio, transcriber, translator, source_language="en", target_language="en", timestamps=True)

    assert transcriber.calls[0] == (audio, "en", True)
    assert result.transcription.segments == segments


async def test_scribe_uses_auto_when_no_language_detected(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    transcriber = FakeTranscriber(text="some text", source_language=None)
    translator = FakeTranslator(translated_text="translated")

    result = await scribe(audio, transcriber, translator, target_language="en")

    assert result.translation is not None
    assert translator.calls[0][1] == "auto"


async def test_scribe_with_no_translator(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    transcriber = FakeTranscriber(text="hello world", source_language="en")

    result = await scribe(audio, transcriber, None, target_language="en")

    assert result.transcription.text == "hello world"
    assert result.translation is None


async def test_scribe_raises_when_translator_missing(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.touch()
    transcriber = FakeTranscriber(text="hola mundo", source_language="es")

    with pytest.raises(ScribeError, match="No translator configured"):
        await scribe(audio, transcriber, None, target_language="en")
