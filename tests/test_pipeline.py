import asyncio
from pathlib import Path

from babel_scribe.pipeline import scribe, scribe_batch
from babel_scribe.types import Segment, TranscriptionResult

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

    result = await scribe(
        audio, transcriber, translator, source_language="en", target_language="en", timestamps=True
    )

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


async def test_scribe_batch_processes_all_files(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        f = tmp_path / f"test{i}.mp3"
        f.touch()
        files.append(f)

    transcriber = FakeTranscriber(text="hola", source_language="es")
    translator = FakeTranslator(translated_text="hello")

    results = await scribe_batch(files, transcriber, translator, target_language="en")

    assert len(results) == 3
    assert transcriber.call_count == 3
    assert translator.call_count == 3


async def test_scribe_batch_respects_concurrency(tmp_path: Path) -> None:
    files = []
    for i in range(10):
        f = tmp_path / f"test{i}.mp3"
        f.touch()
        files.append(f)

    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    class TrackingTranscriber:
        async def transcribe(
            self,
            audio_path: Path,
            language: str | None = None,
            timestamps: bool = False,
        ) -> "TranscriptionResult":
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return TranscriptionResult(text="text", source_language="es")

    transcriber = TrackingTranscriber()
    translator = FakeTranslator(translated_text="translated")

    results = await scribe_batch(
        files, transcriber, translator, target_language="en", concurrency=3  # type: ignore[arg-type]
    )

    assert len(results) == 10
    assert max_concurrent <= 3


async def test_scribe_batch_empty_list() -> None:
    transcriber = FakeTranscriber()
    translator = FakeTranslator()

    results = await scribe_batch([], transcriber, translator)

    assert results == []
