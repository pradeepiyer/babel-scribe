import os
from pathlib import Path

import pytest

from babel_scribe.transcriber import SarvamTranscriber, WhisperTranscriber, create_transcriber
from babel_scribe.types import ScribeError

pytestmark = pytest.mark.integration


@pytest.fixture
def transcriber() -> WhisperTranscriber:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return WhisperTranscriber(
        model="whisper-1",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
    )


@pytest.mark.usefixtures("require_api_key")
async def test_transcribe_local_file(transcriber: WhisperTranscriber, tmp_path: Path) -> None:
    # This test requires a real audio file and API key
    audio_files = list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
    if not audio_files:
        pytest.skip("No audio file available for testing")

    result = await transcriber.transcribe(audio_files[0])
    assert result.text
    assert isinstance(result.text, str)


@pytest.mark.usefixtures("require_api_key")
async def test_transcribe_with_timestamps(transcriber: WhisperTranscriber, tmp_path: Path) -> None:
    audio_files = list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
    if not audio_files:
        pytest.skip("No audio file available for testing")

    result = await transcriber.transcribe(audio_files[0], timestamps=True)
    assert result.text
    assert result.segments is not None
    assert len(result.segments) > 0
    for seg in result.segments:
        assert seg.text
        assert seg.start >= 0
        assert seg.end >= seg.start


def test_create_transcriber_indian_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    t = create_transcriber("hi")
    assert isinstance(t, SarvamTranscriber)


def test_create_transcriber_non_indian_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    t = create_transcriber("es")
    assert isinstance(t, WhisperTranscriber)


def test_create_transcriber_passes_target_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    t = create_transcriber("es", target_language="fr")
    assert isinstance(t, WhisperTranscriber)
    assert t.target_language == "fr"


def test_create_transcriber_normalizes_target_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    t = create_transcriber("es", target_language="pt-BR")
    assert isinstance(t, WhisperTranscriber)
    assert t.target_language == "pt"


def test_create_transcriber_passes_job_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    t = create_transcriber("hi", job_timeout=3600)
    assert isinstance(t, SarvamTranscriber)
    assert t.job_timeout == 3600


def test_create_transcriber_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ScribeError, match="Missing API key"):
        create_transcriber("es")


def test_create_transcriber_bcp47_indian(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    t = create_transcriber("hi-IN")
    assert isinstance(t, SarvamTranscriber)


def test_create_transcriber_bcp47_non_indian(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    t = create_transcriber("en-US")
    assert isinstance(t, WhisperTranscriber)
