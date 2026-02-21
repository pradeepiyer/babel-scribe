import os
from pathlib import Path

import pytest

from babel_scribe.transcriber import WhisperTranscriber, create_transcriber

pytestmark = pytest.mark.integration


@pytest.fixture
def transcriber() -> WhisperTranscriber:
    api_key = os.environ.get("GROQ_API_KEY", "")
    return WhisperTranscriber(
        model="whisper-large-v3-turbo",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


@pytest.fixture
def require_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


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
async def test_transcribe_with_timestamps(
    transcriber: WhisperTranscriber, tmp_path: Path
) -> None:
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


def test_create_transcriber_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    t = create_transcriber("groq/whisper-large-v3-turbo")
    assert isinstance(t, WhisperTranscriber)


def test_create_transcriber_sarvam(monkeypatch: pytest.MonkeyPatch) -> None:
    from babel_scribe.transcriber import SarvamTranscriber

    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    t = create_transcriber("sarvam/saaras:v3")
    assert isinstance(t, SarvamTranscriber)
    assert t.job_timeout == 1800


def test_create_transcriber_sarvam_custom_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from babel_scribe.transcriber import SarvamTranscriber

    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    t = create_transcriber("sarvam/saaras:v3", job_timeout=3600)
    assert isinstance(t, SarvamTranscriber)
    assert t.job_timeout == 3600


def test_create_transcriber_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        create_transcriber("unknown/model")
