import os
from pathlib import Path

import pytest

from babel_scribe.transcriber import LitellmTranscriber

pytestmark = pytest.mark.integration


@pytest.fixture
def transcriber() -> LitellmTranscriber:
    return LitellmTranscriber()


@pytest.fixture
def require_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


@pytest.mark.usefixtures("require_api_key")
async def test_transcribe_local_file(transcriber: LitellmTranscriber, tmp_path: Path) -> None:
    # This test requires a real audio file and API key
    audio_files = list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
    if not audio_files:
        pytest.skip("No audio file available for testing")

    result = await transcriber.transcribe(audio_files[0])
    assert result.text
    assert isinstance(result.text, str)


@pytest.mark.usefixtures("require_api_key")
async def test_transcribe_with_timestamps(
    transcriber: LitellmTranscriber, tmp_path: Path
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
