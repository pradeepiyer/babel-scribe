import os
from pathlib import Path

import pytest

from babel_scribe.types import Segment, TranscriptionResult


@pytest.fixture
def require_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


class FakeTranscriber:
    def __init__(
        self,
        text: str = "transcribed text",
        source_language: str | None = None,
        segments: list[Segment] | None = None,
    ) -> None:
        self.text = text
        self.source_language = source_language
        self.segments = segments
        self.call_count = 0
        self.calls: list[tuple[Path, str | None, bool]] = []

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult:
        self.call_count += 1
        self.calls.append((audio_path, language, timestamps))
        return TranscriptionResult(
            text=self.text,
            source_language=self.source_language,
            segments=self.segments if timestamps else None,
        )


class FakeTranslator:
    def __init__(self, translated_text: str = "translated text") -> None:
        self.translated_text = translated_text
        self.call_count = 0
        self.calls: list[tuple[str, str, str]] = []

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.call_count += 1
        self.calls.append((text, source_language, target_language))
        return self.translated_text
