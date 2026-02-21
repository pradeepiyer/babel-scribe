from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    source_language: str | None = None
    segments: list[Segment] | None = None


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source_language: str
    target_language: str


@dataclass(frozen=True)
class ScribeResult:
    transcription: TranscriptionResult
    translation: TranslationResult | None = None
