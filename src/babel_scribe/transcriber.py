from pathlib import Path
from typing import Protocol, runtime_checkable

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from babel_scribe.config import DEFAULT_TRANSCRIPTION_MODEL
from babel_scribe.errors import TranscriptionError
from babel_scribe.types import Segment, TranscriptionResult


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult: ...


class LitellmTranscriber:
    def __init__(self, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> None:
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(
            (litellm.RateLimitError, litellm.Timeout, litellm.ServiceUnavailableError)
        ),
        reraise=True,
    )
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult:
        kwargs: dict[str, object] = {
            "model": self.model,
            "file": audio_path,
        }
        if language is not None:
            kwargs["language"] = language
        if timestamps:
            kwargs["response_format"] = "verbose_json"

        try:
            response = await litellm.atranscription(**kwargs)  # type: ignore[arg-type]
        except (litellm.RateLimitError, litellm.Timeout, litellm.ServiceUnavailableError):
            raise
        except Exception as e:
            raise TranscriptionError(str(e)) from e

        text: str = response.text or ""
        source_language: str | None = None
        segments: list[Segment] | None = None

        if timestamps:
            source_language = getattr(response, "language", None)
            raw_segments = getattr(response, "segments", None)
            if raw_segments is not None:
                segments = [
                    Segment(
                        text=seg.get("text", ""),
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                    )
                    for seg in raw_segments
                ]

        return TranscriptionResult(
            text=text,
            source_language=source_language,
            segments=segments,
        )
