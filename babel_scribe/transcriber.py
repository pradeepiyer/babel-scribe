import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Protocol

import openai

from babel_scribe.providers import (
    OPENAI_BASE_URL,
    TRANSIENT_ERRORS,
    api_retry,
    get_api_key,
    is_indian_language,
    normalize_language_code,
)
from babel_scribe.types import ScribeError, Segment, TranscriptionResult


class Transcriber(Protocol):
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult: ...


class WhisperTranscriber:
    def __init__(self, model: str, base_url: str, api_key: str, target_language: str = "en") -> None:
        self.model = model
        self.target_language = target_language
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    @api_retry
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "file": audio_path,
        }
        if timestamps:
            kwargs["response_format"] = "verbose_json"

        translate_directly = self.target_language.lower() == "en"

        if not translate_directly and language is not None:
            kwargs["language"] = language

        try:
            if translate_directly:
                response = await self._client.audio.translations.create(**kwargs)
            else:
                response = await self._client.audio.transcriptions.create(**kwargs)
        except TRANSIENT_ERRORS:
            raise
        except openai.OpenAIError as e:
            raise ScribeError(str(e)) from e

        if timestamps:
            # verbose_json returns a Transcription object with extra fields
            text: str = response.text
            source_language: str | None = "en" if translate_directly else getattr(response, "language", None)
            raw_segments = getattr(response, "segments", None)
            segments: list[Segment] | None = None
            if raw_segments is not None:
                segments = [Segment(text=seg.text, start=float(seg.start), end=float(seg.end)) for seg in raw_segments]
            return TranscriptionResult(text=text, source_language=source_language, segments=segments)

        return TranscriptionResult(
            text=response.text,
            source_language="en" if translate_directly else None,
        )


class SarvamTranscriber:
    """Sarvam AI batch speech-to-text; uses translate mode for English targets."""

    def __init__(self, model: str, api_key: str, target_language: str = "en", job_timeout: int = 1800) -> None:
        self.model = model
        self.api_key = api_key
        self.target_language = target_language
        self.job_timeout = job_timeout

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult:
        mode = "translate" if self.target_language == "en" else "transcribe"
        lang_code = f"{normalize_language_code(language)}-IN" if language else "unknown"

        try:
            return await asyncio.to_thread(self._run_batch_job, audio_path, mode, lang_code, timestamps)
        except ScribeError:
            raise
        except Exception as e:
            raise ScribeError(str(e)) from e

    def _run_batch_job(
        self,
        audio_path: Path,
        mode: str,
        lang_code: str,
        timestamps: bool,
    ) -> TranscriptionResult:
        from sarvamai import SarvamAI

        client = SarvamAI(api_subscription_key=self.api_key)
        job = client.speech_to_text_job.create_job(
            model=self.model,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            with_diarization=True,
            with_timestamps=timestamps,
            language_code=lang_code,  # type: ignore[arg-type]
        )
        job.upload_files([str(audio_path)])
        job.start()
        job.wait_until_complete(timeout=self.job_timeout)

        with tempfile.TemporaryDirectory() as tmp_dir:
            job.download_outputs(tmp_dir)
            json_files = list(Path(tmp_dir).glob("*.json"))
            if not json_files:
                raise ScribeError("No output from Sarvam batch job")
            data = json.loads(json_files[0].read_text())

        return self._parse_response(data, mode)

    def _parse_response(self, data: dict[str, Any], mode: str) -> TranscriptionResult:
        text: str = data.get("transcript", "")

        # When mode="translate", the text is already English.
        # Setting source_language="en" causes the pipeline to skip the translation step.
        source_language: str | None = "en" if mode == "translate" else data.get("language_code")

        segments: list[Segment] | None = None
        diarized = data.get("diarized_transcript")
        if diarized and diarized.get("entries"):
            segments = [
                Segment(
                    text=entry["transcript"],
                    start=float(entry["start_time_seconds"]),
                    end=float(entry["end_time_seconds"]),
                    speaker=entry.get("speaker_id"),
                )
                for entry in diarized["entries"]
            ]

        return TranscriptionResult(text=text, source_language=source_language, segments=segments)


WHISPER_MODEL = "whisper-1"
SARVAM_MODEL = "saaras:v3"


def create_transcriber(source_language: str, target_language: str = "en", job_timeout: int = 1800) -> Transcriber:
    """Select and configure a transcriber based on source language."""
    normalized_target = normalize_language_code(target_language)

    if is_indian_language(source_language):
        return SarvamTranscriber(
            model=SARVAM_MODEL,
            api_key=get_api_key("SARVAM_API_KEY"),
            target_language=normalized_target,
            job_timeout=job_timeout,
        )
    return WhisperTranscriber(
        model=WHISPER_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=get_api_key("OPENAI_API_KEY"),
        target_language=normalized_target,
    )
