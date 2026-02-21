import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import openai

from babel_scribe.errors import TranscriptionError
from babel_scribe.providers import PROVIDERS, api_retry, handle_api_errors, parse_model
from babel_scribe.types import Segment, TranscriptionResult


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        timestamps: bool = False,
    ) -> TranscriptionResult: ...


class WhisperTranscriber:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
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
        if language is not None:
            kwargs["language"] = language
        if timestamps:
            kwargs["response_format"] = "verbose_json"

        with handle_api_errors(TranscriptionError):
            response = await self._client.audio.transcriptions.create(**kwargs)

        if timestamps:
            # verbose_json returns a Transcription object with extra fields
            text: str = response.text
            source_language: str | None = getattr(response, "language", None)
            raw_segments = getattr(response, "segments", None)
            segments: list[Segment] | None = None
            if raw_segments is not None:
                segments = [
                    Segment(
                        text=seg.text,
                        start=float(seg.start),
                        end=float(seg.end),
                    )
                    for seg in raw_segments
                ]
            return TranscriptionResult(
                text=text, source_language=source_language, segments=segments
            )

        return TranscriptionResult(text=response.text)


class SarvamTranscriber:
    """Transcriber using Sarvam AI's batch speech-to-text API.

    When target_language is "en", uses mode="translate" to get English output
    in a single API call (no separate translation step needed).
    """

    def __init__(
        self, model: str, api_key: str, target_language: str = "en", job_timeout: int = 1800
    ) -> None:
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
        lang_code = f"{language}-IN" if language else "unknown"

        try:
            return await asyncio.to_thread(
                self._run_batch_job, audio_path, mode, lang_code, timestamps
            )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(str(e)) from e

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
                raise TranscriptionError("No output from Sarvam batch job")
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


def create_transcriber(
    model: str, target_language: str = "en", job_timeout: int = 1800
) -> Transcriber:
    provider, model_name = parse_model(model)
    api_key = os.environ.get(provider.api_key_env, "")
    if provider is PROVIDERS["sarvam"]:
        return SarvamTranscriber(
            model=model_name, api_key=api_key, target_language=target_language,
            job_timeout=job_timeout,
        )
    return WhisperTranscriber(model=model_name, base_url=provider.base_url, api_key=api_key)
