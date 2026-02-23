from pathlib import Path

from babel_scribe.transcriber import Transcriber
from babel_scribe.translator import Translator
from babel_scribe.types import ScribeError, ScribeResult, TranslationResult


async def translate(
    text: str,
    translator: Translator,
    source_language: str,
    target_language: str,
) -> TranslationResult:
    translated_text = await translator.translate(text, source_language, target_language)
    return TranslationResult(
        text=translated_text,
        source_language=source_language,
        target_language=target_language,
    )


async def scribe(
    audio_path: Path,
    transcriber: Transcriber,
    translator: Translator | None,
    source_language: str | None = None,
    target_language: str = "en",
    timestamps: bool = False,
) -> ScribeResult:
    transcription = await transcriber.transcribe(audio_path, language=source_language, timestamps=timestamps)

    detected_language = transcription.source_language or source_language
    if detected_language and detected_language.lower() == target_language.lower():
        return ScribeResult(transcription=transcription)

    if translator is None:
        raise ScribeError("No translator configured but translation is required")

    translated_text = await translator.translate(
        transcription.text,
        source_language=detected_language or "auto",
        target_language=target_language,
    )

    translation = TranslationResult(
        text=translated_text,
        source_language=detected_language or "unknown",
        target_language=target_language,
    )
    return ScribeResult(transcription=transcription, translation=translation)
