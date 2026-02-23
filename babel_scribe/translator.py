import asyncio
from typing import Protocol

import openai

from babel_scribe.providers import (
    OPENAI_BASE_URL,
    TRANSIENT_ERRORS,
    api_retry,
    get_api_key,
    is_indian_language,
    normalize_language_code,
    to_sarvam_language_code,
)
from babel_scribe.types import ScribeError


class Translator(Protocol):
    async def translate(self, text: str, source_language: str, target_language: str) -> str: ...


class ChatTranslator:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    @api_retry
    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        system_content = (
            f"You are a translator. Translate the following text from {source_language} to {target_language}."
            " Output only the translated text, nothing else."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
            )
        except TRANSIENT_ERRORS:
            raise
        except openai.OpenAIError as e:
            raise ScribeError(str(e)) from e

        return response.choices[0].message.content or ""


_SARVAM_MODEL = "sarvam-translate:v1"
_SARVAM_MAX_CHARS = 1900  # API limit is 2000; leave headroom


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks that fit within max_chars, breaking at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # +2 for the "\n\n" separator between paragraphs
        sep_len = 2 if current else 0
        if current_len + sep_len + len(para) <= max_chars:
            current.append(para)
            current_len += sep_len + len(para)
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


class SarvamTranslator:
    """Translates text using the Sarvam AI text translation API."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        src_code = to_sarvam_language_code(source_language)
        tgt_code = to_sarvam_language_code(target_language)

        try:
            return await asyncio.to_thread(self._translate_sync, text, src_code, tgt_code)
        except ScribeError:
            raise
        except Exception as e:
            raise ScribeError(str(e)) from e

    def _translate_sync(self, text: str, source_language_code: str, target_language_code: str) -> str:
        from sarvamai import SarvamAI

        client = SarvamAI(api_subscription_key=self.api_key)
        chunks = _split_text(text, max_chars=_SARVAM_MAX_CHARS)
        translated_chunks = []
        for chunk in chunks:
            response = client.text.translate(
                input=chunk,
                source_language_code=source_language_code,  # type: ignore[arg-type]
                target_language_code=target_language_code,  # type: ignore[arg-type]
                model=_SARVAM_MODEL,  # type: ignore[arg-type]
            )
            translated_chunks.append(response.translated_text)
        return "\n\n".join(translated_chunks)


class ChainedTranslator:
    """Chains two translators via an intermediate language for indirect translation paths."""

    def __init__(self, first: Translator, second: Translator, intermediate: str) -> None:
        self.first = first
        self.second = second
        self.intermediate = intermediate

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        intermediate_text = await self.first.translate(text, source_language, self.intermediate)
        return await self.second.translate(intermediate_text, self.intermediate, target_language)


TRANSLATION_MODEL = "gpt-5-mini"


def create_translator(source_language: str, target_language: str) -> Translator:
    src_indian = is_indian_language(source_language)
    tgt_indian = is_indian_language(target_language)
    src_english = normalize_language_code(source_language) == "en"
    tgt_english = normalize_language_code(target_language) == "en"

    def sarvam() -> SarvamTranslator:
        return SarvamTranslator(api_key=get_api_key("SARVAM_API_KEY"))

    def chat() -> ChatTranslator:
        return ChatTranslator(model=TRANSLATION_MODEL, base_url=OPENAI_BASE_URL, api_key=get_api_key("OPENAI_API_KEY"))

    # Indian↔English: Sarvam directly
    if (src_indian and tgt_english) or (tgt_indian and src_english):
        return sarvam()
    # Indian→non-English: chain Sarvam (Indian→English) + Chat (English→target)
    if src_indian:
        return ChainedTranslator(sarvam(), chat(), intermediate="en")
    # non-English→Indian: chain Chat (source→English) + Sarvam (English→Indian)
    if tgt_indian:
        return ChainedTranslator(chat(), sarvam(), intermediate="en")
    # Default: LLM-based translation
    return chat()
