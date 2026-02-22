from typing import Protocol

import openai

from babel_scribe.providers import OPENAI_BASE_URL, TRANSIENT_ERRORS, api_retry, get_api_key
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


TRANSLATION_MODEL = "gpt-5-mini"


def create_translator() -> Translator:
    return ChatTranslator(model=TRANSLATION_MODEL, base_url=OPENAI_BASE_URL, api_key=get_api_key("OPENAI_API_KEY"))
