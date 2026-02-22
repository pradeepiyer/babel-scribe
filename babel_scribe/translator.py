import os
from typing import Protocol

import openai

from babel_scribe.providers import TRANSIENT_ERRORS, api_retry, parse_model
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


def create_translator(model: str) -> Translator:
    base_url, api_key_env, model_name = parse_model(model)
    api_key = os.environ.get(api_key_env, "")
    return ChatTranslator(model=model_name, base_url=base_url, api_key=api_key)
