import os
from typing import Protocol, runtime_checkable

import openai

from babel_scribe.errors import TranslationError
from babel_scribe.providers import api_retry, handle_api_errors, parse_model


@runtime_checkable
class Translator(Protocol):
    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> str: ...


class ChatTranslator:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    @api_retry
    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"You are a translator. Translate the following text from "
                    f"{source_language} to {target_language}. "
                    f"Output only the translated text, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ]

        with handle_api_errors(TranslationError):
            response = await self._client.chat.completions.create(
                model=self.model, messages=messages  # type: ignore[arg-type]
            )

        return response.choices[0].message.content or ""


def create_translator(model: str) -> Translator:
    provider, model_name = parse_model(model)
    api_key = os.environ.get(provider.api_key_env, "")
    return ChatTranslator(model=model_name, base_url=provider.base_url, api_key=api_key)
