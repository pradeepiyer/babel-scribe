import os
from typing import Protocol, runtime_checkable

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from babel_scribe.errors import TranslationError
from babel_scribe.providers import parse_model


@runtime_checkable
class Translator(Protocol):
    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> str: ...


class ChatTranslator:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        reraise=True,
    )
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

        try:
            response = await self._client.chat.completions.create(
                model=self.model, messages=messages  # type: ignore[arg-type]
            )
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError):
            raise
        except openai.OpenAIError as e:
            raise TranslationError(str(e)) from e

        return response.choices[0].message.content or ""


def create_translator(model: str) -> Translator:
    provider, model_name = parse_model(model)
    api_key = os.environ.get(provider.api_key_env, "")
    return ChatTranslator(model=model_name, base_url=provider.base_url, api_key=api_key)
