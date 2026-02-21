from typing import Protocol, runtime_checkable

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from babel_scribe.config import DEFAULT_TRANSLATION_MODEL
from babel_scribe.errors import TranslationError


@runtime_checkable
class Translator(Protocol):
    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> str: ...


class LitellmTranslator:
    def __init__(self, model: str = DEFAULT_TRANSLATION_MODEL) -> None:
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(
            (litellm.RateLimitError, litellm.Timeout, litellm.ServiceUnavailableError)
        ),
        reraise=True,
    )
    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        messages = [
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
            response = await litellm.acompletion(model=self.model, messages=messages)
        except (litellm.RateLimitError, litellm.Timeout, litellm.ServiceUnavailableError):
            raise
        except Exception as e:
            raise TranslationError(str(e)) from e

        return response.choices[0].message.content or ""  # type: ignore[union-attr]
