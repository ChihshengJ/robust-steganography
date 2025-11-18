import random
import re
from datetime import datetime

from openai import OpenAI

from .attack import Attack

SYSTEM_PROMPT = """
You are a language expert at {language_1} and {language_2}, and you will be assigned with translation tasks that either require you to translate a text from {language_1} to {language_2}.
You must make sure that your translation contrains every information in the original text including the events, the tone or even the style. 
Your output should only contain your translation and nothing else.
"""


class TranslationAttack(Attack):
    """Attack that uses GPT to translate text from English to other language and then back to English."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        language: str = "French",
    ):
        """
        Initialize the paraphrase attack.

        Args:
            client: OpenAI client instance
            model: GPT model to use (default: "gpt-4o-mini")
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            local: If True, paraphrases each sentence independently.
                  If False, paraphrases entire text at once (default: True)
            language: the medium language used
        """
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.language = language

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the paraphrase attack."""
        if local and tampering < 0.99:
            return self._local_attack(text, tampering)
        else:
            return self._global_attack(text)

    def _translate(self, text: str, direction: bool) -> str:
        """Direction is true when translating from English to other languages."""
        lang_1, lang_2 = (
            ("English", self.language) if direction else (self.language, "English")
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        + SYSTEM_PROMPT.format(language_1=lang_1, language_2=lang_2),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
            )
            result = response.choices[0].message.content.strip()

            return result
        except Exception as e:
            print(f"Global paraphrase attack failed: {e}")
            return text

    def _global_attack(self, text: str) -> str:
        """Translate entire text at once."""
        result = self._translate(text, True)
        result = self._translate(result, False)
        return result

    def _local_attack(self, text: str, tampering: float) -> str:
        """Translate each sentence independently while preserving structure."""
        # Split text into sentences while preserving separators
        parts = re.split(r"([.!?]+(?:\s+|$))", text)
        new_parts = []

        # parts[::2] are sentences, parts[1::2] are separators
        for i in range(0, len(parts), 2):
            sentence = parts[i]

            # Skip empty sentences
            if not sentence.strip():
                new_parts.append(sentence)
            if random.random() < tampering:
                result = self._translate(sentence, True)
                result = self._translate(result, False)
                new_parts.append(result)
            else:
                new_parts.append(sentence)

            # Add the separator if it exists
            if i + 1 < len(parts):
                new_parts.append(parts[i + 1])

        result = "".join(new_parts)
        # print("Debug local paraphrase:")
        # print(f"parts:\n{parts}\nnew_parts:\n{result}")
        return result
