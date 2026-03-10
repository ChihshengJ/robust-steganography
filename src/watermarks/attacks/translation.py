import random
import re

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
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        language: str = "French",
    ):
        """
        Initialize the translation attack.

        Arguments:
            client: OpenAI client instance
            model: GPT model to use (default: "gpt-4o-mini")
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            language: the medium language used for back-translation
        """
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.language = language

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the translation attack."""
        if not 0 <= tampering <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if tampering == 0:
            return text

        if self._resolve_local_mode(local, tampering):
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
                        "content": SYSTEM_PROMPT.format(
                            language_1=lang_1, language_2=lang_2
                        ),
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
        """Translate each sentence independently while preserving structure.

        Arguments:
            text: Input text to attack
            tampering: Probability (0.0-1.0) of translating each sentence

        Returns:
            Text with randomly selected sentences back-translated
        """
        # Split text into sentences while preserving separators
        parts = re.split(r"([.!?]+\s*)", text)
        new_parts = []

        # parts[::2] are sentences, parts[1::2] are separators
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            separator = parts[i + 1] if i + 1 < len(parts) else ""

            # Skip empty sentences
            if not sentence.strip():
                new_parts.append(sentence)
                new_parts.append(separator)
                continue

            if random.random() < tampering:
                translated = self._translate(sentence, direction=True)
                back_translated = self._translate(translated, direction=False)
                new_parts.append(back_translated)
            else:
                new_parts.append(sentence)

            new_parts.append(separator)

        return "".join(new_parts)
