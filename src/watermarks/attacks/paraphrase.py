import random
import re
from datetime import datetime

from openai import OpenAI

from .attack import Attack

SYSTEM_PROMPT_LOCAL = """
You are an assistant tasked with paraphrasing text. For each input, your goal 
is to rephrase it using different wording and sentence structures while ensuring 
that the original meaning, intent, and nuances are completely preserved. Do not 
omit or add new information. The paraphrased output should be clear, concise, 
and faithful to the original message.

Important: Preserve all formatting including newlines, spaces, and punctuation 
placement. Return only the paraphrased text with no additional commentary.
"""

SYSTEM_PROMPT = """
You are an assistant tasked with paraphrasing text.
You are encouraged to reword and reorganize the input text entirely to increase the difference between the input text and the paraphrased text.
Apart from word choice, you should try to rearrange the sentence order and the structure of the text.
However, do not omit any existing information or add new information.
The paraphrased output should be clear and faithful to the original message, but significantly different from the original text.
"""

SYSTEM_PROMPT_STRONG = """
You are an assistant tasked with paraphrasing text.
Your objective is to alter the original message as much as possible, while perserving all the crucial information in the original text.
In order to do so, you can first list all important information in the original text that you think should not be omitted, and then write a completely new message based on those key points.
Use a marker "[paraphrased message]" to mark your output.
It is encouraged to structurally change the origianl text entirely.
Overall, the paraphrased output should be clear and faithful to the original message, but significantly different from the original text.
"""


class ParaphraseAttack(Attack):
    """Attack that uses GPT to paraphrase text while preserving meaning."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the paraphrase attack.

        Args:
            client: OpenAI client instance
            model: GPT model to use (default: "gpt-4o-mini")
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            local: If True, paraphrases each sentence independently.
                  If False, paraphrases entire text at once (default: True)
        """
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the paraphrase attack."""
        if local and tampering < 0.99:
            return self._local_paraphrase(text, tampering)
        else:
            return self._global_paraphrase(text)

    def _global_paraphrase(self, text: str) -> str:
        """Paraphrase entire text at once."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        + SYSTEM_PROMPT_STRONG,
                    },
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            result = response.choices[0].message.content.strip()
            match = re.search(r"\[paraphrased message\](.*)", result, re.DOTALL)
            if match:
                # print("\nattack successed\n")
                result = match.group(1).strip().replace("\n", "")
            else:
                print("\n attack failed\n")
                pass
            # print("Debug glbal paraphrase:")
            # print(f"in:\n{text}\nout:\n{result}")

            return result
        except Exception as e:
            print(f"Global paraphrase attack failed: {e}")
            return text

    def _local_paraphrase(self, text: str, tampering: float) -> str:
        """Paraphrase each sentence independently while preserving structure."""
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
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_LOCAL},
                            {"role": "user", "content": sentence},
                        ],
                        temperature=self.temperature,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                    new_parts.append(response.choices[0].message.content.strip())
                except Exception as e:
                    print(f"Local paraphrase attack failed for sentence: {e}")
                    new_parts.append(sentence)
            else:
                new_parts.append(sentence)

            # Add the separator if it exists
            if i + 1 < len(parts):
                new_parts.append(parts[i + 1])

        result = "".join(new_parts).replace("..", ".")
        # print("Debug local paraphrase:")
        # print(f"parts:\n{parts}\nnew_parts:\n{result}")
        return result
