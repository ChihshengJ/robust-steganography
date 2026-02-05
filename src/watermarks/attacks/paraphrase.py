import random
import re

from openai import OpenAI

from .attack import Attack

SYSTEM_PROMPT_LOCAL = """You are a paraphrasing assistant.
Rewrite the given sentence using completely different words and phrasing while preserving the exact meaning.

Rules:
1. Use synonyms and alternative expressions wherever possible
2. Change the sentence structure (e.g., active to passive, reorder clauses)
3. Preserve all factual information exactly - do not add, remove, or alter any details
4. Maintain the same tone and register
5. Output ONLY the paraphrased sentence with no explanation or commentary

Example:
Input: "The scientist discovered a new species in the rainforest."
Output: "A previously unknown species was found by the researcher in the tropical jungle."
"""

SYSTEM_PROMPT_GLOBAL = """You are an expert paraphrasing assistant tasked with completely rewriting text while preserving its meaning.
Your objective is to transform the text as dramatically as possible while keeping all information intact.

Techniques to apply:
1. **Vocabulary**: Replace words with synonyms throughout. Avoid reusing any distinctive phrases from the original.
2. **Sentence structure**: Convert between active/passive voice, change clause order, split long sentences or combine short ones.
3. **Paragraph organization**: Reorder information where logically possible. Lead with different points than the original.
4. **Expression style**: If the original is formal, stay formal but use different formal expressions. Same for informal text.

Critical constraints:
- Every fact, name, number, and detail from the original MUST appear in your output
- Never add any new information or interpretations
- Never omit any information in the text, no matter how minor it seems

Output format:
First, silently identify the key information that must be preserved.
Then output your paraphrase after the marker "[paraphrased message]".
Output only the paraphrased text after the marker - no bullet points, no explanations.
"""

SYSTEM_PROMPT_GLOBAL_SIMPLE = """Completely rewrite the following text using different words, sentence structures, and organization. 

Requirements:
- Change as much of the wording as possible
- Restructure sentences and reorder information where logical
- Preserve ALL facts, names, numbers, and details exactly
- Do not add or remove any information
- Maintain the same overall meaning and tone

Output only your rewritten version, nothing else.
"""


class ParaphraseAttack(Attack):
    """Attack that uses GPT to paraphrase text while preserving meaning."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        local_mode: bool | None = None,
        use_simple_prompt: bool = False,
    ):
        """
        Initialize the paraphrase attack.

        Arguments:
            client: OpenAI client instance
            model: GPT model to use (default: "gpt-4o-mini")
            temperature: Sampling temperature (default: 0.7 for variety in paraphrasing)
            local_mode: Controls local vs global attack behavior.
                - None (default): Use legacy behavior where tampering >= 0.99 forces global mode.
                - True: Force local mode (sentence-level) even at 100% tampering.
                - False: Force global mode regardless of tampering level.
            use_simple_prompt: If True, use the simpler global prompt without the
                marker extraction step. May work better with some models.
        """
        super().__init__(local_mode=local_mode)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.use_simple_prompt = use_simple_prompt

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the paraphrase attack."""
        use_local = self._resolve_local_mode(local, tampering)

        if use_local:
            return self._local_paraphrase(text, tampering)
        else:
            return self._global_paraphrase(text)

    def _global_paraphrase(self, text: str) -> str:
        """Paraphrase entire text at once."""
        if self.use_simple_prompt:
            return self._global_paraphrase_simple(text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_GLOBAL,
                    },
                    {"role": "user", "content": f"Paraphrase this text:\n\n{text}"},
                ],
                temperature=self.temperature,
                top_p=0.95,
            )
            result = response.choices[0].message.content.strip()

            # Extract content after the marker
            match = re.search(
                r"\[paraphrased message\]\s*(.*)", result, re.DOTALL | re.IGNORECASE
            )
            if match:
                result = match.group(1).strip()
                # Clean up any remaining newlines for consistency
                result = " ".join(result.split())
            else:
                # Fallback: if marker not found, use the whole response
                print(
                    "Warning: Marker not found in paraphrase response, using full output"
                )
                result = " ".join(result.split())

            return result
        except Exception as e:
            print(f"Global paraphrase attack failed: {e}")
            return text

    def _global_paraphrase_simple(self, text: str) -> str:
        """Paraphrase entire text using the simpler prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_GLOBAL_SIMPLE,
                    },
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                top_p=0.95,
            )
            result = response.choices[0].message.content.strip()
            result = " ".join(result.split())
            return result
        except Exception as e:
            print(f"Global paraphrase (simple) attack failed: {e}")
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
                if i + 1 < len(parts):
                    new_parts.append(parts[i + 1])
                continue

            if random.random() < tampering:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_LOCAL},
                            {"role": "user", "content": sentence.strip()},
                        ],
                        temperature=self.temperature,
                        top_p=0.95,
                    )
                    paraphrased = response.choices[0].message.content.strip()
                    # Remove any accidentally added punctuation at the end
                    # since we'll add the separator back
                    paraphrased = paraphrased.rstrip(".!?")
                    new_parts.append(paraphrased)
                except Exception as e:
                    print(f"Local paraphrase attack failed for sentence: {e}")
                    new_parts.append(sentence)
            else:
                new_parts.append(sentence)

            # Add the separator if it exists
            if i + 1 < len(parts):
                new_parts.append(parts[i + 1])

        result = "".join(new_parts)
        # Clean up any double punctuation
        result = re.sub(r"([.!?])\1+", r"\1", result)
        return result
