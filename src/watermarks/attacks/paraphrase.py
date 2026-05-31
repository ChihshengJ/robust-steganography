import random
import re

from openai import OpenAI

from .attack import Attack, iter_sentences_with_gaps

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

# SYSTEM_PROMPT_GLOBAL = """You are an expert paraphrasing assistant tasked with completely rewriting text while preserving its meaning.
# Your objective is to transform the text as dramatically as possible while keeping all information intact.
#
# Techniques to apply:
# 1. **Paragraph organization**: This is the priority.
#     Reorder information where logically possible.
#     Change the sentence order, swap close sentences, parallelled key points.
#     Nothing is off-limit as long as the entire text conveys the same information.
# 2. **Vocabulary**: Replace words with synonyms throughout. Avoid reusing any distinctive phrases from the original.
# 3. **Sentence structure**: Convert between active/passive voice, change clause order, split long sentences or combine short ones.
# 4. **Expression style**: If the original is formal, stay formal but use different formal expressions. Same for informal text.
#
# Critical constraints:
# - Every fact, name, number, and detail from the original MUST appear in your output
# - Never add any new information or interpretations
# - Never omit any information in the text
#
# Output format:
# First, identify the key information that must be preserved.
# Output them after the marker "[key points]".
# Then output your paraphrase based on the key points after the marker "[paraphrased message]".
# Output only the paraphrased text after the marker - no bullet points, no explanations.
# """

SYSTEM_PROMPT_GLOBAL = """You are an expert paraphrasing assistant. Your task is to completely rewrite text so that it is structurally unrecognizable from the original while preserving every piece of information.
## Priority 1: Structural Transformation
- Aggressively reorder paragraphs and sentences. Move conclusions to the beginning, supporting details to new positions, or interleave points that were originally separated.
- Split the text into a different number of paragraphs than the original.
- If the original presents A then B then C, consider presenting B then C then A, or weaving them together — any arrangement that conveys the same information in a different structure.
- Merge sentences that were separate; break apart sentences that were combined. No sentence in your output should map 1-to-1 to an original sentence.
## Priority 2: Lexical and Syntactic Transformation
- Replace distinctive words and phrases with synonyms or equivalent expressions.
- Alternate between active and passive voice differently from the original.
- Restructure clause ordering within sentences (e.g., move subordinate clauses, invert cause-effect presentation).
- Match the register (formal/informal) of the original but use entirely different expressions within that register.
## Hard Constraints
- Every fact, name, number, date, statistic, and detail from the original MUST appear in your output. Missing even one is a failure.
- Add nothing: no new information, no interpretations, no editorial commentary.
- Remove nothing: if the original mentions it, your output mentions it.

## Output Format
First, extract every discrete fact and detail that must be preserved. List them after the marker "[key points]" as a numbered list.
Then, using ONLY those key points, write a fully restructured paraphrase after the marker "[paraphrased message]". Output only flowing prose after this marker — no bullet points, no headers, no meta-commentary.

## Self-Check Before Outputting
Verify: (1) Does every key point appear in the paraphrase? (2) Is the paragraph/sentence order substantially different from the original? (3) Would a side-by-side comparison show no sentence-level correspondence? If any answer is no, revise before outputting."""


class ParaphraseAttack(Attack):
    """Attack that uses GPT to paraphrase text while preserving meaning."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
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
        """
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        """Apply the paraphrase attack."""
        if not 0 <= tampering <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if tampering == 0:
            return text

        if self._resolve_local_mode(local, tampering):
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

    def _local_paraphrase(self, text: str, tampering: float) -> str:
        """Paraphrase each sentence independently while preserving structure."""
        new_parts: list[str] = []
        for gap, sentence in iter_sentences_with_gaps(text):
            new_parts.append(gap)
            if not sentence:
                continue
            if not sentence.strip() or random.random() >= tampering:
                new_parts.append(sentence)
                continue
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
                # Preserve the original sentence's trailing punctuation
                m = re.search(r"[.!?]+$", sentence.rstrip())
                trailing = m.group(0) if m else ""
                paraphrased = paraphrased.rstrip(".!?").rstrip() + trailing
                new_parts.append(paraphrased)
            except Exception as e:
                print(f"Local paraphrase attack failed for sentence: {e}")
                new_parts.append(sentence)

        result = "".join(new_parts)
        result = re.sub(r"([.!?])\1+", r"\1", result)
        return result
