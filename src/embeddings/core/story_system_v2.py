import hashlib
import json
import random
import time
from typing import Any

from .encoder import Encoder
from .error_correction import ErrorCorrection
from .steg_system import StegSystem
from .topicQA_system import _BitsPerGroupStub


def _llm(
    client,
    model,
    prompt,
    system="You are a helpful assistant.",
    temperature=0,
    max_tokens=1000,
    top_p=0.7,
    extra_body: dict | None = None,
):
    for attempt in range(3):
        try:
            kwargs = dict(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2**attempt)


def _parse_slots(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        slots = json.loads(raw)
    except json.JSONDecodeError:
        return []
    valid = []
    for s in slots:
        if isinstance(s, dict) and "slot" in s and "A" in s and "B" in s:
            valid.append(s)
    return valid


SLOT_GENERATION_PROMPT = """
Given the following story premise, generate exactly {n} independent story beat slots.
Each slot is one narrative event or detail in the story. For each slot, provide TWO alternative concrete details (A and B) that could fill that slot.

Requirements:
- Alternatives must be clearly distinguishable concrete objects, locations, methods, or actions (not abstract qualities)
- Slot dimensions must focus on plot or setting choices (objects, locations, methods, events, physical descriptions). DO NOT make any slot dimension about a character's name, identity, role, or personal attributes. Character names should not appear in the A/B alternatives.
- No two slots may share the same alternative details or describe the same narrative moment.
- Alternatives must be interchangeable: picking A or B for one slot must not affect any other slot
- Slots should follow a natural narrative order (setup -> rising action -> climax -> resolution)
- Each alternative should be a short phrase (3-10 words)
- Alternatives within a slot must be mutually exclusive and equally plausible

Output ONLY a JSON array. No explanation. NO code block or markdown wrapping!!
Example: [{{"slot": "Intel delivery method", "A": "a sealed envelope under the door", "B": "an encrypted flash drive"}}, {{"slot": "Weapon type discovered", "A": "a compact nuclear device", "B": "a nerve agent canister"}}]

Story premise: {premise}"""

STORY_SYNTHESIS_PROMPT = """Write a short story based on the following premise.
Incorporate each of the specific details listed below naturally into the narrative.
Every detail MUST appear clearly and unambiguously in the story text.
Use flowing prose with natural pacing. Develop the story through concrete actions, physical detail, and brief dialogue only where it advances the plot. Vary paragraph length naturally.

Premise: {premise}

Details to include:
{events_str}"""

SLOT_DECODE_PROMPT = """Read this story carefully:

---
{story}
---

Story premise: "{premise}"

For the narrative detail "{slot_desc}", which concrete version appears in the story?

(A) {option_a}
(B) {option_b}

Reply with ONLY the letter (A or B)."""


class StorySystemV2(StegSystem):
    """Steganography via binary detail selection in story slots.

    Each slot has two alternative concrete details (A/B). The message bits
    determine which detail fills each slot. Decoding is forced-choice per slot.
    Key-permutation controls which message bit maps to which slot.

    1 bit per slot. N slots = N raw channel bits.
    """

    def __init__(
        self,
        client: Any,
        error_correction: ErrorCorrection,
        local_client: Any,
        local_model: str,
        n_slots: int = 12,
        response_model: str = "gpt-4.1",
        decoder_model: str = "gpt-4.1",
        key: str = "default",
        encoder: Encoder | None = None,
        response_temperature: float = 0.7,
    ) -> None:
        stub = _BitsPerGroupStub(1)  # 1 bit per slot
        super().__init__(client, stub, error_correction, encoder)

        self.local_client = local_client
        self.local_model = local_model
        self.n_slots = n_slots
        self.response_model = response_model
        self.decoder_model = decoder_model
        self.key = key
        self.response_temperature = response_temperature

        self._premise: str | None = None
        self._last_metadata: dict | None = None

    @property
    def raw_capacity_bits(self) -> int:
        return self.n_slots

    @property
    def premise(self) -> str | None:
        return self._premise

    @premise.setter
    def premise(self, value: str | None) -> None:
        self._premise = value

    def generate_slots(self, premise: str) -> list[dict]:
        raw = _llm(
            self.local_client,
            self.local_model,
            SLOT_GENERATION_PROMPT.format(n=self.n_slots, premise=premise),
            temperature=0,
            top_p=1.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        slots = _parse_slots(raw)
        if len(slots) > self.n_slots:
            slots = slots[: self.n_slots]
        if len(slots) < self.n_slots:
            print(
                f"  WARNING: local model returned {len(slots)} slots, "
                f"expected {self.n_slots}"
            )
        return slots

    def _key_permutation(self, n: int) -> list[int]:
        """Key-derived permutation mapping bit position -> slot index."""
        rng = random.Random(hashlib.sha256(self.key.encode()).hexdigest())
        perm = list(range(n))
        rng.shuffle(perm)
        return perm

    def _assign_bits_to_slots(
        self, slots: list[dict], bit_chunks: list[list[int]]
    ) -> list[dict]:
        """Map each message bit to a slot via key-permutation.

        Returns ALL slots in narrative order, each with chosen detail.
        Slots without a message bit default to A.
        """
        perm = self._key_permutation(len(slots))
        n_bits = len(bit_chunks)

        slot_bit: dict[int, int] = {}
        for bit_idx in range(n_bits):
            slot_idx = perm[bit_idx]
            slot_bit[slot_idx] = bit_chunks[bit_idx][0]

        assigned = []
        for si, slot in enumerate(slots):
            bit = slot_bit.get(si, 0)
            key = "A" if bit == 0 else "B"
            assigned.append(
                {
                    "slot": slot["slot"],
                    "A": slot["A"],
                    "B": slot["B"],
                    "chosen_key": key,
                    "chosen": slot[key],
                    "bit": bit,
                    "carries_message": si in slot_bit,
                }
            )
        return assigned

    def _decode_slot(self, story: str, premise: str, slot: dict) -> int:
        """Forced-choice: does the story contain detail A (0) or B (1)?"""
        prompt = SLOT_DECODE_PROMPT.format(
            story=story,
            premise=premise,
            slot_desc=slot["slot"],
            option_a=slot["A"],
            option_b=slot["B"],
        )
        answer = (
            _llm(
                self.client,
                self.decoder_model,
                prompt,
                temperature=0,
            )
            .strip()
            .upper()
        )
        if "B" in answer:
            return 1
        return 0

    def _recover_bits_from_slots(
        self, story: str, premise: str, slots: list[dict], n_bits: int
    ) -> list[list[int]]:
        """Decode active slots and un-permute to recover message bit order."""
        perm = self._key_permutation(len(slots))
        active_slot_indices = set(perm[:n_bits])

        slot_detections: dict[int, int] = {}
        for si, slot in enumerate(slots):
            if si not in active_slot_indices:
                continue
            detected = self._decode_slot(story, premise, slot)
            slot_detections[si] = detected
            print(
                f"  slot {si} ({slot['slot'][:35]}): "
                f"detected={'B' if detected else 'A'} "
                f"[A={slot['A'][:25]} | B={slot['B'][:25]}]"
            )

        recovered = []
        for bit_idx in range(n_bits):
            slot_idx = perm[bit_idx]
            bit = slot_detections.get(slot_idx, 0)
            recovered.append([bit])
        return recovered

    def _generate_story(self, premise: str, assigned: list[dict]) -> str:
        events_str = "\n".join(f"- {a['slot']}: {a['chosen']}" for a in assigned)
        prompt = STORY_SYNTHESIS_PROMPT.format(
            premise=premise,
            events_str=events_str,
        )
        return _llm(
            self.client,
            self.response_model,
            prompt,
            temperature=self.response_temperature,
            max_tokens=3000,
        )

    def paraphrase(self, text: str, model: str | None = None) -> str:
        return _llm(
            self.client,
            model or self.response_model,
            f"Rewrite this story completely in your own words, "
            f"preserving all plot events and concrete details:\n\n{text}",
            temperature=0.7,
            max_tokens=3000,
        )

    def encode(
        self,
        chunks: list[list[int]],
        history: Any,
        system_prompt: str = "",
        max_length: int = 200,
        **kwargs,
    ) -> tuple[str, dict]:
        premise = history if isinstance(history, str) else history[0]
        slots = self.generate_slots(premise)

        if len(chunks) > len(slots):
            raise ValueError(
                f"Payload requires {len(chunks)} bits but only "
                f"{len(slots)} slots available"
            )

        assigned = self._assign_bits_to_slots(slots, chunks)
        story = self._generate_story(premise, assigned)

        metadata = {
            "slots": slots,
            "assigned": assigned,
            "chunks": chunks,
        }
        return story, metadata

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        self._premise = seed
        chunks, self._error_encoded_length = self._encode_to_chunks(data)
        print(f"Payload: {len(chunks)} bits (capacity: {self.raw_capacity_bits})")

        story, metadata = self.encode(chunks, seed)
        self._last_metadata = metadata
        print("Slot assignments:")
        for a in metadata["assigned"]:
            tag = "*" if a["carries_message"] else " "
            print(
                f" {tag} {a['slot']}: {a['chosen']} ({a['chosen_key']}, bit={a['bit']})"
            )
        return story

    def recover_message(self, stego_text: str) -> Any:
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first "
                "or set error_encoded_length."
            )
        if self._premise is None:
            raise ValueError("No premise set. Run hide_message first or set premise.")

        expected_bits = self._error_encoded_length

        slots = self.generate_slots(self._premise)
        if expected_bits > len(slots):
            raise ValueError(
                f"Expected {expected_bits} bits but only {len(slots)} slots"
            )

        recovered_chunks = self._recover_bits_from_slots(
            stego_text,
            self._premise,
            slots,
            expected_bits,
        )

        m_bits = self.ecc.decode(recovered_chunks, self._error_encoded_length)
        return self.encoder.decode(m_bits)
