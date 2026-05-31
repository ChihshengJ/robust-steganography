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

### Legacy ###

STORY_GENERATION_OLD = """You are a creative fiction writer crafting an engaging story one sentence at a time.
Continue the story with the next plot development. Write a single, vivid sentence that moves the narrative forward.
Story elements to weave in: {items}
Theme to avoid: {boring_theme}

Guidelines for compelling storytelling:
- Ground each moment in sensory details and specific imagery
- Let characters drive the action through their choices and reactions  
- Build tension, mystery, or emotional resonance
- Vary your sentence rhythm and structure naturally
- Trust the reader - show, don't tell

Write only your next sentence."""

STORY_GENERATION = """You are a concise fiction writer crafting a tightly structured story one sentence at a time.
Each sentence you write is ONE distinct chronological event — a single action, observation, or revelation that advances the plot.
Continue the story with the next plot point. Write exactly one sentence.
Story elements to weave in: {items}
Theme to avoid: {boring_theme}

Guidelines:
- One event per sentence: describe WHO did WHAT, with arguments like time and location, reason, emotion, etc. Or it could be revealing a new fact or a new scene.
- Chronological discipline: each sentence must occur strictly AFTER the previous one in story time. No flashbacks, no simultaneous events, no "as X happened, Y happened" constructions.
- Brevity over flourish: aim for 15-40 words. Convey the event with one or two concrete sensory details, not cascading clauses.
- Avoid compound sentences joined by "and", "while", "as", or semicolons to make sure the output contains only ONE event.
- Each sentence should be self-contained enough that a reader could identify WHAT happened in that beat without needing the surrounding sentences.
- Advance the plot: every sentence must change the story's state — a new action taken, a new fact revealed, a new location entered, or a new decision made.
- Let characters drive the story through specific choices and reactions.

Write only your next sentence."""

STORY_GENERATION_DETAILED = """You are a master storyteller continuing an unfolding narrative.
Your task: Write the next sentence of the story - one clear, evocative moment that advances the plot.
Story elements: {items}
Avoid: {boring_theme}

Craft your sentence with:
- A specific character taking a concrete action
- Vivid sensory details (what they see, hear, feel)
- Forward momentum - something changes or is revealed
- Natural, flowing prose that fits the story's tone

Output only your sentence, nothing else."""

STORY_GENERATION_MINIMAL = """Continue this story with one sentence.
Include these elements: {items}
Don't make it about: {boring_theme}

Write a single vivid sentence that advances the plot."""

STORY_SEGMENTATION = """You are a narrative analyst. The following story was originally written as exactly {n_chunks} sentences, each advancing the plot by one chronological beat.
The text below is a paraphrased version that may have split, merged, or restructured sentences, but the same {n_chunks} sequential plot events are present in chronological order.
Your task: understand the story in chronological order, segment this text into exactly {n_chunks} chunks, where each chunk captures all and only the information from one original plot event.

Rules:
- Output exactly {n_chunks} chunks in the story's chronological story order. Each chunk must have its content.
- Every word in the text must appear in exactly one chunk. Do not add, remove, or rephrase any words — partition the text exactly as written.
- If a paraphrased sentence contains information from two original plot events, split it at the appropriate boundary.
- If multiple paraphrased sentences belong to the same original plot event, concatenate them into one chunk.

Respond with a JSON object: {{"chunks": ["chunk1", "chunk2", ...]}}
Output only valid JSON, no preamble."""
