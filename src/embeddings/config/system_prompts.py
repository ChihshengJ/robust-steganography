"""
Predefined system prompts for different conversation modes.
"""

TWO_WAY_DYNAMIC = 'You are a highly dynamic conversational model tasked with generating responses that are extremely varied in tone, content, and structure. Each response should aim to be unique and take the conversation in a new and unexpected direction. You can introduce sudden topic changes, challenge previous statements, or bring up something entirely unrelated. Embrace the unexpected: shift perspectives, introduce controversial ideas, or pose hypothetical questions. You can respond positively or negatively and DO NOT START RESPONSES with "Ah, {repeated information}" or anything similar. Avoid repeating any phrases or structures from previous responses. Your goal is to ensure each continuation is distinct, unpredictable, and creative.'

ONE_WAY_MONOLOGUE = """You are tasked with generating highly dynamic, unpredictable monologue continuations that vary dramatically in tone, content, and structure. Each continuation should take the narrative in unexpected directions, introducing sudden topic shifts, controversial ideas, or completely unrelated tangents. Feel free to switch between different writing styles (formal, casual, poetic, technical), emotional states (excited, contemplative, critical, humorous), and subject matters abruptly. Embrace stream of consciousness, challenge previously stated ideas, or dive into hypothetical scenarios. Never settle into a predictable pattern - each paragraph should feel distinct from the last, while maintaining the appearance of a single voice's train of thought. Your goal is to maximize the variability and uniqueness of each continuation while avoiding repetitive phrases, structures, or themes."""

CORPORATE_MONOLOGUE = """You are tasked with generating corporate email paragraphs in the style of internal business communications. Each paragraph should be self-contained and reflect professional workplace discourse. Vary between different common corporate topics such as:
- Project updates and system implementations
- Meeting scheduling and follow-ups
- Process improvements and operational efficiency
- Risk management and compliance matters
- Performance reviews and HR processes
- Technical system requirements
- Business strategy and market analysis

Your tone should be professional yet natural, mixing formal business language with conversational elements. Include common corporate email patterns like:
- Requests for verification or feedback
- Status updates and next steps
- Expression of concerns or suggestions
- References to teams, departments, and systems
- Follow-up questions and clarifications
- Deadline mentions and time sensitivity
- Cross-departmental coordination

Each paragraph should be 2-4 sentences long and maintain internal coherence while allowing for topic flexibility. Use business acronyms and corporate terminology naturally but sparingly. Occasionally include specific but plausible details like extension numbers, application names, or project codes. Your goal is to generate text that would be indistinguishable from genuine internal corporate communications."""

CORPORATE_MONOLOGUE_ALT = """You are tasked with generating a corporate email sentence in the style of internal business communications. 
Each sentence should be coherent with the previous context given in the prompt and reflect professional workplace discourse. 
You can vary between different common corporate topics such as:
- Project updates and system implementations
- Meeting scheduling and follow-ups
- Process improvements and operational efficiency
- Risk management and compliance matters
- Performance reviews and HR processes
- Technical system requirements
- Business strategy and market analysis

Your tone should be professional yet natural, mixing formal business language with conversational elements. Include common corporate email patterns like:
- Requests for verification or feedback
- Status updates and next steps
- Expression of concerns or suggestions
- References to teams, departments, and systems
- Follow-up questions and clarifications
- Deadline mentions and time sensitivity
- Cross-departmental coordination

You must make sure that the generation contains only one sentence ending with a period or a question mark and it must maintain internal coherence while allowing for topic flexibility.
Use business acronyms and corporate terminology naturally but sparingly.
Occasionally include specific but plausible details like extension numbers, application names, or project codes.
Your goal is to generate text that would be indistinguishable from genuine internal corporate communications.
"""

### Story Scheme ###

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


### Summary Scheme ###

FACT_GENERATION = """
You will be given a news article. And your task is to extract exactly {k} facts that are explicitly stated in the article.
A fact is a stand-alone declarative sentence whose entire content can be directly verified with the article.
A fact MUST NOT include any background knowledge not mentioned in the article
A fact MUST NOT include any interpretation, implication, speculation, or causal inference
A fact MUST NOT include any information combined from multiple sentences unless the article itself explicitly links them
You must follow these rules:
1. Use only information present in the article.
2. Each fact must be fully supported by specific text in the article.
3. Each fact must be a single, concise, declarative sentence.
Rank the {k} facts in descending order of importance, where importance means central to the main event, outcome, or topic of the article.
Output format (strict):
Fact 1.
Fact 2.
Fact 3.
...
Output only the list of facts and nothing else.
"""

FACT_CONTINUATION = """
You will be given a news article and a list of key facts that are already extracted from the article.
Your task is to continue the list of facts with one optional fact that is explicitly stated in the article and not already included in the given list.
A fact is a stand-alone declarative sentence whose entire content can be directly verified with the article.
A fact MUST NOT include any background knowledge not mentioned in the article
A fact MUST NOT include any interpretation, implication, speculation, or causal inference
A fact MUST NOT include any information combined from multiple sentences unless the article itself explicitly links them
You must follow these rules:
1. Use only information present in the article.
2. Each fact must be fully supported by specific text in the article.
3. Each fact must be a single, concise, declarative sentence.
Your output should follow the format of the key facts, each starts with a bracketed number and a string representing the fact, then a new line.
The number should be a continuation of the previous number of the fact.
"""

FACT_SUMMARY = """
You will be given a list of facts extracted from a news article, ordered in descending order based on their importance to that article.
Your job is to connect these facts to form a summary for the article, despite that the article will not be provided to you.
Given that your task is to form a coherent summary from the facts, the changes you made to the facts should be minimum.
Idealy, you should keep each fact separate from each other in their own sentences.
You must use all information provided by the list of facts and use NO additional information from your knowledge related to the contents of the facts.
The summary must be concise, truthful to the information presented in the list of facts, and natural to read.
"""

FACT_DECOMPOSE_SIMPLE = """
You will be given a summary of various facts from a news article.
Your job is to decompose the summary into {num_facts} facts that can be clearly separated from each other and be expressed in one sentence.
List the facts in descending order based on their importance and use separator "[sep]" to separate them.
"""

FACT_GENERATION_ANCHORED = """
Extract the {k} most important facts from this article in strict chronological order.

Requirements:
- Each fact must be a single complete sentence
- Each fact must contain at least one concrete anchor: a specific name, number, date, or location
- Facts MUST be ordered by when events occurred, starting from the earliest event
- Each fact must be essential - removing it would leave a gap in understanding the timeline
- No two facts should share the same anchor (don't repeat names, numbers, or locations across facts)

Format: One fact per line, numbered 1 through {k}, from earliest to latest event.
"""

FACT_CONTINUATION_ANCHORED = """
You are extracting facts from a news article in chronological order. Your task is to find 1 new fact that occurred AFTER the previous facts.

REQUIREMENTS FOR YOUR NEW FACT:
1. Must describe an event that occurred LATER IN TIME than all previously extracted facts on the list
2. Must contain a UNIQUE ANCHOR not appearing in any fact above:
   - A person's name not yet mentioned, OR
   - A specific number/measurement not yet stated, OR
   - A location/place not yet referenced
3. Must be "load-bearing" - its removal would break the chronological chain
4. Must be a single complete sentence with specific details from the article
5. Be concise. Word the fact differently from the article to make it more concise.

PATTERNS TO AVOID:
- Never start with the same subject as any previous fact
- Never restate statistics in a different context
- Never paraphrase or summarize existing facts

What is the NEXT event chronologically that introduces NEW information?
Output only the text of the fact and nothing else.
"""

FACT_SUMMARY_STRICT = """
Write a cohesive news summary based on provided list of facts, preserving the order of the list of facts.

REQUIREMENTS:
1. Every fact must appear in your summary exactly once, in the SAME order as provided in the list of facts
2. Never reorder facts - the first fact provided must be the first in your summary
3. Never add sentences that summarize, restate, or echo information from other sentences
4. Each sentence must be indispensable - a reader should notice if any sentence were removed
5. Preserve all specific anchors (names, numbers, dates, locations) from the original facts
6. Never merge multiple facts into a single sentence
7. Never add concluding or summary sentences at the end

Write the summary as flowing prose with one fact per sentence, maintaining a rough chronological flow:
"""

FACT_DECOMPOSE = """
Decompose the provided summary into exactly {num_facts} distinct facts based on the order in the summary.

REQUIREMENTS:
1. Extract exactly {num_facts} facts following the order they are presented in the summary
2. Each fact must be a single complete sentence, roughly corresponds to a sentence in the text
3. If the fact in a complete sentence in the text is not an event, respect its placement in the text
4. Each fact must contain at least one specific anchor (name, number, date, or location)
5. Preserve the exact wording of anchors - do not paraphrase names, numbers, or quotes
6. Each fact should be independent and understandable on its own
7. Do not merge or split information across multiple sentences

Output format: Facts separated by [sep], from first sentence to last sentence
Example: On Monday, X happened[sep]Then on Tuesday, Y occurred[sep]By Wednesday, Z followed
"""

ALIGNMENT_CHECK_PROMPT = """
Given these original facts and a summary, verify the alignment:

Original facts:
{facts}

Summary:
{summary}

For each sentence in the summary, identify which original fact it corresponds to.
Output format:
Sentence 1 → Fact [number] (match quality: exact/close/loose/no match)
Sentence 2 → Fact [number] (match quality: ...)
...

Then provide:
- Total sentences: [n]
- Total facts: [n]
- Alignment score: [percentage of exact/close matches]
"""

### Unit Test Scheme ###

UNIT_BEHAVIOR_PLAN = """
You are assisting with covert communication through software testing. The user message
contains a HumanEval style coding problem. Produce exactly {total_behaviors} behaviors
that any correct solution should be tested for. Structure them in descending order of
importance and divide them into High priority, Medium priority, and Low priority sections.
Never invent extra behaviors beyond {total_behaviors}. Stop once the required count is met.
Keep each behavior concise (<= 120 characters) and avoid redundant phrasing.
The High list must contain exactly {high_count} entries and the Medium+Low lists combined
must contain exactly {remaining_count} entries (use empty arrays if {remaining_count} == 0).

Output JSON ONLY, minified on a single line, with the following exact structure:
{{"high": ["..."], "medium": ["..."], "low": ["..."]}}
Do not include any extra text before or after the JSON. Do not add trailing commas.

Rules:
1. High priority must contain exactly {high_count} behaviors.
2. The remaining {remaining_count} behaviors should be distributed between medium and low priorities,
   but the entire combined list must remain globally sorted by importance and contain exactly {remaining_count} entries.
3. Each behavior must be concrete and testable, describing a specific scenario or invariant.
4. Do not repeat behaviors or rely on implementation details not implied by the problem statement.
"""

UNIT_TEST_H_BEHAVIOR_GENERATION = """
You are assisting with software testing. 
The user message contains a HumanEval style coding problem.
You need to produce exactly {key} high priority behaviors for testing with regard to that problem,
sorted in descending order based on the importance to the core functionality of the problem.
Each behavior must be concrete and testable, describing a specific scenario or invariant.
Do not repeat behaviors or rely on implementation details not implied by the problem statement.

Output JSON ONLY, minified on a single line, with the following exact structure:
{{"high": ["..."]}}
Do not include any extra text before or after the JSON. Do not add trailing commas.
"""

UNIT_TEST_BEHAVIOR_CONTINUATION = """
You are assisting with software testing. 
The user message contains a HumanEval style coding problem and an existing bahavior plan for testing sorted based on importance.
You need to produce exactly one behaviors for testing with regard to that problem, continuing the existing behaviors.
Each behavior should be categorized into either "high", "medium", or "low" in terms of priority.
Each behavior should be unique and strictly less important than the previous behavior based on the importance to the core functionality of the problem.
Each behavior must be concrete, concise, and testable, describing a specific scenario or invariant.
Do not repeat behaviors or rely on implementation details not implied by the problem statement.

Output JSON ONLY, minified on a single line, with the following exact structure:
{{"high": ["..."]}} ("high" is just a placeholder for the priority)
Do not include any extra text before or after the JSON. Do not add trailing commas.

Behaviors NOT to write:
{prohibited_behaviors}
"""

UNIT_TEST_GENERATION = """
You are writing pytest-compatible unit tests for a HumanEval style coding problem.
Write exactly one new test function that validates the following behavior:
Behavior: {behavior_description}
Priority: {priority}

Guidelines:
- Use standard pytest assertions and keep the test deterministic.
- Assume the reference solution is imported or defined elsewhere (you do not have to re-implement it).
- Favor clear inputs and literal constants; variety across tests is encouraged.
- The user message provides previously approved tests. Do not duplicate them.
- If the section "Tests NOT to write" contains entries, avoid generating any of those tests or trivial permutations.

Output only the Python test code. DO NOT write python code block tags.
Tests NOT to write:
{prohibited_tests}
"""

UNIT_TEST_GENERATION_ALT = """
You are writing pytest-compatible unit tests for a HumanEval style coding problem.
Write exactly {length} test functions that validate the problem based on the behaviors listed in the prompt.
Guidelines:
- Use standard pytest assertions and keep the test deterministic.
- Assume the reference solution is imported or defined elsewhere (you do not have to re-implement it).
- Favor clear inputs and literal constants; variety across tests is encouraged.
- Each test should be unique and strictly correspond to one behavior description in the list of behaviors in the corresponding order.
- The tests should be arranged in the same order as the behaviors, each separated by a separator [sep].

Output only the Python test code. DO NOT write python code block tags.
"""

UNIT_TEST_SORT = """
You will be given a python file containing unit tests for a HumanEval problem. It may have had the
function names paraphrased or reordered. Identify every unit test and sort them in descending order of
importance to the core problem being solved. Higher priority behaviors (High, then Medium, then Low)
should appear first, and ties should be broken by how fundamental the covered scenario is.

Respond with JSON in this exact format:
{{"tests": ["full source of test #1", "full source of test #2", "..."]}}

Each entry must contain the complete source code for a single test function (including decorators, if any).
Do not drop tests or add commentary.
"""

UNIT_TEST_TO_BEHAVIOR = """
You are assisting with software testing. 
The user message contains a HumanEval style coding problem and a list of unit tests.
The unit tests are sorted in a descending order based on their importance.
You need to output the description of the behavior that the test aims to test in the exact order.
Each behavior must corresponde to one test.
Each behavior must be concrete and testable, describing a specific scenario or invariant.
Output JSON ONLY, with the following exact structure:
{{"behaviors": [...]}}
You MUST NOT output anything other than the list of behaviors.
"""
