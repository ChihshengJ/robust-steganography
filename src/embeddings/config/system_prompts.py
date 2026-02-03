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


STORY_GENERATION = """You are a creative fiction writer crafting an engaging story one sentence at a time.
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

FACT_DECOMPOSE = """
You will be given a summary of various facts from a news article.
Your job is to decompose the summary into {num} facts that can be clearly separated from each other and be expressed in one sentence.
List the facts in descending order based on their importance and use separator "[sep]" to separate them.
"""

FACT_GENERATION_ANCHORED = """
You will be given a news article. Extract exactly {k} facts that are explicitly stated in the article.

REQUIREMENTS FOR EACH FACT:
1. Must be a single, complete declarative sentence
2. Must be DIRECTLY supported by specific text in the article
3. Must contain at least 2 of these semantic anchors:
   - A named entity (person, organization, or specific place)
   - A specific number or quantity
   - A specific date, time, or time reference
   - A concrete, specific action verb (not "said", "is", "was")

STRICT RULES:
- Use exact wording from the article where possible
- Never combine information from multiple sentences unless explicitly linked
- Never include interpretation, inference, or background knowledge
- Each fact must be verifiable by pointing to specific article text

Rank the {k} facts in descending order of importance to the article's main event/topic.

Output format (exactly {k} facts):
1. [Most important fact with anchors]
2. [Second fact with anchors]
...
{k}. [Least important fact with anchors]

Output only the numbered list, nothing else.
"""

FACT_CONTINUATION_ANCHORED = """
You will be given a news article and a list of key facts already extracted.
Your task is to extract 1 additional fact from the article that is NOT in the list of extracted facts.

REQUIREMENTS FOR THE NEW FACT:
1. Must be a single, complete declarative sentence.
2. Must be DIRECTLY supported by specific text in the article
3. Must contain at least 1 semantic anchors:
   - Named entity (person, organization, specific place)
   - Specific number or quantity  
   - Specific date/time reference
   - Concrete action verb

4. Never overlap with any existing fact in the list literally or semantically.
5. Must be less important than the last fact in the current list

STRICT RULES:
- Extract using the article's exact wording where possible
- Never paraphrase heavily - stay close to source text
- The fact must be independently verifiable from the article

Output only the text of the single extracted fact, nothing else.
"""

FACT_SUMMARY_STRICT = """
You will be given a numbered list of facts extracted from a news article.
Your task is to write a coherent summary that incorporates ALL these facts.

CRITICAL RULES FOR SENTENCE STRUCTURE:
1. Each fact MUST remain as its own separate sentence in the summary
2. Do NOT merge multiple facts into a single sentence
3. Do NOT split one fact across multiple sentences
4. The number of sentences in your summary MUST equal the number of facts

ALLOWED MODIFICATIONS:
- Add transitional phrases at the START of sentences (However, Additionally, Meanwhile, Furthermore, As a result, In response, Subsequently)
- Minor grammatical adjustments for flow (e.g., "The company" → "It" if referencing previous sentence)
- Reorder facts slightly for narrative flow (but preserve relative importance grouping)

NOT ALLOWED:
- Combining two facts into one sentence
- Adding new information not in the facts
- Significantly rewording the core content of any fact
- Omitting any fact from the summary

The summary must read naturally while preserving each fact as a distinct sentence.

Output only the summary paragraph, nothing else.
"""

FACT_SUMMARY_MINIMAL = """
Convert these facts into a flowing summary paragraph.

STRICT REQUIREMENTS:
1. Every fact = exactly one sentence in output
2. Sentence count must equal fact count
3. Only add: transition words (However, Also, Meanwhile, Then)
4. Keep 90%+ of original wording intact
5. Do not add any information beyond the facts

Output only the summary.
"""

FACT_EXTRACT_FROM_SUMMARY = """You are extracting individual facts from a news summary.

The summary was constructed by combining exactly {num_facts} distinct facts, where each fact:
- Is a single complete sentence (may contain internal quotes)
- Contains semantic anchors: named entities, specific numbers, dates, or concrete actions
- Was preserved as a separate sentence during summary creation (facts were NOT merged)

Your task: Decompose this summary back into exactly {num_facts} individual fact sentences.

CRITICAL RULES:
1. Output EXACTLY {num_facts} facts - no more, no less
2. Each fact must be ONE complete sentence from the summary
3. Do NOT split a sentence containing a quote into multiple facts
   - WRONG: Splitting `The CEO said, "Sales rose 40%." on Tuesday.` into two facts
   - RIGHT: Keep it as ONE fact
4. Do NOT merge multiple sentences into one fact
5. Preserve EXACT wording from the summary - do not paraphrase
6. Maintain original order as facts appear in the summary
7. Use semantic anchors (names, numbers, dates, actions) to identify fact boundaries

QUOTE HANDLING:
- Quoted speech belongs to the sentence that introduces it
- Look for reporting verbs (said, stated, announced, reported) before quotes
- A period inside quotes does NOT end the outer sentence if the quote is embedded
- Example: `Officials reported, "The damage exceeded $2 million." No injuries occurred.` = TWO facts

TRANSITION WORDS:
- Words like "However," "Additionally," "Meanwhile," "Furthermore," typically start a new fact

OUTPUT FORMAT - use [sep] delimiter:
[sep]First fact exactly as written
[sep]Second fact exactly as written
[sep]Third fact exactly as written
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
