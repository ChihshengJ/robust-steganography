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

STORY_GENERATION = """
You will be given three elements (e.g., car, wheel, drive) and then asked to write the plot of a short story that contains these
three elements. Instead of writing a standard story such as "I went for a drive in my car with my hands on
the steering wheel.", you come up with a novel and unique story that uses the required elements in unconventional ways.

Write or continue the plot of a short story. The story must include the following three elements: {items}.
However, the story should not be about {boring_theme}.
You should output one and only one clearly stated plot event based on the given context in the user prompt, if any, that pushes the narrative forward.
It should not be excessively long. One sentence is ideal. And it should be coherent with the story so far.
Your output should only contain the text for the plot. The use of fancy words or overly detailed descriptions is not suggested.
"""

STORY_SEGMENTATION = """
You are tasked with segmenting a story into {chunk_length} parts based on the events in chronological order.
Each chunk should contain a clear event that keeps the story going, and should contain 2-3 sentences, but the length can vary.
So please prioritize making sure that the number of chunks is exactly {chunk_length}.
Your output should only consists of chunks from the original text.
Please output these chunks in order and segment them with a separator [sep], do not change a word of the original text in your output.
"""

STORY_SEGMENTATION_NOCUE = """
You are tasked with decomposing a story into various singular plot events in chronological order.
Your response should be in JSON format, the entire json object should be formated like:
{"events" : [event_1, event_2, event_3, ... , event_i]}
Each event should be stated with clarity and it should contain all information of the original plot.
"""

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
[1]Fact 1.
[2]Fact 2.
[3]Fact 3.
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
You should NOT output any facts described in the following list:
{prohibited_facts}
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

Output only the Python test code.
Tests NOT to write:
{prohibited_tests}
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
