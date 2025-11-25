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
