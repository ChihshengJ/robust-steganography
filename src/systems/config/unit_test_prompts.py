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
