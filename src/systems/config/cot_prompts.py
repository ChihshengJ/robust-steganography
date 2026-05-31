BASE_GENERATION = """You are writing the opening of a step-by-step causal reasoning analysis. Generate exactly {k} reasoning steps that begin to answer the given question.

Requirements:
- Each step identifies ONE specific causal factor, mechanism, or process
- Each step MUST causally depend on the previous step — no independent observations
- Each step must be 20-40 words, written as a declarative causal claim
- Each step must name a concrete mechanism, substance, process, or entity
- Do NOT start multiple steps with the same grammatical structure
- Number each step

Output exactly {k} numbered steps, one per line."""

STEP_CONTINUATION = """You are continuing a step-by-step causal reasoning chain. Write ONE next step.

Requirements:
- Identify the next causal factor or mechanism that follows from the previous step's conclusion
- The step MUST causally depend on the most recent step — it should be a consequence, elaboration, or downstream effect
- Name a specific mechanism, process, substance, or entity (be concrete)
- The step must be clearly DISTINCT from every previous step:
  * Different causal factor — do not restate or rephrase an earlier step
  * Different sentence structure and opening words
  * No echoed phrasing or vocabulary from previous steps
- 20-40 words, self-contained declarative claim, natural analytical tone

Output only the step, nothing else."""

STEP_CONTINUATION_TEMPLATE = """Question: {question}

Previous reasoning steps ({n_previous}):
{previous_steps}

Write the next causal step:"""

SYNTHESIZE = """You are given a sequence of causal reasoning steps that form a chain answering a question.
Your goal is to synthesize them into a natural, flowing analytical paragraph.

Question: {question}

Requirements:
- Preserve every causal claim exactly as stated — do NOT alter the specific mechanisms, entities, or causal relationships described in any step
- Do NOT merge two steps into one claim or split one step into multiple claims
- Do NOT add new causal information or remove any existing causal claims
- You may add connective phrases (therefore, as a result, consequently, this leads to, building on this) to improve flow
- You may lightly adjust phrasing for grammatical flow between steps, but each step's core causal claim must remain individually identifiable
- The output should read as a single coherent analytical paragraph"""

DECOMPOSE = """Split this passage into exactly {n} segments, where each segment contains one distinct causal claim (one mechanism or factor causing one effect).

Rules:
1. Cut at the boundary between causal claims — do NOT reword, simplify, or paraphrase
2. Output the EXACT text from the passage for each segment, preserving original wording
3. If a sentence contains two chained causal claims, split at the junction between them
4. If consecutive sentences elaborate on the same causal mechanism, keep them as one segment
5. Maintain the original ordering
6. You MUST produce exactly {n} segments

Output format: segments separated by [sep]"""
