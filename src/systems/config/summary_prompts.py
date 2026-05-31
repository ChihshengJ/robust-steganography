# =============================================================================
# EXTRACTION PROMPTS
# =============================================================================

FACT_GENERATION_ANCHORED = """Extract the {k} most important facts from this article, following the order they appear in the article from top to bottom.

Requirements:
- Each fact must be a single, concise sentence
- Each fact must contain at least one concrete anchor: a specific name, number, date, or location
- Facts must follow ARTICLE ORDER — extract from early paragraphs first, then later paragraphs
- Each fact must cover a DIFFERENT ASPECT of the story. Aspects include:
  * Core event (what happened)
  * Key actors (who was involved, their roles)
  * Quantitative details (statistics, amounts, measurements)
  * Direct quotes or attributed statements
  * Location or setting specifics
  * Causes or background context
  * Consequences or outcomes
  * Reactions from other parties
  * Timeline details (when things happened)
  * Broader significance or implications
- No two facts should cover the same aspect or share the same primary anchor
- Each fact must be indispensable — removing it should lose unique information

Format: One fact per line, numbered 1 through {k}, from earliest to latest in article position.
"""


FACT_CONTINUATION_ANCHORED = """You are extracting facts from a news article in article order. Your task is to find 1 new fact from a LATER part of the article than the facts already extracted.

REQUIREMENTS FOR YOUR NEW FACT:
1. Must come from a LATER POSITION in the article than all previously extracted facts
2. Must cover a DIFFERENT ASPECT of the story than any fact above. Aspects include:
   core event, key actors, statistics, quotes, locations, causes, consequences, reactions, timeline, significance
3. Must contain a UNIQUE ANCHOR not appearing in any fact above:
   - A person's name not yet mentioned, OR
   - A specific number/statistic not yet stated, OR
   - A location/organization not yet referenced, OR
   - A direct quote not yet cited
4. Must be a single concise sentence (under 35 words)
5. Must be indispensable — it adds information no other fact captures

PATTERNS TO AVOID:
- Never restate any previously extracted fact with different wording
- Never start with the same subject as any previous fact
- Never cover the same aspect as any previous fact, even with different details
- Never paraphrase or generalize information already captured

What is the next NEW ASPECT from a later part of the article?
Output only the text of the fact and nothing else.
"""

# Template for the continuation call
FACT_CONTINUATION_TEMPLATE = """{article}

FACTS ALREADY EXTRACTED ({n_total} total) — you must find something CATEGORICALLY DIFFERENT:
{facts_numbered}

ASPECTS ALREADY COVERED: {aspect_hints}

Extract ONE NEW fact that:
- Comes from a later part of the article than the facts above
- Covers an aspect NOT in the list above
- Is a single concise sentence (under 35 words)
- Contains specific names, numbers, dates, or locations from the article

Look for UNCOVERED aspects: statistics, quotes, reactions, locations, background, consequences, significance.

Continue with fact {next_number}:
"""


# =============================================================================
# SUMMARY GENERATION PROMPT
# =============================================================================

FACT_SUMMARY_STRICT = """Write a cohesive news summary from the provided list of facts.

STRUCTURE RULES:
1. One fact per sentence — never merge multiple facts into a single sentence
2. Preserve the EXACT ORDER of the provided list: fact 1 becomes sentence 1, fact 2 becomes sentence 2, etc.
3. Every anchor (name, number, date, location, quote) from each fact must appear in the corresponding sentence
4. Sentences should flow naturally with appropriate transitions, but transitions must not introduce new information

ANTI-REDUNDANCY RULES:
5. Never add introductory, concluding, or summary sentences
6. Never restate, echo, or paraphrase information from another sentence
7. Each sentence must be self-contained — a reader should be able to identify exactly ONE distinct piece of information per sentence
8. If two facts seem related, keep them as separate sentences without blending their content

Write the summary as flowing prose, one fact per sentence, maintaining the provided order:
"""


# =============================================================================
# DECOMPOSITION PROMPTS (for recovery)
# =============================================================================

FACT_DECOMPOSE = """Decompose the provided summary into exactly {num_facts} distinct facts based on sentence order.

REQUIREMENTS:
1. Extract exactly {num_facts} facts in the order they appear in the summary
2. Each fact corresponds to roughly one sentence in the summary
3. Each fact must be a single complete sentence containing at least one specific anchor (name, number, date, location, or quote)
4. Preserve the exact wording of all anchors — do not paraphrase names, numbers, or quotes
5. Each fact must capture a DIFFERENT ASPECT of the story (event, actor, statistic, quote, location, cause, consequence, reaction, etc.)
6. If a summary sentence contains information from two aspects, assign it to the dominant one
7. Do not merge or split information — maintain 1:1 correspondence with summary sentences

Output format: Facts separated by [sep], from first to last in summary order.
Example: Company X announced a $2B merger on Monday[sep]CEO Jane Smith called it transformative[sep]Shares rose 15% in after-hours trading
"""


FACT_SEGMENTATION = """You are a news analyst. The following summary was originally composed as exactly {n_units} sentences, each conveying one distinct factual aspect of the story (e.g., core event, key statistic, quote, reaction, consequence, background detail).

The text below is a paraphrased version that may have split, merged, or lightly restructured sentences, but the same {n_units} distinct factual aspects are present and their relative order is preserved.

Your task: segment this text into exactly {n_units} chunks, where each chunk captures all and only the information from one original factual aspect.

Rules:
- Output exactly {n_units} chunks in the same order as the summary's information flow.
- Every word in the text must appear in exactly one chunk. Do not add, remove, or rephrase any words — partition the text exactly as written.
- Each chunk should cover one distinct aspect: an event, a statistic, a quote, a reaction, a consequence, a background detail, etc. Use aspect boundaries as your guide for where to split.
- If a paraphrased sentence blends two original aspects, split it at the boundary between aspects.
- If multiple paraphrased sentences elaborate the same original aspect, concatenate them into one chunk.

Respond with a JSON object: {{"chunks": ["chunk1", "chunk2", ...]}}
Output only valid JSON, no preamble."""
