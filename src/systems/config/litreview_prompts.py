EXTRACT_CITATIONS = """You are given a literature review passage. Extract every referenced work as a (last_name, year) pair.

Rules:
1. For each reference mentioned, output the first author's last name and the publication year
2. Handle ALL citation formats: "Smith et al. (2020)", "Smith and colleagues in 2020", "Smith and Jones (2020)", "the work of Smith in 2020", etc.
3. Use only the FIRST author's last name — ignore co-authors
4. Each reference should appear exactly once even if cited multiple times
5. Sort by (year ascending, then last name alphabetically)

Output format: one pair per line as "LastName YEAR", nothing else.
Example:
Smith 2018
Jones 2019
Li 2020"""

GENERATE_REVIEW = """You are writing the Related Work section of an academic paper.

Paper: "{seed_title}"
Abstract: {seed_abstract}

Write a Related Work section that contextualizes this paper within the broader 
research landscape. Organize thematically, grouping related works by research 
direction or methodology across multiple paragraphs.

Where works are closely related, discuss them together in the same sentence or 
passage rather than giving each its own isolated sentence. Include contextual 
sentences that provide background or transitions without citing specific papers.
Some works may warrant more discussion than others depending on their relevance.

Cite as "LastName (YEAR)" or "LastName et al. (YEAR)".
All provided references must be cited."""

## Legacy ##

BASE_GENERATION = """You are writing a related work section for an academic paper. Generate exactly {k} sentences, one for each reference listed below, in the given order.

Requirements:
- Each sentence MUST cite its reference as "LastName et al. (YEAR)" or "LastName (YEAR)"
- Describe a plausible contribution of the referenced paper based on its title
- Each sentence must be self-contained, 15-35 words, and sound natural in academic prose
- Each sentence must focus on a DIFFERENT technical aspect — vary the framing (e.g., method, dataset, evaluation, theoretical insight)
- Do NOT start multiple sentences with the same grammatical structure

Output exactly {k} numbered sentences, one per line."""

SENTENCE_CONTINUATION = """You are continuing a related work section. Write ONE plausible sentence about the given reference.

Requirements:
- Describe a plausible contribution of the reference based on its title
- Write it as a continuation of the previous sentences so they can form a literature review
- Cite it as "LastName et al. (YEAR)" or "LastName (YEAR)" — the citation MUST appear
- The claim must be clearly DISTINCT from every previous claim:
  * Different sentence structure and opening
  * Different technical framing (method, dataset, finding, application, comparison, limitation, formulation)
  * No echoed phrasing or vocabulary from previous sentences
- 15-35 words, self-contained, natural academic tone

Output only the sentence, nothing else."""

SENTENCE_CONTINUATION_TEMPLATE = """Paper: "{seed_title}"
Abstract: {seed_abstract}

Reference to describe:
  {author_text} ({year}). "{ref_title}"

Previous sentences in this review ({n_previous}):
{previous_sentences}

Write the next sentence:"""

DECOMPOSE = """You are given a literature review passage. Extract each distinct reference discussion as a separate span.
There should be {expected_total} spans in total.

Rules:
1. Each span must discuss exactly ONE reference (identified by an "AuthorName (YEAR)" or "AuthorName et al. (YEAR)" citation)
2. Copy each span VERBATIM from the passage — do not add, remove, or change any words
3. If consecutive text discusses the same reference, combine it into one span
4. If a sentence discusses multiple references, assign it to the reference whose citation appears first

After extraction, sort the spans by (year ascending, then author name initial alphabetically).

Output format: sorted spans separated by [sep]
Example: Smith et al. (2018) proposed X for Y.[sep]Jones (2019) extended this by Z.[sep]Li et al. (2020) achieved W."""

DECOMPOSE_CLAIM = """You are given a literature review passage. Extract each distinct reference discussion as a separate claim on who did what with what approach.

Rules:
1. Each claim must discuss exactly ONE reference (identified by an "AuthorName (YEAR)" or "AuthorName et al. (YEAR)" citation)
2. Simplify the sentence talking about that reference to a claim on who did what with what approach
3. If consecutive text discusses the same reference, combine it into one claim about that reference
4. If a sentence discusses multiple references, decompose them to their respective claims with regrad to the each reference

After extraction, sort the claims by (year ascending, then author name initial alphabetically).

Output format: sorted claims separated by [sep]
Example: Smith et al. (2018) proposed X for Y.[sep]Jones (2019) extended this by Z.[sep]Li et al. (2020) achieved W."""

GENERATE_REVIEW_BY_SENTENCES = """You are writing a related work section for an academic paper. Generate exactly {k} sentences, one for each reference listed below, in the given order.

Requirements:
- Each sentence MUST cite its reference as "LastName et al. (YEAR)" or "LastName (YEAR)"
- Describe a plausible contribution of the referenced paper based on its title
- Each sentence must be self-contained, 15-35 words, and sound natural in academic prose
- Each sentence must focus on a DIFFERENT technical aspect — vary the framing (e.g., method, dataset, evaluation, theoretical insight)
- Do NOT start multiple sentences with the same grammatical structure

Output exactly {k} numbered sentences, one per line."""

SYNTHESIZE = """You are given some claims generated from research references, each of them describes what the researchers did based on an actual reference.
Your goal is to form a seemingly convincing literature review that could appear in this paper:
Paper title: {seed_title}
Paper Abstract: {seed_abstract}

Requirements:
- Synthesize a plausible literature review section that can fit into the paper mentioned above.
- Do NOT alter the wording of the claims, do NOT add new information, miss any information, or change the angle of a claim, i.e., if Smith et al. (2022) made a dataset for evaluation, do NOT change it to Smith et al. (2022) invented a new evaluation scheme.
- You can add more connectives or phrases that fuse them better as a paragraph that appears more natural in an academic paper.
- Make sure every single claim gets represented separately in the output, NO overlap, NO merging, keep every claim separate.
- Every citation must appear exactly as given — do NOT alter author names, years, or citation format.
"""

