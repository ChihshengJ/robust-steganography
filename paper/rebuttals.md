# Reviewer Au6T

**Weakness 1: how encoding and decoding work.** A working example for SG. The inputs are a public premise $x$ ("A food critic loses her sense of taste...") and a payload $h = 0101...$

- _Encode_ The sender asks a **scaffold oracle** (a pinned LLM, temperature 0) for $m$ plot slots, each with two mutually exclusive variants: "where she hides the file" -> ($u$: she hides it in the drawer / $u'$: she hides it under the table). The key $k$ fixes the slot order and which variant of each pair is labeled 0 vs. 1. Bit $i$ picks one variant of pair $i$; the $m$ selected units $u_h$ go to a generator LLM, which writes **one** coherent story realizing exactly those choices.
- _Decode_ The receiver holds only $x$ and $k$. She re-queries the same oracle, regenerating the same pair list, re-derives order and labels from $k$, then asks a model, for each pair, a binary question about the received text $s'$: "which variant appears?" That answer is bit $i$.

The oracle plays the role a shared PRG plays in classical schemes: determinism is a correctness requirement. No token probabilities are involved, so nothing depends on surface form surviving. QA is identical with subtopics as units; LR needs no oracle, its pairs coming from a keyed HMAC over the seed paper's bibliography $ref(x)$.

**Weakness 2: motivation and novelty.** Our channel lives in discourse-level semantics, not token-level statistics. Meteor, Discop and form-based watermarks hide bits in the sampling distribution, which is trivial for semantic preserving attacks to take effect (see the general comment on the baselines). Of the semantically-robust schemes, Bai et al. (2024) needs a shared codebook and an agent pipeline, and Perry et al. (2025) is limited to very low capacity. The symmetric key is what buys the difference: sender and receiver share only $k$ and the public input $x$, so the channel fits inside an ordinary API call, which is also what makes it a threat model for AI safety.

**Weakness 3: coverage.** The attack suite has a variety of configurations: 3 mechanisms (synonym substitution, paraphrase, round-trip translation) * 2 scopes (local/global), local attacks swept at tampering rates 0.2/0.5/1.0, stochastic attacks run 3 times. Per stego-text that is 27 attacked texts + 1 clean decode, so it’s **840 decodes per configuration** (7,560 total).

On message length, 18 is the per-system structural ceiling we get from the results:

| System | Ceiling set by      | Swept          | Bitwise, global paraphrase   |
| ------ | ------------------- | -------------- | ---------------------------- |
| QA     | #subtopics / 2      | 6, 8, 10       | 100 -> 99.4 -> 96.3          |
| SG     | #plot slots         | 14, 16, 18     | 97.5 -> 97.7 -> 98.0         |
| LR     | #refs in seed paper | 14, 16, 18, 20 | 97.6 -> 96.6 -> 94.8 -> 92.8 |

- **QA**'s ceiling is semantic: one question supports only so many distinguishable subtopics. 10 bits is past the cliff: perfect recovery under global paraphrase falls to 56.7% even as bitwise holds at 96.3%.
- **SG**'s ceiling is the scaffold generator, not the scheme. Both parties must derive an identical slot list, so we request capacity+2 slots of headroom; at 18 bits the 4B model already returns only 19.87/20 on average (min 19), and beyond that the list is shorter than the payload, so decoding fails before any attack. Bitwise is flat across 14-18, so a stronger open-weights model raises this ceiling.
- **LR**'s ceiling is its decoder, not its robustness. At 18 bits it recovers 73.3% perfectly under global paraphrase against 80.0% on the clean channel: a 6.7-point attack cost, versus 30 for SG and 33 for QA. What LR loses is lost mostly before any attacker touches it, to citation-extraction precision, which we are refining.

18 bits is also per-document, not a message-length cap: longer payloads chunk across documents, at the cost of more cover text.

On variance, the scaffold is deterministic **by construction** (temperature 0, top_p 1.0, pinned single-stream server); had it varied, decoding would break outright, and we verify exact list-match on every run. Residual spread is across the 30 stego-texts: bitwise s.d. under global paraphrase is 0.0-5.5 points for QA/SG and 5.1-11.0 for LR. Perfect recovery is a per stego-text rate over n=30 (a message counts only if all three attack runs decode it exactly). We will add the s.d. to the figures, plus a scaffold-sensitivity study across local models in the revision.

**Weakness 4: self-containedness.** We accept this. In revision we will add a notation table, define cover-text/stego-text, payload, and bitwise vs. perfect recovery on first use, and move the example above into §3.2.

**Figure 2.** We will re-render it as a vector PDF with distinct accessible hues for P (local) and P (global), and give LR its own x-axis.

# Reviewer wuHb

We thank the reviewer for the careful assessment and we ran new experiments to address the concerns.

**Weakness 1: Baselines** We added **Meteor** and **Discop** (GPT-2) and ran them through the identical attack/decode pipeline at matched payloads (14/16/18 bits; 30 stego-texts each, 3 runs per attack, n=90). Token-level schemes have no notion of the task, so length is set by the payload alone: natively, 18 bits is 3-4 words. We report two configurations. Length-matched (payload inflated with a repetition code to reach our ~575 words):

| System (18bits)      | No attack (bit/perfect) | Global paraphrase (bit/perfect) |
| -------------------- | ----------------------- | ------------------------------- |
| Meteor (644 words)   | 0.754 / 0.333           | 0.502 / 0.000                   |
| Discop (502 words)   | 0.765 / 0.200           | 0.499 / 0.000                   |
| Ours, SG (578 words) | 0.989 / 0.900           | 0.980 / 0.767                   |
| Ours, LR (622 words) | 0.941 / 0.800           | 0.948 / 0.811                   |

_Native-rate_ (3-4-word stego-texts): both are lossless absent attack (1.000/1.000), yet still collapse to 0.000 perfect recovery under global paraphrase.

Under paraphrase both baselines sit at chance bitwise (0.48-0.51) with 0 perfect recoveries at every payload in both configurations. Matching our length is itself costly: it takes ~190× payload inflation, which alone degrades clean-channel recovery to 0.75 bitwise and yields incoherent prose as a byproduct.

**Weakness 2: forced semantic injection** this misunderstanding might result completely from a typo we made: in §3.4, we defined $ref_1(x) = (u ∈ U : b(u)=1)$ where it should read $ref(x)$, not $U$. Both streams partition $ref(x)$ which is the seed paper's own bibliography. The encoder selects a subset and an ordering of that list and can never introduce a paper from outside the context, so the failure mode described is not possible. And thus, for QA the inventory is always dynamic to the input question. The generator lists topics that "a comprehensive answer could address" for this question, and the encoder picks between two alternatives the model itself proposed, which reweights which topics appear. This reweighting is exactly the channel, inherent to any steganography, but always between contextually licensed options.

On the other hand, the 66.5% perplexity classifier does not measure degradation. Per `Table 1`, on SG our stego-texts have **lower** GPT-2 perplexity than type-2 covers (18.69 vs 21.27), as on LR (14.40 vs 15.15). The classifier detects a shift in either direction, so reading 66.5% as quality loss would equally commit one to calling our texts more fluent than unconstrained LLM output.

**Weakness 3: cover-text definition** Type 2 is already the unconstrained single-stage output described in your comment. The same LLM is prompted directly with `x` (premise / question / paper title) and writes in one pass. Its only intervention is a target length, removing a confound that would otherwise separate the classes trivially; we will state this explicitly. We admit the prompts for type-2 covers are not "write a story about <premise>", but the payload doesn't specify the model used to synthesize the response, so there is noway for the monitor to know what distribution to compare it against.

**Weakness 4: text quality** We agree that perplexity cannot detect text that are fluent but incoherent. But we've already used LLM-as-judge to classify stego-texts and type-2 cover texts and yielded at-chance accuracy. If we assume that the cover texts, generated by GPT-4.1 are mostly coherent, the stego-text should be at least on par with them. If this result is not robust enough, we will add task specific LLM-as-judge or human evaluation in the revision.

## Reviewer 5ePE

We thank the reviewer for the positive feedback and for acknowledging the importance of our work. We believe this is a new route to covert communication under strong censorship, and also a new threat model for agentic systems.

**Weakness 1: baselines.** We added Meteor and Discop (GPT-2) and pushed them through the identical attack and decode pipeline at matched payloads (14/16/18 bits, 30 stego-texts each, 3 runs per stochastic attack, $n=90$). Token-level schemes have no notion of the task, so their length is set by the payload alone: 18 bits is natively 3 to 4 words. We therefore report two configurations.

_Length-matched_ (payload inflated with a repetition code to reach our $\approx 575$ words):

| System (18 bits)     | No attack (bit/perfect) | Global paraphrase (bit/perfect) |
| -------------------- | ----------------------- | ------------------------------- |
| Meteor (644 words)   | 0.754 / 0.333           | 0.502 / 0.000                   |
| Discop (502 words)   | 0.765 / 0.200           | 0.499 / 0.000                   |
| Ours, SG (578 words) | 0.989 / 0.900           | 0.980 / 0.767                   |
| Ours, LR (622 words) | 0.941 / 0.800           | 0.948 / 0.811                   |

_Native-rate_ (3 to 4 word stego-texts): both baselines are lossless with no attack (1.000 / 1.000), and still fall to 0.000 perfect recovery under global paraphrase.

In both configurations, at every payload, both baselines sit at chance for bitwise accuracy (0.48 to 0.51) with zero perfect recoveries under any semantic-preserving attack. Matching our cover length is itself costly: it needs about $190\times$ payload inflation, which alone drops clean-channel recovery to 0.75 bitwise and produces incoherent prose as a byproduct. We will add these results to the paper in the revision.

**Weakness 2: similarity metrics.** In our setting the stego-text is generated, so there is no paired cover to compare against, and LLM sampling means two stego-texts for the same input are not expected to resemble each other either, so a similarity score would have no fixed reference point.

The closest analogue to a cover text is our type-2 cover. Generated by the same model, same public input $x$, same target length, written in one pass with no payload. Against that reference, we read imperceptibility as “no classifier can separate stego-texts from covers produced in the same context,” which is what our steganalysis measures. Both signals point the same way. The GPT-2 perplexity of our stego-texts is even lower than that of type-2 covers (SG: 18.69 vs 21.27; LR: 14.40 vs 15.15), and an LLM-as-judge asked to tell the two classes apart performs at random. If this is insufficient, we are happy to add a task-specific quality judge or a small human evaluation in the revision.

**Comment: how the key controls the semantic units.** This is under-explained in the main text and we will fix it. Take SG for an example, both parties see the public premise $x$. The sender queries a pinned LLM for $m$ plot slots, each slot holding two mutually exclusive variants. The key $k$ fixes the order of the slots, and which variant in each pair is labeled 0 and which is labeled 1. Bit $i$ then selects one variant of pair $i$, and the generator writes one coherent text realizing exactly those $m$ choices.

The receiver holds only $x$ and $k$. They re-queries the same LLM with the same $x$, which regenerates the same pair list, re-derives the order and the labels from $k$, and then asks a model, for each pair, a binary question about the received text: "which variant appears?" That answer is bit $i$. The pairs are reproduced by determinism of the oracle LLM, not by the key; the key only permutes and labels them, exactly as a shared PRG does in classical schemes. QA works the same way with subtopics as units, and LR needs no oracle LLM at all: its pairs come from a keyed HMAC over the seed paper's own bibliography $ref(x)$.
