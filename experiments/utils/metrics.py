"""Evaluation metrics: BERTScore, cosine similarity, BLEU, TER, bit error rate."""

import numpy as np

_SBERT_MODEL = None


def bertscore(reference: str, hypothesis: str, lang: str = "en") -> dict:
    """Compute BERTScore P/R/F1 between reference and hypothesis."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn([hypothesis], [reference], lang=lang, verbose=False)
    return {
        "precision": P[0].item(),
        "recall": R[0].item(),
        "f1": F1[0].item(),
    }


def cosine_similarity(
    text_a: str, text_b: str, model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """Cosine similarity of sentence-transformer embeddings."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SBERT_MODEL = SentenceTransformer(model_name)

    embeddings = _SBERT_MODEL.encode([text_a, text_b])
    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score using sacrebleu."""
    import sacrebleu

    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score


def ter(reference: str, hypothesis: str) -> float:
    """Compute Translation Edit Rate using sacrebleu."""
    import sacrebleu

    result = sacrebleu.sentence_ter(hypothesis, [reference])
    return result.score


def bit_error_rate(original: list[int], recovered: list[int]) -> dict:
    """Compute bit error rate between original and recovered bit sequences.

    Returns {"ber": float, "bitwise_accuracy": float, "num_errors": int, "perfect": bool}.
    """
    if len(original) != len(recovered):
        return {
            "ber": 1.0,
            "bitwise_accuracy": 0.0,
            "num_errors": max(len(original), len(recovered)),
            "perfect": False,
        }
    errors = sum(o != r for o, r in zip(original, recovered))
    n = len(original)
    ber = errors / n if n > 0 else 0.0
    return {
        "ber": ber,
        "bitwise_accuracy": 1.0 - ber,
        "num_errors": errors,
        "perfect": errors == 0,
    }
