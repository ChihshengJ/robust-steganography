from functools import cache
from typing import Iterator, Tuple

import nltk


# Academic / common abbreviations whose trailing period should NOT end a sentence.
# Keeps citations like "Wu et al. (2018)" intact during sentence splitting.
_SENTENCE_ABBREVS = {
    "et al", "al", "e.g", "i.e", "cf", "fig", "figs", "eq", "eqs",
    "no", "nos", "vol", "vols", "pp", "eds", "ed", "inc", "ltd", "co", "corp",
    "ref", "refs", "sec", "secs", "ch", "app", "approx", "vs", "etc",
}


@cache
def _sentence_tokenizer():
    """Lazily load Punkt and extend it with academic abbreviations."""
    try:
        tok = nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)
        tok = nltk.data.load("tokenizers/punkt/english.pickle")
    tok._params.abbrev_types.update(_SENTENCE_ABBREVS)
    return tok


def iter_sentences_with_gaps(text: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(gap_before, sentence)`` pairs covering ``text`` exactly.

    Concatenating ``gap + sentence`` across all yielded pairs reconstructs
    ``text`` byte-for-byte. Uses NLTK Punkt extended with academic
    abbreviations so citations like "Wu et al. (2018)" stay within a single
    sentence rather than fragmenting at the "et al." period.
    """
    tok = _sentence_tokenizer()
    cursor = 0
    for start, end in tok.span_tokenize(text):
        yield text[cursor:start], text[start:end]
        cursor = end
    if cursor < len(text):
        yield text[cursor:], ""


class Attack:
    def __init__(
        self,
    ): ...

    def _resolve_local_mode(self, local: bool, tampering: float) -> bool:
        """
        Resolve whether to use local mode based on initialization setting and call parameters.

        Arguments:
            local: The local parameter passed to __call__
            tampering: The tampering percentage (0.0 to 1.0)

        Returns:
            True if local mode should be used, False for global mode.
            If None, use global only for tampering == 1
        """
        if local is not None:
            return local
        else:
            return False if tampering < 0.99 else True

    def __call__(
        self,
        text: str,
        tampering: float,
        local: bool,
    ) -> str:
        raise NotImplementedError


class NullAttack(Attack):
    def __init__(self):
        super().__init__()

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        return text
