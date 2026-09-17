"""Clean-channel gate for the SyncPool baselines — run this before Phase 1.

`paper/revision-baselines/ECC_AND_RATE_PLAN.md` §6 makes SyncPool a hard
precondition for the length-matched Discop runs, with a pass/fail
criterion: **0% clean text-channel error**. The reason is arithmetic. At the
length-matched repetition rates the clean channel has to be exact over thousands
of encoded bits across hundreds of tokens, and with a per-token segmentation
ambiguity of epsilon the probability of an exact undisambiguated decode is
(1-epsilon)^n_tokens — at the epsilon ~= 0.05 Qi et al. measure for GPT-2 that is
~1e-17 over 750 tokens. Either disambiguation works perfectly or the t=0 point of
the whole figure is not 1.000, and every attacked point is measured against a
ceiling the scheme never reaches.

It also keeps the figure honest: attacked text arrives as a *string*, so every
attacked point must be text-channel. Without SyncPool the clean point is either
text-channel (and starts below 1, inviting "your baseline was broken") or token
channel (1.00, but then clean and attacked points sit on different channels in
one figure).

Usage:

    python -m experiments.utils.syncpool_gate                    # both systems
    python -m experiments.utils.syncpool_gate --system discop    # Qi et al.'s own
    python -m experiments.utils.syncpool_gate --repetitions 125 --n 3

Exits non-zero if any configured system fails, so it can gate a pipeline script.
The default rate is deliberately small: this checks *correctness*, and a desync
shows up at r=5 as surely as at r=125, far faster.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from experiments.utils.system_factory import make_discop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Short, varied, and unrelated to the paper's prompt files so the gate runs
# standalone. Content is irrelevant here — only the round trip is under test.
SEEDS = [
    "The following is a short article about renewable energy policy.",
    "A brief report on urban transit planning follows.",
    "What follows is a short essay on coastal erosion.",
    "Below is a summary of the quarterly maintenance review.",
    "This note describes the changes to the grant application process.",
]


def run_system(
    name: str,
    capacity: int,
    repetitions: int,
    n: int,
    syncpool: bool,
) -> bool:
    """Encode/decode `n` payloads on both channels. True iff the text channel is exact."""
    # Generous budget: SyncPool lowers bits/token, so the same payload needs
    # more of them, and a truncated generation would fail the gate for a
    # reason that has nothing to do with disambiguation.
    system = make_discop(repetitions=repetitions, max_length=20000, syncpool=syncpool)

    rng = np.random.default_rng(20260904 + capacity)
    n_text_ok = n_token_ok = 0
    totals: dict[str, int] = {}
    n_tokens = []

    for i in range(n):
        bits = rng.integers(0, 2, size=capacity).tolist()
        seed = SEEDS[i % len(SEEDS)]
        t0 = time.time()
        text = system.hide_message(bits, seed)
        meta = system._last_metadata
        n_tokens.append(meta["n_tokens"])

        token_stats: dict = {}
        text_stats: dict = {}
        token_ok = list(system.recover_message(text, token_ids=meta["token_ids"], stats=token_stats)) == bits
        text_ok = list(system.recover_message(text, stats=text_stats)) == bits
        n_token_ok += token_ok
        n_text_ok += text_ok
        for key, value in text_stats.items():
            totals[key] = totals.get(key, 0) + value

        log.info(
            "  %s %d/%d: %d tokens, %d words, %.0fs | token channel %s | text channel %s",
            name, i + 1, n, meta["n_tokens"], len(text.split()), time.time() - t0,
            "OK" if token_ok else "FAIL", "OK" if text_ok else "FAIL",
        )
        if not text_ok:
            log.error("    text-channel stats: %s", text_stats)

    mean_tokens = sum(n_tokens) / len(n_tokens)
    log.info(
        "%s: token channel %d/%d, text channel %d/%d | %.1f tokens/doc, "
        "%.3f bits/token at r=%d | text-channel fallbacks: %s",
        name, n_token_ok, n, n_text_ok, n, mean_tokens,
        (capacity * repetitions) / mean_tokens, repetitions,
        {k: v for k, v in totals.items() if k != "n_steps"},
    )
    return n_text_ok == n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--system",
        choices=["discop"],
        default="discop",
        help=(
            "Which baseline to gate. Discop is the scheme Qi et al. validated "
            "SyncPool on, so a failure here is a bug in our implementation."
        ),
    )
    parser.add_argument("--capacity", type=int, default=14)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help=(
            "Repetition rate (default: 5). The gate tests correctness, and a "
            "desync is just as visible at r=5 as at the length-matched rate but "
            "orders of magnitude faster to run."
        ),
    )
    parser.add_argument("--n", type=int, default=3, help="Payloads per system (default: 3).")
    parser.add_argument(
        "--syncpool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --no-syncpool to see the failure the gate exists to catch.",
    )
    args = parser.parse_args()

    names = [args.system]
    failed = []
    for name in names:
        log.info("=== %s (syncpool=%s) ===", name, args.syncpool)
        if not run_system(name, args.capacity, args.repetitions, args.n, args.syncpool):
            failed.append(name)

    if failed:
        log.error(
            "GATE FAILED for %s: the clean text channel is not exact. Do not run the "
            "length-matched grid — at those rates the t=0 point will not be 1.000.",
            ", ".join(failed),
        )
        return 1
    log.info("GATE PASSED: clean text-channel decode is exact for %s.", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
