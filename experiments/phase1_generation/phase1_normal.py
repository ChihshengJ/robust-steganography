"""Phase 1b: length-matched normal generations for steganalysis.

A normal generation is what the task produces with no message: the sender's
synthesizer answers the task input directly. Each stegotext gets one, written
by its own synthesizer at its sampling (temperature, top_p), from the same
input, at its length. The classes then share writer, sampling, input and
length, and differ only in the steganographic scaffold, so a detector cannot
succeed by telling writers apart. Written next to the stegotexts, as
``{dir}/{system}_normal.jsonl``.

Writers miss a requested length, and a length gap alone separates the classes,
so a generation outside ``--length-tolerance`` of its target is retried with
the request scaled by how far the last attempt missed; the closest attempt is
kept and its ``length_ratio`` recorded.

Checkpoint-resumable: ids already written are skipped. Failed generations go
to ``{system}_normal_failures.jsonl`` and are retried on the next run.

Usage:
    # A normal generation for every stegotext in the matching dirs:
    python -m experiments.phase1_generation.phase1_normal --system story \\
        --dirs 'story_cap*'
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1_generation.phase1_generate import make_text_record
from experiments.utils.api import chat
from experiments.utils.configs import CONFIG_SYSTEMS, stego_config
from experiments.utils.io import (
    append_jsonl,
    load_completed_ids,
    make_record_id,
    read_jsonl,
)
from experiments.utils.system_factory import make_client
from experiments.utils.token_counter import count_words, round_words
from systems.config.litreview_prompts import GENERATE_REVIEW_NORMAL
from systems.config.story_prompts import STORY_NORMAL_PROMPT
from systems.core.litreview import load_corpus, prepare_references
from systems.paths import litreview_references

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# The synthesizers' completion budgets (StorySystem / LitReviewSystem).
# Same caps as the stegotexts' writers, which never bind: normal texts are
# length matched to stegotexts that can run past 2000 words at large F.
# Room for a reasoning writer's thinking before the text.
MAX_TOKENS = {"story": 16000, "litreview": 16000}


def normal_path(directory: Path, system: str) -> Path:
    return directory / f"{system}_normal.jsonl"


# ---------------------------------------------------------------------------
# Inputs and prompts
# ---------------------------------------------------------------------------


def task_input(system: str, stego: dict, corpus: list[dict] | None) -> dict:
    """The task input x of a stegotext, plus what its normal prompt needs."""
    if system == "story":
        return {"premise": stego["prompt"]}
    corpus_idx = int(stego["system_state"]["corpus_idx"])
    return {
        "corpus_idx": corpus_idx,
        "paper": corpus[corpus_idx],
        "n_refs": len(stego["metadata"]["selected_refs"]),
    }


def normal_messages(system: str, x: dict, requested_words: int) -> list[dict]:
    """Chat messages for a normal generation, in the layout the synthesizer
    uses for the stegotext (see StorySystem._generate_story and
    LitReviewSystem._generate_review)."""
    if system == "story":
        prompt = STORY_NORMAL_PROMPT.format(
            target_words=requested_words, premise=x["premise"]
        )
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    paper = x["paper"]
    refs = "\n".join(
        f"  - {r['author_text']} ({r['year']}). {r['ref_title']}"
        for r in prepare_references(paper["references"])
    )
    instructions = GENERATE_REVIEW_NORMAL.format(
        seed_title=paper["title"],
        seed_abstract=paper.get("abstract", "")[:600],
        n_refs=x["n_refs"],
        target_words=requested_words,
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"References:\n{refs}"},
    ]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def find_stego_dirs(phase1_root: Path, system: str, patterns: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for pattern in patterns:
        for d in sorted(phase1_root.glob(pattern)):
            if (d / f"{system}_stego.jsonl").exists() and d not in dirs:
                dirs.append(d)
    return dirs


def normal_jobs(system: str, stego_dir: Path, corpus) -> list[dict]:
    """One job per stegotext in ``stego_dir``: its own synthesizer and sampling."""
    jobs = []
    for stego in read_jsonl(stego_dir / f"{system}_stego.jsonl"):
        config = stego_config(stego)
        jobs.append(
            {
                "id": make_record_id(system, "normal", stego["prompt_idx"]),
                "out_dir": stego_dir,
                "system": system,
                "prompt_idx": stego["prompt_idx"],
                "prompt": stego["prompt"],
                "input": task_input(system, stego, corpus),
                "m": len(stego["message_bits"]),
                "target_words": stego["word_count"],
                "paired_stego_id": stego["id"],
                "writer": {
                    "provider": config["synth_provider"],
                    "model": config["synth_model"],
                    "temperature": config["synth_temperature"],
                    "top_p": config["synth_top_p"],
                    # The stegotext's own request body (its reasoning setting).
                    "extra_body": config.get("synth_extra_body"),
                    "max_tokens": MAX_TOKENS[system],
                },
            }
        )
    return jobs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(job: dict, client, tolerance: float, max_attempts: int) -> dict:
    """Generate one normal text, retrying toward the target length."""
    writer = job["writer"]
    target = job["target_words"]
    requested = round_words(target, step=10)
    best = None
    for attempt in range(1, max_attempts + 1):
        text = chat(
            client,
            writer["model"],
            normal_messages(job["system"], job["input"], requested),
            # None: the writer's default sampling (it rejects the parameter).
            **{k: writer[k] for k in ("temperature", "top_p") if writer[k] is not None},
            max_tokens=writer["max_tokens"],
            **({"extra_body": writer["extra_body"]} if writer["extra_body"] else {}),
        )
        words = count_words(text)
        if best is None or abs(words - target) < abs(best["words"] - target):
            best = {
                "text": text,
                "words": words,
                "requested": requested,
                "attempt": attempt,
            }
        if abs(words / target - 1) <= tolerance:
            break
        requested = max(10, round_words(requested * target / max(words, 1), step=10))

    x = job["input"]
    metadata = {
        "writer_model": writer["model"],
        "writer_provider": writer["provider"],
        "temperature": writer["temperature"],
        "top_p": writer["top_p"],
        "extra_body": writer["extra_body"],
        "m": job["m"],
        "requested_words": best["requested"],
        "attempts": attempt,
        "kept_attempt": best["attempt"],
        "length_ratio": best["words"] / target,
    }
    if job["system"] == "litreview":
        metadata |= {"corpus_idx": x["corpus_idx"], "n_refs": x["n_refs"]}
    return make_text_record(
        record_id=job["id"],
        system=job["system"],
        text_type="normal",
        prompt_idx=job["prompt_idx"],
        prompt=job["prompt"],
        text=best["text"],
        message_bits=None,
        system_state=None,
        metadata=metadata,
        length_target=target,
        paired_stego_id=job["paired_stego_id"],
    )


def run_jobs(jobs: list[dict], workers: int, tolerance: float, max_attempts: int):
    clients = {
        p: make_client(p) for p in sorted({j["writer"]["provider"] for j in jobs})
    }
    n_ok = n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                generate,
                job,
                clients[job["writer"]["provider"]],
                tolerance,
                max_attempts,
            ): job
            for job in jobs
        }
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                record = fut.result()
            except Exception as e:
                n_failed += 1
                log.warning(f"  FAILED {job['out_dir'].name}/{job['id']}: {e!r}")
                append_jsonl(
                    job["out_dir"] / f"{job['system']}_normal_failures.jsonl",
                    {
                        "id": job["id"],
                        "writer_model": job["writer"]["model"],
                        "error": repr(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                continue
            append_jsonl(normal_path(job["out_dir"], job["system"]), record)
            n_ok += 1
            ratio = record["metadata"]["length_ratio"]
            log.info(
                f"  {job['out_dir'].name}/{job['id']}: {record['word_count']} words "
                f"(target {job['target_words']}, ratio {ratio:.2f})"
            )
    return n_ok, n_failed


def length_report(paths: list[Path], tolerance: float) -> None:
    for path in paths:
        ratios = [r["metadata"]["length_ratio"] for r in read_jsonl(path)]
        if not ratios:
            continue
        off = sum(1 for r in ratios if abs(r - 1) > tolerance)
        mean_abs = sum(abs(r - 1) for r in ratios) / len(ratios)
        log.info(
            f"{path.parent.name}: {len(ratios)} normal texts, mean |ratio - 1| "
            f"{mean_abs:.3f}, {off} outside ±{tolerance:g}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=CONFIG_SYSTEMS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Glob patterns under phase1_texts/ selecting the stego dirs, "
        "e.g. 'story_cap*'.",
    )
    parser.add_argument(
        "--length-tolerance",
        type=float,
        default=0.15,
        help="Accept a generation within this relative distance of its target "
        "length (default 0.15).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Generations per text while outside the tolerance (default 3).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the pending jobs and one prompt, with no API calls.",
    )
    args = parser.parse_args()

    phase1_root = args.data_dir / "phase1_texts"
    stego_dirs = find_stego_dirs(phase1_root, args.system, args.dirs)
    if not stego_dirs:
        parser.error(f"no {args.system}_stego.jsonl under {phase1_root}/{args.dirs}")
    log.info(f"Stego dirs: {[d.name for d in stego_dirs]}")

    corpus = (
        load_corpus(*litreview_references()) if args.system == "litreview" else None
    )

    jobs = [j for d in stego_dirs for j in normal_jobs(args.system, d, corpus)]

    out_paths = sorted({normal_path(j["out_dir"], args.system) for j in jobs})
    completed = {p: load_completed_ids(p) for p in out_paths}
    pending = [
        j
        for j in jobs
        if j["id"] not in completed[normal_path(j["out_dir"], args.system)]
    ]
    log.info(f"{len(jobs)} normal texts planned, {len(pending)} pending")

    if args.dry_run:
        by_writer = Counter(
            f"{j['writer']['provider']}:{j['writer']['model']}" for j in pending
        )
        for key, n in sorted(by_writer.items()):
            log.info(f"  {key}: {n}")
        if pending:
            j = pending[0]
            for msg in normal_messages(
                args.system, j["input"], round_words(j["target_words"], step=10)
            ):
                print(f"--- {msg['role']} ---\n{msg['content'][:1500]}")
        return

    n_ok, n_failed = run_jobs(
        pending, args.workers, args.length_tolerance, args.max_attempts
    )
    log.info(f"Generated {n_ok}, failed {n_failed} (rerun to retry failures)")
    length_report(out_paths, args.length_tolerance)


if __name__ == "__main__":
    main()
