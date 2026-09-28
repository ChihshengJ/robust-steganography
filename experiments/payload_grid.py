"""Run the payload grid (ARR October revision): every cell, stage by stage.

A cell is one task x one model setup x one frame size F. Detection and recovery
texts are generated separately (REVISION_PLAN_ARR_OCT.md §1 "Runs"):

    detection  SG: 2 G x 2 writers x 4 F = 16 cells   (G on OpenRouter, per call)
               LR: 2 writers x 4 F        =  8 cells
    recovery   SG: pinned G x DeepSeek V4.1 Flash x 4 F = 4 cells
               LR: DeepSeek V4.1 Flash x 4 F           = 4 cells

Stages, in order (each resumable; rerun a stage to retry what failed):

    generate  Phase 1 stegotexts on N_INPUTS inputs per cell      (API; recovery
              SG needs the pinned G server at LOCAL_BASE_URL, serving LOCAL_MODEL)
    select    the first 30 inputs that encoded in every cell of a task, both tracks
    normal    length-matched normal generations, detection cells   (API)
    attack    Phase 3 on the recovery cells                        (API)
    decode    Phase 4a/4b on the recovery cells                    (API; SG needs
              the pinned G server)
    analyze   steganalysis + quality (detection), recovery CSV and capacity
              table (recovery)

Cells of a stage run in parallel (--parallel), each logging to
data/experiments/logs/payload_grid/{stage}/{cell}.log.

The Discop baseline runs from scripts/baselines_syncpool.sh on its own prompts.

Usage:
    python -m experiments.payload_grid generate --dry-run
    python -m experiments.payload_grid generate --track detection --parallel 8
    python -m experiments.payload_grid select
    python -m experiments.payload_grid all --systems litreview
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from experiments.utils.configs import config_tag
from experiments.utils.io import TRACKS
from experiments.utils.system_factory import LOCAL_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEMS = ("story", "litreview")
GRID_F = (8, 16, 24, 32)
N_INPUTS = 40  # generated per cell; select keeps the first 30 shared ones
N_KEEP = 30
LITREVIEW_INDICES = "litreview_indices_min80.json"

# provider:model
WRITERS = ("openai:gpt-6-sol", "openrouter:deepseek/deepseek-v4.1-flash")
DETECTION_GENERATORS = ("openrouter:qwen/qwen3.5-9b", "openrouter:google/gemma-4-31b-it")
RECOVERY_WRITER = "openrouter:deepseek/deepseek-v4.1-flash"
RECOVERY_GENERATOR = "local"  # the pinned server: LOCAL_BASE_URL serving LOCAL_MODEL

STAGES = ("generate", "select", "normal", "attack", "decode", "analyze")


def _split(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"expected provider:model, got {spec!r}")
    return provider, model


@dataclass(frozen=True)
class Cell:
    track: str
    system: str
    F: int
    writer: tuple[str, str]
    generator: tuple[str, str] | None  # SG only

    @property
    def subdir(self) -> str:
        """The dir phase1_generate writes this cell to (--track/--capacity plus
        the configuration tag)."""
        config = {
            "synth_model": self.writer[1],
            "generator_model": self.generator[1] if self.generator else None,
        }
        return f"{self.track}/{self.system}_cap{self.F}_{config_tag(self.system, config)}"

    @property
    def name(self) -> str:
        return self.subdir.replace("/", "__")


def grid_cells(args) -> list[Cell]:
    cells = []
    for system in args.systems:
        for F in args.F:
            if "detection" in args.tracks:
                for w in args.writers:
                    gens = args.detection_generators if system == "story" else [None]
                    for g in gens:
                        cells.append(Cell("detection", system, F, _split(w), g and _split(g)))
            if "recovery" in args.tracks:
                g = ("local", LOCAL_MODEL) if system == "story" else None
                cells.append(Cell("recovery", system, F, _split(args.recovery_writer), g))
    return cells


# ---------------------------------------------------------------------------
# Commands per stage
# ---------------------------------------------------------------------------

PY = [sys.executable, "-m"]


def generate_cmd(cell: Cell, args) -> list[str]:
    cmd = PY + [
        "experiments.phase1_generation.phase1_generate",
        "--system", cell.system,
        "--capacity", str(cell.F),
        "--track", cell.track,
        "--data-dir", str(args.data_dir),
        "--limit", str(args.n_inputs),
        "--synth-provider", cell.writer[0],
        "--synth-model", cell.writer[1],
    ]  # fmt: skip
    if cell.system == "litreview":
        cmd += ["--litreview-indices", args.litreview_indices]
    if cell.generator is not None:
        # Phase 1 runs a hosted G with reasoning off, as the pinned server does.
        provider, model = cell.generator
        cmd += ["--generator-provider", provider, "--generator-model", model]
    return cmd


def select_cmd(system: str, cells: list[Cell], args) -> list[str]:
    return PY + [
        "experiments.phase1_generation.select_inputs",
        "--system", system,
        "--data-dir", str(args.data_dir),
        "--n", str(args.n_keep),
        "--dirs", *[c.subdir for c in cells],
    ]  # fmt: skip


def normal_cmd(system: str, cells: list[Cell], args) -> list[str]:
    return PY + [
        "experiments.phase1_generation.phase1_normal",
        "--system", system,
        "--data-dir", str(args.data_dir),
        "--workers", str(args.workers),
        "--dirs", *[c.subdir for c in cells],
    ]  # fmt: skip


def attack_cmd(cell: Cell, args) -> list[str]:
    return PY + [
        "experiments.phase3_attacks",
        "--system", cell.system,
        "--subdir", cell.subdir,
        "--data-dir", str(args.data_dir),
        "--n-stegos", str(args.n_keep),
        "--max-workers", str(args.workers),
    ]  # fmt: skip


def decode_cmds(cell: Cell, args) -> list[list[str]]:
    common = ["--system", cell.system, "--subdir", cell.subdir, "--data-dir", str(args.data_dir)]
    return [
        PY + ["experiments.phase4_decode.phase4a_decode", *common],
        PY + ["experiments.phase4_decode.phase4b_attack_metrics", *common],
    ]


def analyze_cmds(args) -> list[list[str]]:
    d = ["--data-dir", str(args.data_dir)]
    systems = ["--systems", ",".join(args.systems)]
    cmds = []
    if "detection" in args.tracks:
        cmds += [
            PY + ["experiments.phase2_metrics.phase2a_token_counts", *d, *systems],
            PY + ["experiments.phase2_metrics.phase2b_perplexity", *d, *systems],
            PY + ["experiments.phase2_metrics.phase2c_transformer", *d, *systems],
            PY + ["experiments.phase2_metrics.phase2c_embeddings", *d, *systems],
        ]
        if args.llm_judge:
            cmds.append(PY + ["experiments.phase2_metrics.phase2c_llm_judge", *d, *systems])
        cmds += [
            PY + ["experiments.phase2_metrics.phase2c_summary", *d, *systems],
            PY + ["experiments.phase2_metrics.phase2f_quality_judge", *d, *systems],
        ]
    if "recovery" in args.tracks:
        cmds += [
            PY + [
                "experiments.phase4_decode.recovery_csv", *d,
                "--track", "recovery", "--runs", "all",
                "--out", str(args.data_dir / "phase4_decode" / "recovery" / "recovery_results.csv"),
            ],
            PY + [
                "experiments.phase2_metrics.phase2e_capacity_normalized", *d,
                "--track", "recovery",
                "--tex-out", str(args.data_dir / "phase2_metrics" / "recovery" / "capacity.tex"),
            ],
        ]  # fmt: skip
    return cmds


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_jobs(stage: str, jobs: list[tuple[str, list[list[str]]]], args) -> list[str]:
    """Run (name, [commands]) jobs, up to --parallel at once; a job's commands
    run in order and stop at the first failure. Returns the failed names."""
    log_dir = args.data_dir / "logs" / "payload_grid" / stage
    if args.dry_run:
        for name, cmds in jobs:
            for cmd in cmds:
                print(f"[{stage}] {name}: {' '.join(cmd[1:])}")
        return []
    log_dir.mkdir(parents=True, exist_ok=True)

    def _run(name: str, cmds: list[list[str]]) -> tuple[int, float]:
        t0 = time.time()
        with open(log_dir / f"{name}.log", "a", encoding="utf-8") as f:
            for cmd in cmds:
                f.write(f"\n$ {' '.join(cmd)}\n")
                f.flush()
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
                if rc:
                    return rc, time.time() - t0
        return 0, time.time() - t0

    failed = []
    log.info("[%s] %d jobs, %d at a time; logs in %s", stage, len(jobs), args.parallel, log_dir)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_run, name, cmds): name for name, cmds in jobs}
        for fut in as_completed(futures):
            name = futures[fut]
            rc, secs = fut.result()
            if rc:
                failed.append(name)
                log.warning("[%s] FAILED %s (exit %d, %.0fs); see %s.log", stage, name, rc, secs, name)
            else:
                log.info("[%s] done %s (%.0fs)", stage, name, secs)
    return failed


def check_local_generator(cells: list[Cell], stage: str) -> None:
    """Recovery SG encodes and decodes with the pinned G: LOCAL_MODEL must name
    it rather than fall back to the factory default."""
    if any(c.generator and c.generator[0] == "local" for c in cells) and not os.environ.get(
        "LOCAL_MODEL"
    ):
        raise SystemExit(
            f"{stage}: recovery SG cells use the pinned G, but LOCAL_MODEL is not set "
            f"(the default would be {LOCAL_MODEL!r}). Set LOCAL_MODEL (and "
            "LOCAL_BASE_URL) to the pinned Qwen3.5-9B server, or pass --track detection "
            "or --systems litreview."
        )


def run_stage(stage: str, args) -> list[str]:
    cells = grid_cells(args)
    rec = [c for c in cells if c.track == "recovery"]
    det = [c for c in cells if c.track == "detection"]
    if stage == "generate":
        if not args.dry_run:
            check_local_generator(cells, stage)
        return run_jobs(stage, [(c.name, [generate_cmd(c, args)]) for c in cells], args)
    if stage == "select":
        # Every cell of a task, both tracks, whatever --track says: the inputs
        # must be the same across all of them.
        full = argparse.Namespace(**{**vars(args), "tracks": list(TRACKS)})
        jobs = []
        for system in args.systems:
            sys_cells = [c for c in grid_cells(full) if c.system == system]
            jobs.append((system, [select_cmd(system, sys_cells, args)]))
        return run_jobs(stage, jobs, args)
    if stage == "normal":
        jobs = [
            (s, [normal_cmd(s, [c for c in det if c.system == s], args)])
            for s in args.systems
            if any(c.system == s for c in det)
        ]
        return run_jobs(stage, jobs, args)
    if stage == "attack":
        return run_jobs(stage, [(c.name, [attack_cmd(c, args)]) for c in rec], args)
    if stage == "decode":
        if not args.dry_run:
            check_local_generator(rec, stage)
        return run_jobs(stage, [(c.name, decode_cmds(c, args)) for c in rec], args)
    if stage == "analyze":
        # One after another: the steps read each other's outputs.
        return run_jobs(stage, [("analyze", analyze_cmds(args))], argparse.Namespace(**{**vars(args), "parallel": 1}))
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=[*STAGES, "all", "cells"],
                        help="A stage, 'all' (every stage in order), or 'cells' (list the grid).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--systems", type=lambda s: s.split(","), default=list(SYSTEMS))
    parser.add_argument("--track", dest="tracks", choices=[*TRACKS, "all"], default="all")
    parser.add_argument("--F", type=lambda s: [int(x) for x in s.split(",")], default=list(GRID_F),
                        help="Frame sizes (default 8,16,24,32).")
    parser.add_argument("--writers", type=lambda s: s.split(","), default=list(WRITERS),
                        help="Detection writers, provider:model, comma separated.")
    parser.add_argument("--detection-generators", type=lambda s: s.split(","),
                        default=list(DETECTION_GENERATORS),
                        help="Detection G models (SG), provider:model, comma separated.")
    parser.add_argument("--recovery-writer", default=RECOVERY_WRITER)
    parser.add_argument("--n-inputs", type=int, default=N_INPUTS)
    parser.add_argument("--n-keep", type=int, default=N_KEEP)
    parser.add_argument("--litreview-indices", default=LITREVIEW_INDICES)
    parser.add_argument("--parallel", type=int, default=6, help="Cells run at once (default 6).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads inside a job (normal generation, attacks).")
    parser.add_argument("--llm-judge", action="store_true",
                        help="analyze: also run the LLM-judge detector (OPENROUTER_API_KEY).")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    args = parser.parse_args()  # fmt: skip
    args.tracks = list(TRACKS) if args.tracks == "all" else [args.tracks]

    if args.stage == "cells":
        for c in grid_cells(args):
            print(c.subdir)
        return

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        failed = run_stage(stage, args)
        if failed:
            log.error("[%s] %d failed: %s. Fix and rerun the stage (it resumes).",
                      stage, len(failed), ", ".join(failed))
            sys.exit(1)
    log.info("done: %s", ", ".join(stages))


if __name__ == "__main__":
    main()
