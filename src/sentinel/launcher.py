"""Multi-GPU launcher — auto-detects every available GPU and fans work across all of them.

vLLM owns one GPU per process, so the right parallelism model is *one process per GPU*. This
module detects ``torch.cuda.device_count()`` and runs a worker pool with exactly that many
concurrent slots: each slot pins ``CUDA_VISIBLE_DEVICES`` to one physical GPU and pulls jobs
(``model x config``) from a shared queue until the queue drains. So:

  * 8 GPUs + 8 jobs  -> all 8 run at once (full utilization).
  * 8 GPUs + 4 jobs  -> 4 GPUs busy, 4 idle (add the evo config to fill them).
  * 2 GPUs + 8 jobs  -> 2 at a time, queued.

After all jobs finish it runs the cross-model aggregation (two-way ANOVA + figures).
No code path changes inside the per-GPU run — each subprocess is the normal single-GPU
pipeline, which is exactly what vLLM expects.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .core.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODELS = ["qwen3_14b", "deepseek_r1_distill_14b", "mistral_small", "qwen3_32b"]


def detect_gpus() -> int:
    """Number of visible CUDA GPUs (0 if torch/CUDA unavailable)."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception as exc:  # torch missing (CPU dev box) or no driver
        log.warning("could not query CUDA device count; assuming 0", error=str(exc))
        return 0


@dataclass
class Job:
    model: str
    config: str
    label: str  # for the log filename


@dataclass
class JobResult:
    job: Job
    gpu: int
    returncode: int
    seconds: float
    log_path: str


def build_jobs(models: list[str], config: str, evo_config: str | None) -> list[Job]:
    cfg_label = Path(config).stem
    jobs = [Job(m, config, f"{m}__{cfg_label}") for m in models]
    if evo_config and evo_config.lower() != "none":
        evo_label = Path(evo_config).stem
        jobs += [Job(m, evo_config, f"{m}__{evo_label}") for m in models]
    return jobs


def _run_job(job: Job, gpu: int, log_dir: Path) -> JobResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.label}.log"
    cmd = [sys.executable, "-m", "sentinel.cli", "run", "--config", job.config, f"model={job.model}"]
    env_line = f"CUDA_VISIBLE_DEVICES={gpu}"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {env_line} {' '.join(cmd)}\n\n")
        fh.flush()
        import os

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return JobResult(job, gpu, proc.returncode, time.time() - t0, str(log_path))


def run_parallel(
    models: list[str] | None = None,
    config: str = "conf/config_b200.yaml",
    evo_config: str | None = "conf/config_b200_evo.yaml",
    gpus: int | None = None,
    runs_dir: str = "experiments/runs",
    aggregate: bool = True,
    log_dir: str = "experiments/runs/_logs",
) -> list[JobResult]:
    models = models or DEFAULT_MODELS
    n_gpus = gpus if gpus is not None else detect_gpus()
    if n_gpus <= 0:
        raise SystemExit(
            "No CUDA GPUs detected. run-parallel needs at least one GPU. "
            "On a CPU box use `sentinel run` only after installing the GPU stack on a GPU host."
        )
    jobs = build_jobs(models, config, evo_config)
    n_workers = min(n_gpus, len(jobs))
    log.info("launching parallel runs", gpus=n_gpus, workers=n_workers, jobs=len(jobs),
             models=models)
    print(f"[run-parallel] detected {n_gpus} GPU(s); dispatching {len(jobs)} job(s) "
          f"across {n_workers} worker(s)")

    q: queue.Queue[Job] = queue.Queue()
    for j in jobs:
        q.put(j)
    results: list[JobResult] = []
    results_lock = threading.Lock()
    log_path = Path(log_dir)

    def worker(gpu: int) -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            print(f"  [GPU {gpu}] start {job.label}")
            try:
                res = _run_job(job, gpu, log_path)
            except Exception as exc:  # never let one job kill the pool
                log.error("job crashed", job=job.label, gpu=gpu, error=str(exc))
                res = JobResult(job, gpu, returncode=-1, seconds=0.0, log_path="")
            status = "ok" if res.returncode == 0 else f"FAILED(rc={res.returncode})"
            print(f"  [GPU {gpu}] done  {job.label} [{status}, {res.seconds:.0f}s] "
                  f"-> {res.log_path}")
            with results_lock:
                results.append(res)
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for r in results if r.returncode == 0)
    print(f"[run-parallel] {ok}/{len(results)} jobs succeeded")
    for r in results:
        if r.returncode != 0:
            print(f"  ! {r.job.label} failed (rc={r.returncode}) — see {r.log_path}")

    if aggregate:
        try:
            from .aggregate import aggregate as do_aggregate

            agg = do_aggregate(runs_dir, "experiments/aggregate")
            print(f"[run-parallel] aggregated {len(agg.get('models', []))} models "
                  f"-> experiments/aggregate/aggregate_results.json")
        except SystemExit as exc:
            print(f"[run-parallel] aggregation skipped: {exc}")
    return results
