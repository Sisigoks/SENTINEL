"""Multi-GPU launcher — auto-detects every GPU and fans ONE config across all of them.

vLLM owns one GPU per process, so the parallelism model is *one process per GPU*. This
detects ``torch.cuda.device_count()`` and runs that many concurrent workers; each pins
``CUDA_VISIBLE_DEVICES`` to a physical GPU and pulls jobs from a shared queue.

To use ALL GPUs with a single config and the four model families, it **shards seeds**:

    replicas = clamp(n_gpus // n_models, 1, n_seeds)

Each (model, seed-shard) pair is a job. The first shard of each model also runs the
evolution + ablation phases (the rest are grid-only, to avoid duplicating that work). Shards
write to ``<runs_dir>/_shard<k>/`` and are merged by model at aggregation time (results.json
carries the model name), so 4 models x 2 shards fills 8 GPUs and still yields per-model
multi-seed statistics + the cross-model two-way ANOVA.

Examples:
    8 GPUs, 4 models, seeds [0,1,2] -> replicas=2 -> 8 jobs, all GPUs busy.
    4 GPUs, 4 models                -> replicas=1 -> 4 jobs, one model per GPU.
    2 GPUs, 4 models                -> replicas=1 -> 4 jobs, 2 at a time (queued).
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODELS = ["qwen3_14b", "deepseek_r1_distill_14b", "mistral_small", "qwen3_32b"]


def detect_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception as exc:
        log.warning("could not query CUDA device count; assuming 0", error=str(exc))
        return 0


def _read_seeds(config: str) -> list[int]:
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config)
        seeds = OmegaConf.select(cfg, "experiment.seeds")
        return [int(s) for s in seeds] if seeds else [0, 1, 2]
    except Exception:
        return [0, 1, 2]


def _shard(seeds: list[int], n: int) -> list[list[int]]:
    n = max(1, min(n, len(seeds)))
    return [seeds[i::n] for i in range(n)]  # round-robin -> balanced groups


@dataclass
class Job:
    model: str
    config: str
    overrides: list[str] = field(default_factory=list)
    label: str = ""


@dataclass
class JobResult:
    job: Job
    gpu: int
    returncode: int
    seconds: float
    log_path: str


def build_jobs(models: list[str], config: str, n_gpus: int, runs_dir: str) -> list[Job]:
    seeds = _read_seeds(config)
    n_models = max(1, len(models))
    replicas = max(1, min(n_gpus // n_models if n_gpus else 1, len(seeds)))
    shards = _shard(seeds, replicas)
    jobs: list[Job] = []
    for model in models:
        for k, shard_seeds in enumerate(shards):
            seed_list = "[" + ",".join(str(s) for s in shard_seeds) + "]"
            ov = [f"model={model}", f"experiment.seeds={seed_list}",
                  f"output_dir={runs_dir}/_shard{k}"]
            if k != 0:  # evolution + ablation only on the first shard of each model
                ov += ["experiment.run_evolution=false", "experiment.run_ablation=false"]
            jobs.append(Job(model, config, ov,
                            label=f"{model}__s{'_'.join(str(s) for s in shard_seeds)}"))
    return jobs


def _run_job(job: Job, gpu: int, log_dir: Path) -> JobResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.label}.log"
    cmd = [sys.executable, "-m", "sentinel.cli", "run", "--config", job.config, *job.overrides]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n\n")
        fh.flush()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return JobResult(job, gpu, proc.returncode, time.time() - t0, str(log_path))


def run_parallel(
    models: list[str] | None = None,
    config: str = "conf/config.yaml",
    gpus: int | None = None,
    runs_dir: str = "experiments/runs",
    aggregate: bool = True,
) -> list[JobResult]:
    models = models or DEFAULT_MODELS
    n_gpus = gpus if gpus is not None else detect_gpus()
    if n_gpus <= 0:
        raise SystemExit(
            "No CUDA GPUs detected. run-parallel needs >= 1 GPU. Use `sentinel run` on a GPU host."
        )
    jobs = build_jobs(models, config, n_gpus, runs_dir)
    n_workers = min(n_gpus, len(jobs))
    print(f"[run-parallel] {n_gpus} GPU(s) detected; {len(jobs)} job(s) across {n_workers} worker(s)")
    log.info("launching parallel runs", gpus=n_gpus, workers=n_workers, jobs=len(jobs))

    q: queue.Queue[Job] = queue.Queue()
    for j in jobs:
        q.put(j)
    results: list[JobResult] = []
    lock = threading.Lock()
    log_path = Path(runs_dir) / "_logs"

    def worker(gpu: int) -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            print(f"  [GPU {gpu}] start {job.label}")
            try:
                res = _run_job(job, gpu, log_path)
            except Exception as exc:
                log.error("job crashed", job=job.label, gpu=gpu, error=str(exc))
                res = JobResult(job, gpu, -1, 0.0, "")
            status = "ok" if res.returncode == 0 else f"FAILED(rc={res.returncode})"
            print(f"  [GPU {gpu}] done  {job.label} [{status}, {res.seconds:.0f}s] -> {res.log_path}")
            with lock:
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
