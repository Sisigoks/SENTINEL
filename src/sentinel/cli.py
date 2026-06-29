"""SENTINEL command-line entrypoint.

Usage (on the A100):
    sentinel run --config conf/config.yaml model=qwen3_14b
    sentinel run-all-models            # sweep the three families + 32B scale check

Loads config via OmegaConf, builds the vLLM backend + frozen encoder, assembles the
corpus, and runs the full experiment driver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from .core.logging import configure_logging, get_logger
from .core.types import ThreatClass
from .corpora.loaders import load_builtin_seed_corpus, load_jsonl_corpus

log = get_logger("sentinel.cli")


def _load_cfg(path: str, overrides: list[str]):
    cfg = OmegaConf.load(path)
    # resolve model default file
    model_name = None
    for o in overrides:
        if o.startswith("model="):
            model_name = o.split("=", 1)[1]
    if model_name:
        model_cfg = OmegaConf.load(Path(path).parent / "model" / f"{model_name}.yaml")
        cfg.model = model_cfg
    elif "model" not in cfg or OmegaConf.is_missing(cfg, "model"):
        cfg.model = OmegaConf.load(Path(path).parent / "model" / "qwen3_14b.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o for o in overrides if not o.startswith("model=")]))
    return cfg


def _build_corpus(cfg):
    c = cfg.corpus
    if c.kind == "jsonl":
        return load_jsonl_corpus(c.jsonl_path)
    held = {ThreatClass(x) for x in c.get("held_out_classes", [])}
    return load_builtin_seed_corpus(repeat=c.get("repeat", 8), held_out_classes=held)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config, args.overrides)
    configure_logging(cfg.get("log_level", "INFO"))
    container = OmegaConf.to_container(cfg, resolve=True)

    from .experiment import run_all
    from .models.backend import build_backend
    from .models.encoder import FrozenEncoder

    backend = build_backend(container["model"])
    encoder = FrozenEncoder(**container["encoder"])
    corpus = _build_corpus(cfg)
    out = Path(container["output_dir"]) / backend.model_name.replace("/", "_")
    run_all(backend, encoder, corpus, container, str(out))
    return 0


def cmd_run_all_models(args: argparse.Namespace) -> int:
    for model in ["qwen3_14b", "deepseek_r1_distill_14b", "mistral_small", "qwen3_32b"]:
        log.info("running model family", model=model)
        ns = argparse.Namespace(config=args.config, overrides=[f"model={model}"])
        cmd_run(ns)
    return 0


def cmd_run_parallel(args: argparse.Namespace) -> int:
    configure_logging("INFO")
    from .launcher import DEFAULT_MODELS, run_parallel
    models = args.models.split(",") if args.models else DEFAULT_MODELS
    run_parallel(
        models=models,
        config=args.config,
        evo_config=None if args.no_evo else args.evo_config,
        gpus=args.gpus,
        runs_dir=args.runs_dir,
        aggregate=not args.no_aggregate,
    )
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    configure_logging("INFO")
    from .aggregate import aggregate
    agg = aggregate(args.runs_dir, args.out)
    print(f"[ok] aggregated {len(agg.get('models', []))} models -> {args.out}/aggregate_results.json")
    if "two_way_anova" in agg:
        for term, st_ in agg["two_way_anova"].items():
            print(f"  ANOVA {term}: F={st_['F']:.2f} p={st_['p']:.4g} eta^2={st_['eta_sq']:.3f}")
    elif "note" in agg:
        print("  " + agg["note"])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentinel", description="SENTINEL defensive-security experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the full experiment for one model")
    pr.add_argument("--config", default="conf/config.yaml")
    pr.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. model=mistral_small seed=1")
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("run-all-models", help="sweep all model families + scale check")
    pa.add_argument("--config", default="conf/config.yaml")
    pa.set_defaults(func=cmd_run_all_models)

    pp = sub.add_parser("run-parallel",
                        help="auto-detect all GPUs and fan model runs across them, then aggregate")
    pp.add_argument("--config", default="conf/config_b200.yaml", help="flagship config")
    pp.add_argument("--evo-config", default="conf/config_b200_evo.yaml",
                    help="evolution+ablation config run on spare GPUs")
    pp.add_argument("--no-evo", action="store_true", help="skip the evolution/ablation jobs")
    pp.add_argument("--models", default="", help="comma-separated model configs (default: 4 families)")
    pp.add_argument("--gpus", type=int, default=None, help="override auto-detected GPU count")
    pp.add_argument("--runs-dir", default="experiments/runs")
    pp.add_argument("--no-aggregate", action="store_true")
    pp.set_defaults(func=cmd_run_parallel)

    pg = sub.add_parser("aggregate", help="cross-model two-way ANOVA + figures from completed runs")
    pg.add_argument("--runs-dir", default="experiments/runs")
    pg.add_argument("--out", default="experiments/aggregate")
    pg.set_defaults(func=cmd_aggregate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
