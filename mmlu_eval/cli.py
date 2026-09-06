"""Legacy OpenAI CLI plus explicit reproducibility and debug options."""
import argparse
from pathlib import Path
from .core import EvalConfig, MMLUEvaluator, atomic_csv, summarize


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-e", default="gpt-4o-mini")
    p.add_argument("--ntrain", "-k", type=int, default=5)
    p.add_argument("--data_dir", "-d", required=True)
    p.add_argument("--output_dir", default="results_permutation_shared")
    p.add_argument("--subject")
    p.add_argument("--sample_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--manifest", help="Existing manifest is authoritative; all rows, in its exact order")
    p.add_argument("--permutation_count", type=int, default=4)
    p.add_argument("--permutation_seed", type=int, default=0, help="Recorded, unused by fixed cyclic order")
    p.add_argument("--context_policy", choices=["fixed", "reduce"], default="reduce")
    p.add_argument("--context_tokenizer", default="gpt2")
    p.add_argument("--max_context_length", type=int, default=2048, help="Shared context budget")
    p.add_argument("--api_mode", choices=["completions", "chat"], default="completions")
    p.add_argument("--answer_prefix", choices=["space", "bare"], default="space")
    p.add_argument("--api_max_context_length", type=int, default=128000)
    p.add_argument("--tokenizer", default="server-managed")
    p.add_argument("--request_interval", type=float, default=1.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--dry_run", action="store_true", help="Export all selected cases, no API/key/model required")
    return p


def main(args=None):
    args = parser().parse_args() if args is None else args
    cfg = EvalConfig(**{name: getattr(args, name) for name in EvalConfig.__dataclass_fields__})
    evaluator = MMLUEvaluator(args.data_dir, cfg)
    output_dir = Path(args.output_dir)
    safe_model = args.model.replace("/", "_").replace("\\", "_").replace(":", "_")
    tag = f"{cfg.ntrain}shot_seed{cfg.seed}_frac{str(cfg.sample_frac).replace('.', 'p')}_{cfg.subject or 'all'}"
    output = output_dir / f"results_{safe_model}_{tag}.csv"
    if args.dry_run:
        manifest = evaluator.manifest(args.manifest)
        atomic_csv(output.with_suffix(".manifest.csv"), manifest)
        evaluator.export_debug(manifest, output.with_suffix(".prompts.jsonl"))
        print(f"Prepared {len(manifest)} questions / {len(manifest)*4} prompts: {output.with_suffix('.manifest.csv')}")
        return
    from .backends.openai_backend import OpenAIBackend
    backend = OpenAIBackend(
        args.model, api_mode=args.api_mode,
        answer_prefix=" " if args.answer_prefix == "space" else "",
        workers=args.workers, request_interval=args.request_interval, retries=args.retries,
        tokenizer=args.tokenizer, max_context_length=args.api_max_context_length)
    frame = evaluator.run(backend, output, manifest_path=args.manifest)
    print(summarize(frame))
