#!/usr/bin/env python3
"""
PhCodeSwitch ASR Benchmarking Leaderboard CLI.

Usage:
  phcodeswitch-leaderboard --list
  phcodeswitch-leaderboard --models whisper_small --benchmark core_speakers
  phcodeswitch-leaderboard --test --model-id your-username/your-model --benchmark core_speakers
  phcodeswitch-leaderboard --summary
  phcodeswitch-leaderboard --report
"""

import argparse
import json
import sys
from pathlib import Path

from phcodeswitch_leaderboard.dataset import load_manifest, filter_records, get_speaker_subsets, compute_dataset_stats
from phcodeswitch_leaderboard.evaluate import ModelRegistry, ASREvaluator, load_feature_config
from phcodeswitch_leaderboard.leaderboard import Leaderboard
from phcodeswitch_leaderboard.visualize import generate_html


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def list_available(registry: ModelRegistry, config: dict):
    print_header("Available Models")
    print(f"{'Key':<30} {'Name':<40} {'Type':<15}")
    print("-" * 85)
    for key, info in registry.models.items():
        mtype = info.get("type", "unknown")
        print(f"{key:<30} {info.get('name', key):<40} {mtype:<15}")

    print_header("Available Benchmarks")
    benchmarks = config["benchmarks"]
    print(f"{'Key':<25} {'Description':<50}")
    print("-" * 75)
    for key, bm in benchmarks.items():
        print(f"{key:<25} {bm.get('description', ''):<50}")

    print_header("Dataset Statistics")
    records = load_manifest(
        config["transcript_file"],
        config["metadata_file"],
        config["dataset_root"],
        require_audio=False,
        feature_root=config.get("feature_root"),
    )
    stats = compute_dataset_stats(records)
    print(f"Total samples:        {stats['total_samples']}")
    print(f"Unique speakers:      {stats['unique_speakers']}")
    print(f"Language mix:")
    for lang, count in stats["language_mix_distribution"].items():
        if count > 0:
            print(f"  {lang:<15} {count}")
    print(f"\nTop 10 speakers:")
    for spk, count in list(stats["speaker_distribution"].items())[:10]:
        print(f"  {spk:<25} {count}")


def get_benchmark_records(benchmark_name: str, config: dict, require_audio: bool = True,
                          max_samples: int = None) -> tuple:
    benchmarks = config["benchmarks"]
    bm = benchmarks[benchmark_name]
    records = load_manifest(
        config["transcript_file"],
        config["metadata_file"],
        config["dataset_root"],
        require_audio=require_audio,
        feature_root=config.get("feature_root"),
    )
    limit = max_samples if max_samples else bm.get("max_samples")
    if benchmark_name == "all":
        records = records[:limit] if limit else records
        return bm["description"], records
    if benchmark_name == "by_speaker":
        subsets = get_speaker_subsets(records, min_samples=10)
        if limit:
            subsets = {k: v[:limit] for k, v in subsets.items()}
        return "Per-speaker subsets", subsets
    return bm["description"], filter_records(
        records,
        speakers=bm.get("speakers"),
        language_filter=bm.get("language_filter"),
        max_samples=limit,
    )


def resolve_models(registry: ModelRegistry, model_keys: list,
                   model_id: str = None, model_name: str = None,
                   language: str = "tl", task: str = "transcribe") -> list:
    """Return a list of (model_key, model_info) tuples to evaluate.

    A ``model_id`` (Hugging Face repo id or local path) is used as-is,
    otherwise registered model keys are looked up in the registry.
    """
    if model_id:
        return [(model_name or "custom_model", {
            "name": model_name or model_id,
            "model_id": model_id,
            "type": "custom",
            "language": language,
            "task": task,
            "num_beams": 5,
        })]
    if model_keys:
        return [(mk, registry.get_model_info(mk)) for mk in model_keys]
    return []


def run_evaluation(model_key: str, model_info: dict,
                   benchmark_name: str, benchmark_records: list,
                   leaderboard: Leaderboard, benchmark_desc: str,
                   feature_config: dict = None, cache_dir: str = None) -> dict:
    print_header(f"Evaluating {model_info['name']} on [{benchmark_name}]")
    evaluator = ASREvaluator(model_key, model_info, feature_config=feature_config,
                             cache_dir=cache_dir)
    metrics = evaluator.evaluate_benchmark(benchmark_records)
    leaderboard.add_result(
        model_key=model_key,
        model_name=model_info["name"],
        benchmark_name=benchmark_name,
        benchmark_desc=benchmark_desc,
        metrics=metrics,
    )
    print(f"  Results -> WER: {metrics['corpus']['wer']:.2f}% | CER: {metrics['corpus']['cer']:.2f}% | Samples: {metrics['num_samples']}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="PhCodeSwitch ASR Benchmarking Leaderboard",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all models on all benchmarks")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys")
    parser.add_argument("--benchmark", type=str, default="core_speakers", help="Benchmark name")
    parser.add_argument("--list", action="store_true", help="List available models and benchmarks")
    parser.add_argument("--summary", action="store_true", help="Show leaderboard summary in terminal")
    parser.add_argument("--report", action="store_true", help="Generate HTML report from existing results")
    parser.add_argument("--init", action="store_true", help="Create default config files in the current directory")
    parser.add_argument("--results-dir", type=str, default="leaderboard_results", help="Results directory (published leaderboard)")
    parser.add_argument("--config-dir", type=str, default=".", help="Directory containing benchmark_config.json and models_config.json")

    parser.add_argument("--test", action="store_true",
                        help="Trial run: evaluate against the dataset WITHOUT publishing to the leaderboard")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Hugging Face model id or local path to test (e.g. your-username/your-whisper)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Display name for --model-id")
    parser.add_argument("--model-language", type=str, default="tl",
                        help="Decoding language for --model-id (default: tl)")
    parser.add_argument("--model-task", type=str, default="transcribe",
                        help="Decoding task for --model-id (default: transcribe)")
    parser.add_argument("--test-dir", type=str, default="test_results",
                        help="Directory for trial (--test) results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to the first N samples (useful for smoke tests)")
    args = parser.parse_args()

    if args.init:
        from shutil import copy2
        pkg_config = Path(__file__).parent / "config"
        for fname in ["benchmark_config.json", "models_config.json"]:
            src = pkg_config / fname
            dst = Path(fname)
            if dst.exists():
                print(f"  SKIP {fname} (already exists)")
            else:
                copy2(str(src), str(dst))
                print(f"  Created {fname}")
        print("Done. Edit these files to configure the leaderboard.")
        print("The default config evaluates against the audio-features dataset:")
        print("  ../phcodeswitch-dataset/data/audio_features")
        print("(git clone https://github.com/eemberda/phcodeswitch-dataset.git ../phcodeswitch-dataset)")
        return

    config_dir = Path(args.config_dir)
    config_path = config_dir / "benchmark_config.json"
    models_path = config_dir / "models_config.json"

    if not config_path.exists() or not models_path.exists():
        print(f"ERROR: Config files not found in {config_dir}")
        print("Run: phcodeswitch-leaderboard --init")
        print("Or specify: --config-dir <path>")
        sys.exit(1)

    config = load_json(str(config_path))
    models_config = load_json(str(models_path))
    registry = ModelRegistry(models_config)
    feature_config = load_feature_config(config)

    if args.list:
        list_available(registry, config)
        return

    if args.report:
        results_file = Path(args.results_dir) / "results.json"
        if not results_file.exists():
            print(f"No results found at {results_file}. Run evaluations first.")
            sys.exit(1)
        output = Path(args.results_dir) / "index.html"
        path = generate_html(str(results_file), str(output))
        print(f"Open {path} in your browser to view the leaderboard.")
        return

    if args.summary:
        print(Leaderboard(results_dir=args.results_dir).summary_text())
        return

    if args.test:
        run_trial(args, config, registry, feature_config)
        return

    leaderboard = Leaderboard(results_dir=args.results_dir)

    if args.all:
        model_keys = list(registry.models.keys())
        benchmark_names = [k for k in config["benchmarks"].keys() if k != "by_speaker"]
    elif args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
        benchmark_names = [args.benchmark]
    else:
        print("No models specified. Use --all, --models, --list, --summary, --report, or --test.")
        parser.print_help()
        return

    print_header("Published leaderboard run")
    print(f"  Results will be written to {args.results_dir}/ (this IS the public leaderboard).")
    print(f"  Use --test to try a model without publishing it.")

    for bm_name in benchmark_names:
        desc, records = get_benchmark_records(bm_name, config, max_samples=args.max_samples)
        if bm_name == "by_speaker":
            for spk, spk_records in records.items():
                print_header(f"Benchmark: speaker/{spk} ({len(spk_records)} samples)")
                for mk in model_keys:
                    try:
                        run_evaluation(mk, registry.get_model_info(mk), f"speaker/{spk}",
                                       spk_records, leaderboard, desc, feature_config,
                                       config.get("feature_cache_dir", "data/feature_cache"))
                    except Exception as e:
                        print(f"  SKIP {mk}: {e}")
        else:
            print_header(f"Benchmark: {bm_name} ({len(records)} samples)")
            for mk in model_keys:
                try:
                    run_evaluation(mk, registry.get_model_info(mk), bm_name,
                                   records, leaderboard, desc, feature_config,
                                   config.get("feature_cache_dir", "data/feature_cache"))
                except Exception as e:
                    print(f"  SKIP {mk}: {e}")

    results_file = Path(args.results_dir) / "results.json"
    output = Path(args.results_dir) / "index.html"
    generate_html(str(results_file), str(output))
    print(f"\nOpen {output} in your browser to view the interactive leaderboard.")
    print(leaderboard.summary_text())


def run_trial(args, config, registry, feature_config):
    """Evaluate user-supplied models against the dataset without publishing."""
    print_header("TRIAL MODE (not published to the leaderboard)")
    print(f"  Results will be written to {args.test_dir}/ and are NOT published.")

    models = resolve_models(
        registry,
        [m.strip() for m in args.models.split(",")] if args.models else None,
        model_id=args.model_id,
        model_name=args.model_name,
        language=args.model_language,
        task=args.model_task,
    )
    if not models:
        print("No model specified. Provide --model-id <hf-id-or-local-path> or --models <keys>.")
        sys.exit(1)

    benchmark_names = [args.benchmark]
    for bm_name in benchmark_names:
        if bm_name not in config["benchmarks"]:
            print(f"ERROR: Unknown benchmark: {bm_name}. Available: {list(config['benchmarks'].keys())}")
            sys.exit(1)

    trial_leaderboard = Leaderboard(
        name="PhCodeSwitch ASR Trial (Not Published)",
        results_dir=args.test_dir,
    )

    for bm_name in benchmark_names:
        desc, records = get_benchmark_records(bm_name, config, max_samples=args.max_samples)
        if bm_name == "by_speaker":
            for spk, spk_records in records.items():
                print_header(f"Benchmark: speaker/{spk} ({len(spk_records)} samples)")
                for mk, info in models:
                    try:
                        run_evaluation(mk, info, f"speaker/{spk}", spk_records,
                                       trial_leaderboard, desc, feature_config,
                                       config.get("feature_cache_dir", "data/feature_cache"))
                    except Exception as e:
                        print(f"  SKIP {mk}: {e}")
        else:
            print_header(f"Benchmark: {bm_name} ({len(records)} samples)")
            for mk, info in models:
                try:
                    run_evaluation(mk, info, bm_name, records,
                                   trial_leaderboard, desc, feature_config,
                                   config.get("feature_cache_dir", "data/feature_cache"))
                except Exception as e:
                    print(f"  SKIP {mk}: {e}")

    results_file = Path(args.test_dir) / "results.json"
    if not results_file.exists():
        print("  No trial results produced.")
        sys.exit(1)

    output = Path(args.test_dir) / "index.html"
    generate_html(str(results_file), str(output))
    print(f"\nTrial report: {output}")
    print(trial_leaderboard.summary_text())

    print_header("Next steps")
    if args.model_id:
        print("  To publish this model on the leaderboard:")
        print("    1. Add it to models_config.json under 'fine_tuned', e.g.:")
        print("         \"my_model\": {")
        print("           \"name\": \"My Model\",")
        print(f"           \"model_id\": \"{args.model_id}\",")
        print("           \"type\": \"local\" | \"pretrained\",")
        print("           \"language\": \"tl\", \"task\": \"transcribe\", \"num_beams\": 5")
        print("         }")
        print("    2. Re-run WITHOUT --test to publish:")
        print("         phcodeswitch-leaderboard --models my_model --benchmark core_speakers")
    else:
        print("  These models are registered in models_config.json.")
        print("  Re-run WITHOUT --test to publish them on the leaderboard:")
        print(f"    phcodeswitch-leaderboard --models {args.models} --benchmark {args.benchmark}")


if __name__ == "__main__":
    main()
