#!/usr/bin/env python3
"""
PhCodeSwitch ASR Benchmarking Leaderboard CLI.

Usage:
  phcodeswitch-leaderboard --list
  phcodeswitch-leaderboard --models whisper_small --benchmark core_speakers
  phcodeswitch-leaderboard --summary
  phcodeswitch-leaderboard --report
"""

import argparse
import json
import sys
from pathlib import Path

from phcodeswitch_leaderboard.dataset import load_manifest, filter_records, get_speaker_subsets, compute_dataset_stats
from phcodeswitch_leaderboard.evaluate import ModelRegistry, ASREvaluator
from phcodeswitch_leaderboard.leaderboard import Leaderboard
from phcodeswitch_leaderboard.visualize import generate_html


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_package_config(name: str) -> str:
    return str(Path(__file__).parent / "config" / name)


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


def get_benchmark_records(benchmark_name: str, config: dict, require_audio: bool = True) -> tuple:
    benchmarks = config["benchmarks"]
    bm = benchmarks[benchmark_name]
    records = load_manifest(
        config["transcript_file"],
        config["metadata_file"],
        config["dataset_root"],
        require_audio=require_audio,
    )
    if benchmark_name == "all":
        return bm["description"], records
    if benchmark_name == "by_speaker":
        subsets = get_speaker_subsets(records, min_samples=10)
        return "Per-speaker subsets", subsets
    return bm["description"], filter_records(
        records,
        speakers=bm.get("speakers"),
        language_filter=bm.get("language_filter"),
        max_samples=bm.get("max_samples"),
    )


def run_evaluation(model_key: str, registry: ModelRegistry,
                   benchmark_name: str, benchmark_records: list,
                   leaderboard: Leaderboard, benchmark_desc: str):
    model_info = registry.get_model_info(model_key)
    print_header(f"Evaluating {model_info['name']} on [{benchmark_name}]")
    evaluator = ASREvaluator(model_key, model_info)
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
    parser.add_argument("--results-dir", type=str, default="leaderboard_results", help="Results directory")
    parser.add_argument("--config-dir", type=str, default=".", help="Directory containing benchmark_config.json and models_config.json")
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
        print("Then clone the dataset: git submodule add https://github.com/eemberda/phcodeswitch-dataset.git dataset")
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
    leaderboard = Leaderboard(results_dir=args.results_dir)

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
        print(leaderboard.summary_text())
        return

    if args.all:
        model_keys = list(registry.models.keys())
        benchmark_names = [k for k in config["benchmarks"].keys() if k != "by_speaker"]
    elif args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
        benchmark_names = [args.benchmark]
    else:
        print("No models specified. Use --all, --models, --list, --summary, or --report.")
        parser.print_help()
        return

    for bm_name in benchmark_names:
        desc, records = get_benchmark_records(bm_name, config)
        if bm_name == "by_speaker":
            for spk, spk_records in records.items():
                print_header(f"Benchmark: speaker/{spk} ({len(spk_records)} samples)")
                for mk in model_keys:
                    try:
                        run_evaluation(mk, registry, f"speaker/{spk}", spk_records, leaderboard, desc)
                    except Exception as e:
                        print(f"  SKIP {mk}: {e}")
        else:
            print_header(f"Benchmark: {bm_name} ({len(records)} samples)")
            for mk in model_keys:
                try:
                    run_evaluation(mk, registry, bm_name, records, leaderboard, desc)
                except Exception as e:
                    print(f"  SKIP {mk}: {e}")

    results_file = Path(args.results_dir) / "results.json"
    output = Path(args.results_dir) / "index.html"
    generate_html(str(results_file), str(output))
    print(f"\nOpen {output} in your browser to view the interactive leaderboard.")
    print(leaderboard.summary_text())


if __name__ == "__main__":
    main()
