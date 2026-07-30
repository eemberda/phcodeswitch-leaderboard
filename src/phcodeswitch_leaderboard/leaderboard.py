import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


LEADERBOARD_VERSION = "1.0.0"


class Leaderboard:
    def __init__(self, name: str = "PhCodeSwitch ASR Leaderboard",
                 results_dir: str = "leaderboard_results"):
        self.name = name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = []
        self._load_existing()

    def _load_existing(self):
        results_file = self.results_dir / "results.json"
        if results_file.exists():
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            self.name = data.get("name", self.name)

    def add_result(self, model_key: str, model_name: str,
                   benchmark_name: str, benchmark_desc: str,
                   metrics: Dict):
        entry = {
            "model_key": model_key,
            "model_name": model_name,
            "benchmark": benchmark_name,
            "benchmark_description": benchmark_desc,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "corpus_wer": metrics["corpus"]["wer"],
                "corpus_cer": metrics["corpus"]["cer"],
                "mean_case_wer": metrics["mean_case_wer"],
                "mean_case_cer": metrics["mean_case_cer"],
                "num_samples": metrics["num_samples"],
                "total_time_s": metrics["total_time_s"],
                "avg_time_per_sample_s": metrics["avg_time_per_sample_s"],
            },
        }

        existing = self._find_entry(model_key, benchmark_name)
        if existing:
            self.entries.remove(existing)

        self.entries.append(entry)
        self._save()

    def _find_entry(self, model_key: str, benchmark_name: str) -> Optional[Dict]:
        for e in self.entries:
            if e["model_key"] == model_key and e["benchmark"] == benchmark_name:
                return e
        return None

    def _save(self):
        data = {
            "name": self.name,
            "version": LEADERBOARD_VERSION,
            "generated_at": datetime.now().isoformat(),
            "entries": self.entries,
        }
        with open(self.results_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_rankings(self, benchmark: str,
                     metric: str = "corpus_wer") -> List[Dict]:
        filtered = [e for e in self.entries if e["benchmark"] == benchmark]
        filtered.sort(key=lambda e: e["metrics"][metric])
        return filtered

    def get_available_benchmarks(self) -> List[str]:
        return list(set(e["benchmark"] for e in self.entries))

    def get_available_models(self) -> List[str]:
        return list(set(e["model_key"] for e in self.entries))

    def summary_text(self) -> str:
        lines = [f"=== {self.name} ==="]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Total entries: {len(self.entries)}")
        benchmarks = self.get_available_benchmarks()
        for bm in benchmarks:
            rankings = self.get_rankings(bm)
            lines.append(f"\n[{bm}]")
            lines.append(f"{'Rank':<5} {'Model':<30} {'WER %':<8} {'CER %':<8} {'Samples':<8}")
            lines.append("-" * 60)
            for rank, entry in enumerate(rankings, 1):
                m = entry["metrics"]
                lines.append(
                    f"{rank:<5} {entry['model_name']:<30} "
                    f"{m['corpus_wer']:<8} {m['corpus_cer']:<8} "
                    f"{m['num_samples']:<8}"
                )
        return "\n".join(lines)
