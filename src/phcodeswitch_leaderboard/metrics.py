import re
from typing import List, Dict

import jiwer


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    return 100.0 * jiwer.wer(ref_norm, hyp_norm)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    return 100.0 * jiwer.cer(ref_norm, hyp_norm)


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    return {
        "wer": round(compute_wer(reference, hypothesis), 2),
        "cer": round(compute_cer(reference, hypothesis), 2),
    }


def compute_corpus_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    refs_norm = [normalize_text(r) for r in references]
    hyps_norm = [normalize_text(h) for h in hypotheses]
    return {
        "wer": round(100.0 * jiwer.wer(refs_norm, hyps_norm), 2),
        "cer": round(100.0 * jiwer.cer(refs_norm, hyps_norm), 2),
    }


def compute_detailed_metrics(references: List[str], hypotheses: List[str]) -> Dict:
    results = []
    for ref, hyp in zip(references, hypotheses):
        results.append({
            "reference": ref,
            "hypothesis": hyp,
            "reference_normalized": normalize_text(ref),
            "hypothesis_normalized": normalize_text(hyp),
            "wer": compute_wer(ref, hyp),
            "cer": compute_cer(ref, hyp),
        })
    corpus = compute_corpus_metrics(references, hypotheses)
    mean_wer = sum(r["wer"] for r in results) / len(results) if results else 0
    mean_cer = sum(r["cer"] for r in results) / len(results) if results else 0
    return {
        "corpus": corpus,
        "mean_case_wer": round(mean_wer, 2),
        "mean_case_cer": round(mean_cer, 2),
        "details": results,
        "num_samples": len(results),
    }
