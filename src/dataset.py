import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CEBUANO_KEYWORDS = {
    "maayong", "buntag", "hapón", "gabii", "adlaw", "kumusta", "palihug",
    "salamat", "ako", "ikaw", "siya", "kita", "kami", "kamo", "sila",
    "mao", "dili", "ug", "sa", "nga", "ang", "mga", "ba", "na", "pa",
    "kanang", "kini", "kana", "didto", "dinhi", "diin", "asa", "kanus-a",
    "ngano", "unsa", "kinsa", "pila", "tagpila", "bisan", "kay", "kon",
    "apan", "o", "pero", "gawas", "lang", "daw", "man", "gayud", "jud",
    "gyud", "kaayo", "pud", "sad", "ra", "lamang", "ta", "ko", "ka",
    "mo", "mi", "nato", "ila", "akoang", "imong", "niya", "amoa", "inyo",
    "daghan", "gamay", "taas", "mubo", "dako", "maayo", "daot", "nindot",
    "gwapa", "gwapo", "tam-is", "parat", "mapait", "asin", "asukal",
    "bugas", "tubig", "isda", "manok", "baboy", "baka", "kanding",
    "balay", "eskwelahan", "tindahan", "simbaan", "ospital", "merkado",
    "dalagan", "lakaw", "lingkod", "tindog", "katulog", "kaon", "inom",
    "basa", "sulat", "paminaw", "tan-aw", "sulti", "isturya", "hangad",
    "pangita", "hatag", "kuha", "butang", "himu", "buhat", "tabang",
    "tawag", "adto", "ari", "anhi", "anha", "lakaw", "balik", "sulod",
    "gawas", "saka", "kanaog", "labay", "agi", "sunod", "una", "human",
    "bago", "tigulang", "bata", "batan-on", "tawo", "lalaki", "babaye",
    "amahan", "inahan", "igsoon", "pag-umangkon", "paryente", "higala",
    "silingan", "maestra", "maestro", "doktor", "nars", "abogado",
    "ngalan", "panahon", "oras", "karong", "kaganina", "ugma", "ganiha",
    "karon", "unyang", "buntag", "udto", "hapón", "gabii", "kagabhion",
    "adlaw", "semana", "bulan", "tuig", "inyo", "akoa", "kanako",
    "kanimo", "kaniya", "kanato", "kaninyo", "kanila", "yano", "saba",
    "hinay", "kusog", "tul-id", "liko", "uban", "uban-uban", "tug-ani",
    "walay", "naay", "mangaon", "manihapon", "manihaw", "tilaw", "huwam",
    "ulam", "pamahaw", "paniudto", "panihapon", "meryenda"
}


ENGLISH_KEYWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "shall", "should", "may", "might", "must", "i", "you", "he",
    "she", "it", "we", "they", "my", "your", "his", "her", "its", "our",
    "their", "me", "him", "us", "them", "this", "that", "these", "those",
    "and", "but", "or", "nor", "for", "yet", "so", "if", "because",
    "when", "where", "how", "what", "which", "who", "whom", "whose",
    "not", "no", "never", "always", "sometimes", "often", "rarely",
    "here", "there", "everywhere", "please", "thank", "yes", "no",
    "hello", "hi", "goodbye", "bye", "okay", "ok", "sorry", "excuse",
    "help", "stop", "go", "come", "get", "give", "take", "make", "put",
    "let", "ask", "tell", "show", "try", "leave", "keep", "start",
    "check", "save", "send", "open", "close", "file", "folder", "cloud",
    "app", "error", "process", "system", "meeting", "office", "team",
    "work", "project", "final", "report", "presentation", "meeting",
    "schedule", "appointment", "complete", "ready", "done", "quality",
    "time", "minute", "hour", "day", "week", "month", "year", "morning",
    "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
    "always", "sometimes", "never", "often", "now", "then", "later",
    "before", "after", "during", "while", "until", "since", "already",
    "just", "still", "yet", "again", "also", "very", "really", "quite",
    "too", "enough", "actually", "basically", "probably", "maybe",
    "perhaps", "sure", "certainly", "absolutely", "definitely"
}


def detect_language_mix(text: str) -> str:
    words = set(re.findall(r"[a-z']+", text.lower()))
    if not words:
        return "unknown"
    ceb_count = len(words & CEBUANO_KEYWORDS)
    eng_count = len(words & ENGLISH_KEYWORDS)
    total_known = ceb_count + eng_count
    if total_known == 0:
        return "unknown"
    ceb_ratio = ceb_count / total_known
    if ceb_ratio > 0.8:
        return "cebuano"
    elif ceb_ratio < 0.2:
        return "english"
    else:
        return "code_switch"


def extract_speaker(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts[0].isdigit():
        return "unknown"
    return parts[0]


def load_manifest(transcript_file: str, metadata_file: str, dataset_root: str) -> List[Dict]:
    records = []
    seen = set()

    if Path(transcript_file).is_file():
        with open(transcript_file, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                stem, text = line.split("|", 1)
                stem = stem.strip()
                text = text.strip()
                for ext in ["", ".wav", ".mp3"]:
                    candidate = stem + ext if ext else stem
                    full = Path(dataset_root) / candidate
                    if full.is_file():
                        key = str(full)
                        if key not in seen:
                            seen.add(key)
                            records.append({
                                "path": str(full),
                                "filename": full.name,
                                "sentence": text,
                                "speaker": extract_speaker(full.name),
                                "language_mix": detect_language_mix(text),
                            })
                        break

    if Path(metadata_file).is_file():
        with open(metadata_file, encoding="utf-8-sig") as f:
            meta = json.load(f)
        entries = meta if isinstance(meta, list) else meta.get("recordings", [])
        for entry in entries:
            fname = entry.get("filename", "")
            text = entry.get("text", "")
            full = Path(dataset_root) / fname
            if full.is_file():
                key = str(full)
                if key not in seen:
                    seen.add(key)
                    records.append({
                        "path": str(full),
                        "filename": full.name,
                        "sentence": text,
                        "speaker": extract_speaker(full.name),
                        "language_mix": detect_language_mix(text),
                    })

    return records


def filter_records(records: List[Dict], speakers: Optional[List[str]] = None,
                   language_filter: Optional[str] = None,
                   max_samples: Optional[int] = None) -> List[Dict]:
    result = list(records)
    if speakers is not None:
        speaker_set = set(s.lower() for s in speakers)
        result = [r for r in result if r["speaker"].lower() in speaker_set]
    if language_filter is not None:
        result = [r for r in result if r["language_mix"] == language_filter]
    if max_samples is not None and len(result) > max_samples:
        result = result[:max_samples]
    return result


def load_benchmark(benchmark_name: str, config: Dict) -> Tuple[str, List[Dict]]:
    benchmarks = config["benchmarks"]
    if benchmark_name not in benchmarks:
        raise KeyError(f"Unknown benchmark: {benchmark_name}. Available: {list(benchmarks.keys())}")
    bm = benchmarks[benchmark_name]
    records = load_manifest(
        config["transcript_file"],
        config["metadata_file"],
        config["dataset_root"],
    )
    if benchmark_name == "all":
        return bm["description"], records
    if benchmark_name == "by_speaker":
        return "Per-speaker subsets", records
    return bm["description"], filter_records(
        records,
        speakers=bm.get("speakers"),
        language_filter=bm.get("language_filter"),
        max_samples=bm.get("max_samples"),
    )


def get_speaker_subsets(records: List[Dict], min_samples: int = 10) -> Dict[str, List[Dict]]:
    speakers: Dict[str, List[Dict]] = {}
    for r in records:
        spk = r["speaker"]
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(r)
    return {k: v for k, v in speakers.items() if len(v) >= min_samples}


def compute_dataset_stats(records: List[Dict]) -> Dict:
    speakers = {}
    lang_mix = {"cebuano": 0, "english": 0, "code_switch": 0, "unknown": 0}
    for r in records:
        spk = r["speaker"]
        speakers[spk] = speakers.get(spk, 0) + 1
        lang_mix[r["language_mix"]] = lang_mix.get(r["language_mix"], 0) + 1
    return {
        "total_samples": len(records),
        "unique_speakers": len(speakers),
        "speaker_distribution": dict(sorted(speakers.items(), key=lambda x: -x[1])),
        "language_mix_distribution": lang_mix,
    }
