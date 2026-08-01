import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import librosa
import numpy as np
import scipy.optimize
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from phcodeswitch_leaderboard.metrics import compute_detailed_metrics
from phcodeswitch_leaderboard.dataset import filter_records

WHISPER_SR = 16_000

# Whisper processes a fixed 30 s context window (10 ms frame hop).
WHISPER_CONTEXT_FRAMES = 3_000
# Stored features are log-mel dB in [-80, 0]; -80 dB is the silence floor.
DEFAULT_LOW_DB = -80.0
DEFAULT_FEATURE_SCALE = 100.0
# Whisper's normalized log-mel value for silence is -1.0.
SILENT_FRAME_VALUE = -1.0


def load_feature_config(config: Dict) -> Dict:
    """Load the dataset's feature_config.json if configured, else {} defaults."""
    path = config.get("feature_config_file")
    if path and Path(path).is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def reconstruct_log_mel(features_path: str,
                        feature_config: Optional[Dict] = None) -> np.ndarray:
    """Load stored int16 log-mel features and reconstruct dB values.

    Returns an (n_mels, n_frames) float32 array in dB. The npz stores
    ``mel = round((db - min_db) * scale)``; ``db = mel / scale + min_db``
    with ``min_db`` taken from ``feature_config.json`` (default -80 dB).
    """
    with np.load(features_path, allow_pickle=False) as data:
        stored = data["mel"].astype(np.float32)
        scale = float(data.get("scale", DEFAULT_FEATURE_SCALE))
    cfg = feature_config or {}
    value_range = (cfg.get("storage", {}).get("value_range_db")
                   or [DEFAULT_LOW_DB, 0.0])
    min_db = float(value_range[0])
    return stored / scale + min_db


def build_torch_filterbank(feature_config: Dict) -> np.ndarray:
    """Reconstruct the (n_mels, n_fft//2+1) filterbank used to make the dataset.

    The dataset was extracted with torchaudio ``MelSpectrogram`` using the
    parameters recorded in ``feature_config.json``. Returns float64.
    """
    cfg = feature_config.get("extraction", {}) or {}
    sample_rate = int(cfg.get("sample_rate", 16_000))
    n_fft = int(cfg.get("n_fft", 400))
    n_mels = int(cfg.get("n_mels", 80))
    f_min = float(cfg.get("f_min", 0.0))
    f_max = float(cfg.get("f_max", 8000.0))
    norm = cfg.get("norm")
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=f_min, f_max=f_max, n_mels=n_mels,
        sample_rate=sample_rate, norm=norm,
    ).numpy().astype(np.float64)
    return fb.T  # (n_mels, n_fft//2+1)


def whisper_filterbank(processor: WhisperProcessor) -> np.ndarray:
    """Return the model's (n_mels, n_freqs) mel filterbank as float64."""
    fb = processor.feature_extractor.mel_filters
    fb = np.asarray(fb, dtype=np.float64)
    return fb.T if fb.shape[0] > fb.shape[1] else fb


def _reconstruct_power(db: np.ndarray,
                       torch_fb: np.ndarray) -> np.ndarray:
    """Invert the torchaudio mel transform with NNLS to recover power.

    The dataset only ships log-mel features, so the underlying power
    spectrogram is underdetermined. NNLS finds a non-negative power
    estimate that exactly reproduces the stored mel. Returns an
    (n_fft//2+1, n_frames) float64 array.
    """
    target = np.power(10.0, db.astype(np.float64) / 10.0)
    n_bins = torch_fb.shape[1]
    power = np.zeros((n_bins, db.shape[1]))
    for t in range(db.shape[1]):
        col = target[:, t]
        if col.max() <= 0.0:
            continue
        power[:, t] = scipy.optimize.nnls(torch_fb, col)[0]
    return power


def _hash_fb(fb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(fb).tobytes()).hexdigest()[:16]


def log_mel_to_whisper_input(db: np.ndarray,
                             whisper_fb: np.ndarray,
                             torch_fb: np.ndarray,
                             max_frames: int = WHISPER_CONTEXT_FRAMES) -> np.ndarray:
    """Convert dataset dB log-mel features into Whisper's expected input.

    The dataset stores ``db = 10 * log10(power)`` peak-normalized to 0 dB.
    The stored mel was produced by a *torchaudio* filterbank, which differs
    from Whisper's own filterbank, so a naive ``db/40 + 1`` mapping drifts
    out of Whisper's expected input distribution and causes decoder
    repetition loops. Instead we recover the power spectrogram with NNLS,
    re-apply Whisper's exact filterbank and its log10 / max-8 normalization.

    Returns an (n_mels, n_frames) float32 array in Whisper's input range.
    """
    power = _reconstruct_power(db, torch_fb)
    mel = whisper_fb @ power
    log = np.log10(np.clip(mel, 1e-10, None))
    log = np.maximum(log, log.max() - 8.0)
    features = ((log + 4.0) / 4.0).astype(np.float32)
    if features.shape[-1] > max_frames:
        features = features[..., :max_frames]
    elif features.shape[-1] < max_frames:
        pad_width = max_frames - features.shape[-1]
        pad_value = float(features.min()) if features.size else SILENT_FRAME_VALUE
        pad = np.full(
            features.shape[:-1] + (pad_width,),
            pad_value, dtype=np.float32,
        )
        features = np.concatenate([features, pad], axis=-1)
    return features


class ModelRegistry:
    def __init__(self, models_config: Dict):
        self.models = {}
        for category in ["baselines", "fine_tuned"]:
            for key, cfg in models_config.get(category, {}).items():
                self.models[key] = cfg

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def get_model_info(self, model_key: str) -> Dict:
        if model_key not in self.models:
            raise KeyError(f"Unknown model: {model_key}. Available: {self.list_models()}")
        return self.models[model_key]


class ASREvaluator:
    def __init__(self, model_key: str, model_info: Dict, device: Optional[str] = None,
                 feature_config: Optional[Dict] = None, cache_dir: Optional[str] = None):
        self.model_key = model_key
        self.model_info = model_info
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_config = feature_config or {}
        self.cache_dir = cache_dir
        self.processor = None
        self.model = None
        self._whisper_fb = None
        self._torch_fb = None
        self._torch_fb_hash = None
        self._whisper_fb_hash = None
        self._load_model()

    def _load_model(self):
        model_id = self.model_info["model_id"]
        print(f"  Loading model '{self.model_key}' from: {model_id}")
        t0 = time.time()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self._whisper_fb = whisper_filterbank(self.processor)
        self._torch_fb = build_torch_filterbank(self.feature_config)
        self._torch_fb_hash = _hash_fb(self._torch_fb)
        self._whisper_fb_hash = _hash_fb(self._whisper_fb)
        print(f"  Model loaded in {time.time() - t0:.1f}s (device: {self.device})")

    def _cache_path(self, record: Dict) -> Optional[Path]:
        if not self.cache_dir or not record.get("features_path"):
            return None
        stem = Path(record["features_path"]).stem
        return Path(self.cache_dir) / f"{stem}.npz"

    def _cached_whisper_input(self, db: np.ndarray,
                              record: Dict) -> np.ndarray:
        """Convert db features to whisper input, disk-cached per utterance.

        All Whisper-family models share the same 80-mel input, so the
        converted array only depends on the stored features and the two
        filterbanks. The cache is invalidated when either filterbank changes.
        """
        cache_path = self._cache_path(record)
        if cache_path is not None and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as data:
                if (data["torch_fb_hash"] == self._torch_fb_hash
                        and data["whisper_fb_hash"] == self._whisper_fb_hash):
                    return data["input"].astype(np.float32)
        features = log_mel_to_whisper_input(
            db, self._whisper_fb, self._torch_fb)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                input=features.astype(np.float32),
                torch_fb_hash=self._torch_fb_hash,
                whisper_fb_hash=self._whisper_fb_hash,
            )
        return features

    def _get_input_features(self, record: Dict) -> torch.Tensor:
        """Return model-ready log-mel input features for one record.

        Records from the features dataset carry a ``features_path`` pointing
        at a pre-computed log-mel ``.npz``; those features are loaded and
        converted straight into Whisper's expected input format, bypassing
        ``librosa`` and the feature extractor. Raw-audio records are handled
        as before.
        """
        if record.get("features_path"):
            db = reconstruct_log_mel(record["features_path"], self.feature_config)
            features = self._cached_whisper_input(db, record)
            return torch.from_numpy(features).unsqueeze(0).to(self.device)
        audio, _ = librosa.load(record["path"], sr=WHISPER_SR, mono=True)
        return self.processor.feature_extractor(
            audio, sampling_rate=WHISPER_SR, return_tensors="pt",
        ).input_features.to(self.device)

    @torch.no_grad()
    def transcribe(self, record: Dict) -> str:
        inputs = self._get_input_features(record)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.model_info.get("language", "tl"),
            task=self.model_info.get("task", "transcribe"),
        )
        pred_ids = self.model.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=self.model_info.get("num_beams", 5),
        )
        text = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True,
        )[0].strip()
        return text

    def evaluate_benchmark(self, records: List[Dict],
                           batch_size: int = 1) -> Dict:
        references = []
        hypotheses = []
        num_samples = len(records)
        print(f"  Evaluating {num_samples} samples (batch_size={batch_size})...")
        t0 = time.time()

        for i, rec in enumerate(records):
            try:
                hyp = self.transcribe(rec)
            except Exception as e:
                print(f"    [ERROR] Sample {i+1}/{num_samples}: {e}")
                hyp = ""
            references.append(rec["sentence"])
            hypotheses.append(hyp)

            if (i + 1) % 50 == 0 or (i + 1) == num_samples:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"    Progress: {i+1}/{num_samples} ({rate:.1f} samples/s)")

        total_time = time.time() - t0
        metrics = compute_detailed_metrics(references, hypotheses)
        metrics["model_key"] = self.model_key
        metrics["model_name"] = self.model_info.get("name", self.model_key)
        metrics["num_samples"] = num_samples
        metrics["total_time_s"] = round(total_time, 1)
        metrics["avg_time_per_sample_s"] = round(total_time / num_samples, 2) if num_samples else 0

        return metrics
