import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import librosa
import numpy as np
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from phcodeswitch_leaderboard.metrics import compute_detailed_metrics
from phcodeswitch_leaderboard.dataset import filter_records

WHISPER_SR = 16_000


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
    def __init__(self, model_key: str, model_info: Dict, device: Optional[str] = None):
        self.model_key = model_key
        self.model_info = model_info
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        model_id = self.model_info["model_id"]
        print(f"  Loading model '{self.model_key}' from: {model_id}")
        t0 = time.time()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Model loaded in {time.time() - t0:.1f}s (device: {self.device})")

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        audio, _ = librosa.load(audio_path, sr=WHISPER_SR, mono=True)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.model_info.get("language", "tl"),
            task=self.model_info.get("task", "transcribe"),
        )
        inputs = self.processor.feature_extractor(
            audio, sampling_rate=WHISPER_SR, return_tensors="pt",
        ).input_features.to(self.device)
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
                hyp = self.transcribe(rec["path"])
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
