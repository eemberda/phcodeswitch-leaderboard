# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Features-mode dataset evaluation.** The benchmark now runs against the
  pre-computed log-mel audio features in `../phcodeswitch-dataset/data/audio_features`
  instead of requiring a ~2.3 GB raw-wav download. Records loaded from the features
  dataset carry a `features_path` and are transcribed directly from the stored
  features (`dataset.py`, `evaluate.py`).
- **Trial mode (`--test`).** Users can evaluate their own model with
  `--test --model-id <hf-id-or-local-path>` and the results are written only to
  `test_results/` — they are never added to the published leaderboard
  (`cli.py`). Publishing still requires adding the model to `models_config.json`
  and re-running without `--test`.
- **`--max-samples` CLI flag.** Limits an evaluation to the first N samples of a
  benchmark, useful for smoke tests before committing to a full (multi-hour) run.
- **Per-utterance feature cache.** Converted Whisper inputs are cached under
  `data/feature_cache/` (gitignored). Because every Whisper-family model shares
  the same 80-mel input, the conversion result is model-independent and is
  computed once, then reused across runs and models. The cache is invalidated
  automatically when either filterbank changes.
- **`--model-name`, `--model-language`, `--model-task`** CLI options to control
  display name and decoding language/task for `--model-id`.
- `scipy` and `torchaudio` added to `requirements.txt` (used by the NNLS power
  reconstruction and torchaudio filterbank).

### Changed

- **Whisper input conversion rewritten (`evaluate.py`).** The previous naive
  `db / 40 + 1` mapping of the dataset's torchaudio log-mel features into
  Whisper's expected input drove the decoder into repetition loops on roughly
  20% of utterances, inflating corpus WER to 165–210% vs. a 76% raw-audio
  baseline. The dataset features are peak-normalized (absolute level discarded)
  and were produced by a torchaudio filterbank that differs from Whisper's, so
  no fixed affine mapping matches Whisper's input distribution.
  The conversion now:
  1. Reconstructs the power spectrogram from the stored dB mel via
     non-negative least squares (`scipy.optimize.nnls`) against the torchaudio
     filterbank rebuilt from `feature_config.json`;
  2. Re-applies Whisper's exact mel filterbank and its
     `log10` / `max - 8` / `(log + 4) / 4` normalization;
  3. Pads short utterances with the utterance's own silence floor (matching
     Whisper's native padding) instead of a fixed value.
  Validation on the first 100 `core_speakers` samples with `whisper-tiny`:
  corpus WER **76.55%** (features) vs. **76.24%** (raw audio), with zero
  repetition loops. Integrated CLI run: **72.65%** WER.
- **Config files** (`benchmark_config.json`,
  `src/phcodeswitch_leaderboard/config/benchmark_config.json`) now default to
  `dataset_mode: "features"` with `feature_root`, `feature_config_file`, and
  `feature_cache_dir` pointing at the features dataset.
- **`--init`** now prints the features-mode setup instructions.
- **README** updated with the features-dataset workflow, trial-mode usage, and
  the NNLS conversion rationale.
- `.gitignore` now excludes `test_results/` and `data/feature_cache/`.
