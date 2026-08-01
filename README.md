# PhCodeSwitch ASR Leaderboard

A standardized benchmarking platform for evaluating ASR models on Cebuano-English code-switched speech.

**Live leaderboard:** https://eemberda.github.io/phcodeswitch-leaderboard/

## Installation

```bash
pip install phcodeswitch-leaderboard
```

Or from source:

```bash
git clone --recurse-submodules https://github.com/eemberda/phcodeswitch-leaderboard.git
cd phcodeswitch-leaderboard
pip install -e .
```

## Quick Start

```bash
# Create default config files
phcodeswitch-leaderboard --init

# Clone the dataset (contains pre-computed log-mel features, no raw audio needed)
git clone https://github.com/eemberda/phcodeswitch-dataset.git ../phcodeswitch-dataset

# List available models and benchmarks
phcodeswitch-leaderboard --list

# Evaluate a model on the core speakers benchmark
phcodeswitch-leaderboard --models whisper_small --benchmark core_speakers

# Try a custom model WITHOUT publishing it to the leaderboard
phcodeswitch-leaderboard --test --model-id your-username/your-whisper --benchmark core_speakers

# Limit a run to the first N samples (smoke test)
phcodeswitch-leaderboard --test --model-id your-username/your-whisper --max-samples 100

# View the leaderboard in terminal
phcodeswitch-leaderboard --summary

# Generate interactive HTML leaderboard report
phcodeswitch-leaderboard --report
```

Open `leaderboard_results/index.html` (published) or `test_results/index.html` (trials) in a browser.

## Trial Mode (not published)

Use `--test` to evaluate your own model against the dataset before publishing it:

```bash
phcodeswitch-leaderboard --test \
  --model-id your-username/your-whisper \
  --model-name "My Fine-Tuned Whisper" \
  --model-language tl \
  --model-task transcribe \
  --benchmark core_speakers
```

- Results are written to `test_results/` and are **never** added to the published leaderboard.
- `--model-id` accepts any Hugging Face repo id or local model path.
- `--max-samples N` runs on the first N samples for a quick check.
- To publish afterwards, add the model to `models_config.json` and re-run without `--test`.

## Dataset

The dataset is maintained as a [separate repository](https://github.com/eemberda/phcodeswitch-dataset). It ships anonymized **log-mel audio features** (`data/audio_features/{stem}.npz`) instead of raw audio — no wav download required. Stored features are 80-band log-mel dB (`torchaudio` `MelSpectrogram`, 16 kHz, 80 mels, hop 160), peak-normalized to 0 dB and quantized to `int16` with `scale=100`.

Because Whisper uses its own mel filterbank, features-mode evaluation reconstructs the power spectrogram (non-negative least squares) and re-applies Whisper's exact filterbank and normalization. The converted input is identical for every Whisper-family model, so it is cached once in `data/feature_cache/` (gitignored) and reused across runs and models.

## Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `all` | Full dataset evaluation |
| `core_speakers` | Speakers with 50+ recordings (S001-S010) |
| `by_speaker` | Per-speaker evaluation subsets |
| `code_switched` | Utterances mixing Cebuano and English |
| `cebuano_only` | Primarily Cebuano utterances |

## Models

**Baselines:** `whisper_tiny`, `whisper_base`, `whisper_small`, `whisper_medium`, `whisper_large_v3` (and English-only variants)

**Fine-tuned:** Add your own model to `models_config.json` with its Hugging Face ID or local path.

## How It Works

1. Loads transcripts, metadata, and pre-computed log-mel features from the dataset
2. Converts each utterance's features into Whisper's input format (cached per utterance)
3. Runs each model through every utterance in the benchmark
4. Computes Word Error Rate (WER) and Character Error Rate (CER) using `jiwer`
5. Stores results in `leaderboard_results/results.json` (trials go to `test_results/`)
6. Generates an interactive HTML leaderboard with sorting and filtering
