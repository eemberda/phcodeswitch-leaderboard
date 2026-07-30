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

# Clone the dataset
git clone https://github.com/eemberda/phcodeswitch-dataset.git dataset

# Prepare audio files (required for evaluation)
python dataset/scripts/prepare_dataset.py /path/to/original/audio_data

# List available models and benchmarks
phcodeswitch-leaderboard --list

# Evaluate a model on the core speakers benchmark
phcodeswitch-leaderboard --models whisper_small --benchmark core_speakers

# View the leaderboard in terminal
phcodeswitch-leaderboard --summary

# Generate interactive HTML leaderboard report
phcodeswitch-leaderboard --report
```

Open `leaderboard_results/index.html` in a browser to view the interactive leaderboard.

## Dataset

The dataset is maintained as a [separate repository](https://github.com/eemberda/phcodeswitch-dataset) with anonymized speaker IDs (`S001`–`S123`). Audio files (~2.3 GB) are not stored in git; run `prepare_dataset.py` to copy and rename them from the original project.

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

1. Loads audio and transcripts from the dataset
2. Runs each model through every utterance in the benchmark
3. Computes Word Error Rate (WER) and Character Error Rate (CER) using `jiwer`
4. Stores results in `leaderboard_results/results.json`
5. Generates an interactive HTML leaderboard with sorting and filtering
