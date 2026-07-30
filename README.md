# PhCodeSwitch ASR Leaderboard

A standardized benchmarking platform for evaluating ASR models on Cebuano-English code-switched speech, built on the [PhCodeSwitch Cebu-Davao dataset](https://github.com/eemberda/phcodeswitch-dataset).

Inspired by the [Open ASR Leaderboard](https://huggingface.co/spaces/open-asr-leaderboard/leaderboard) and [SWE-bench](https://www.swebench.com/).

## Setup

```bash
# Clone with dataset submodule
git clone --recurse-submodules https://github.com/eemberda/phcodeswitch-leaderboard.git
cd phcodeswitch-leaderboard

# Prepare audio files (required for evaluation)
# Point to the original audio_data directory
python dataset/scripts/prepare_dataset.py /path/to/original/audio_data
```

## Quick Start

```bash
# List available models and benchmarks
python run_leaderboard.py --list

# Evaluate a specific model on the core speakers benchmark
python run_leaderboard.py --models whisper_small --benchmark core_speakers

# Evaluate the fine-tuned model
python run_leaderboard.py --models phcodeswitch_small_ceb --benchmark core_speakers

# View the leaderboard in terminal
python run_leaderboard.py --summary

# Generate interactive HTML leaderboard report
python run_leaderboard.py --report
```

Open `leaderboard_results/index.html` in a browser to view the interactive leaderboard.

## Dataset

The dataset is maintained as a [separate repository](https://github.com/eemberda/phcodeswitch-dataset) with anonymized speaker IDs (`S001`–`S123`). Audio files (~2.3 GB) are not stored in git; run the `prepare_dataset.py` script to copy and rename them from the original project.

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

**Fine-tuned:** `phcodeswitch_small_ceb` - Whisper Small fine-tuned on 4,692 Cebuano-English utterances (20.86% WER)

To add your own model, edit `models_config.json`.

## How It Works

1. Loads audio and transcripts from the dataset submodule
2. Runs each model through every utterance in the benchmark
3. Computes Word Error Rate (WER) and Character Error Rate (CER) using `jiwer`
4. Stores results in `leaderboard_results/results.json`
5. Generates an interactive HTML leaderboard with sorting and filtering
