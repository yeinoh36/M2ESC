# M2ESC Ours

Anonymous code release for a multi-stage emotional support conversation pipeline.

## Overview

This repository contains two entry scripts:

- `main.py`: full pipeline run
- `ablation_study.py`: ablation run for dialog context settings

Core modules are under `src/`:

- `src/memory.py`: long-context summary update
- `src/router.py`: stage/strategy routing
- `src/specialized.py`: exploration, comforting, and action experts
- `src/respond.py`: final response synthesis
- `src/tag.py`: emotion tagging utilities
- `src/rag.py`: hybrid retrieval for external knowledge

## Requirements

- Python 3.10+
- CUDA-enabled GPU environment (recommended)
- Model/runtime dependencies used by this codebase:
  - `torch`
  - `vllm`
  - `transformers`
  - `sentence-transformers`
  - `rank-bm25`
  - `tqdm`

Install dependencies with your preferred environment manager.

## Data and Paths

The code is path-anonymized and can be configured with environment variables.

- `M2ESC_MODEL_PATH`: local model path (default: `/data1/llm-models/Qwen3-14B`)
- `M2ESC_PROMPT_DIR`: prompt directory (default: `src/prompts`)
- `M2ESC_ACTION_DB_PATH`: knowledge DB JSON path (default: `data/knowledge_db.json`)
- `M2ESC_VAD_CKPT_PATH`: VAD checkpoint path (default: `src/emotion/emobank-vad-regression-2520-20.ckpt`)
- `M2ESC_NRC_VAD_DIR`: NRC-VAD directory (default: `data/NRC-VAD`)

Expected project-local files:

- `data/ESConv_preprocessed.json`
- `data/knowledge_db.json` (optional, for action retrieval)

Outputs are written to:

- `results/`

## Usage

### Full pipeline

```bash
python main.py --gpu-ids 0 --strategy-mode 2
```

Optional debug run:

```bash
python main.py --gpu-ids 0 --strategy-mode 2 --debug
```

### Ablation run

```bash
python ablation_study.py --gpu-ids 0 --strategy-mode 2 --dialog-mode nopre
```

`dialog-mode` options:

- `nopre`
- `yespre`

## Notes

- This repository is anonymized and does not include personal contact information.
- Keep secrets (API keys, passwords, tokens) outside the repository.
