# Design: Waste Classification CNN

## Problem
Classify a waste item image into one of 6 categories (cardboard, glass, metal, paper, plastic, trash).

## Dataset
Kaggle ["Garbage Classification V2"](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (`sumn2u/garbage-classification-v2`). Class names/count are not hardcoded — `src/data/dataset.py` derives them from the dataset's subfolder names via `ImageFolder`. See `data/README.md`.

## Approach
1. Baseline CNN from scratch (`src/models/baseline_cnn.py`) — plain conv/pool/relu/fc, no regularization.
2. Improved CNN (`src/models/improved_cnn.py`) — + BatchNorm, Dropout, data augmentation.
3. Transfer learning (`src/models/resnet_transfer.py`) — frozen ResNet18 backbone + new head.
4. Compare all three (`src/compare.py`).

Training runs on Kaggle GPU (`notebooks/kaggle_train.ipynb`); code authored/version-controlled in `src/`.

## Serving (later phases)
- Phase 2: FastAPI backend (`app/backend/`) loads best checkpoint, exposes `/predict`.
- Phase 3: Next.js frontend (`app/frontend/`) for upload + result display.

## Status
Implemented: `src/config.py`, `src/data/transforms.py`, `src/data/dataset.py`, all three models, `src/train.py`, `src/evaluate.py`, `src/compare.py`, `notebooks/kaggle_train.ipynb`. Not yet run end-to-end on Kaggle.
