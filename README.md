# Waste Classification CNN

AI-based waste classification: CNN (from scratch, then improved, then compared against ResNet transfer learning), trained on Kaggle GPU, served via FastAPI + Next.js.

See `docs/specs/02-design.md` for the design doc and `data/README.md` for dataset setup.

## Structure
- `src/` — data pipeline, models, training/eval/comparison scripts
- `notebooks/` — Kaggle training notebook
- `models/` — trained checkpoints (gitignored)
- `app/backend/`, `app/frontend/` — Phase 2/3, not yet built

## Status
Scaffold only. See TODOs in `src/`.
