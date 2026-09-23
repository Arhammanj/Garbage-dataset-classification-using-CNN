# Backend (Phase 2 — not yet built)

FastAPI service that loads a trained checkpoint from `models/` and exposes a `/predict` endpoint for the frontend to call.

Planned:
- Reuse `src/data/transforms.py` (`eval_transform`) so inference preprocessing exactly matches training.
- Reuse `src/models/*` to reconstruct the right architecture from the checkpoint's `model_name`.
- `/predict` (POST, multipart image upload) -> `{class, confidence}`.
- `/health` endpoint.
- Deps to confirm before adding: `fastapi`, `uvicorn`, `python-multipart`.

Built once a trained checkpoint exists in `models/`.
