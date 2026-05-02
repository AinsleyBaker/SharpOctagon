# CLAUDE.md

Calibrated UFC win probabilities + CIs for upcoming cards. LightGBM + isotonic + split-conformal; SQLite (Postgres if deployed); static HTML dashboard; GitHub Actions schedules (no n8n).

## Layout
- `ufc_predict/` — package: `ingest/`, `features/`, `models/`, `eval/`, `serve/`, `db/`, `cli.py`
- `data/` — `ufc.db`, `feature_matrix.parquet`, `predictions.json`, `raw/`
- `models/` — trained artifacts, conformal quantiles
- `migrations/` — alembic; `tests/` — pytest; `docs/` — generated dashboard
- `.github/workflows/` — daily refresh + weekly retrain
- `ufc_prediction_system_research.md` — design brief; §9 = build sequence

## Commands
- Test: `pytest -q`
- Lint: `ruff check .`
- Train: `python -m ufc_predict.models.train_runner`
- Predict + dashboard: `ufc-predict && ufc-predict dashboard`
- Odds refresh: `ufc-ingest sportsbet`

## Discipline
- **Root cause first.** Before any change, state ≥95% confidence in the cause and the reason. Otherwise gather data with tools — don't guess.
- **Empty output = failure.** Verify shape at every step; never assume.
- **One change at a time.** Minimal, reversible. Re-run and validate after each.
- **No leakage.** Chronological splits only; canonical fighter IDs, not names; no post-fight stats as features.
- **Tight context.** Targeted reads/greps; never scan blindly.
