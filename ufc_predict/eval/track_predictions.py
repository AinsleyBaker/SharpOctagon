"""
Prediction tracking and backtesting.

Two things:
  1. snapshot_predictions() — called after each prediction run to archive
     the prediction with a timestamp. Stored in data/prediction_history/.

  2. evaluate_past_predictions() — after results are ingested (via
     greco_loader), matches archived predictions to fight outcomes and
     computes accuracy, calibration, and realised EV metrics.

Run manually after a fight week:
    python -m ufc_predict.eval.track_predictions
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HISTORY_DIR     = Path("data/prediction_history")
PERFORMANCE_PATH = Path("data/model_performance.json")
PREDICTIONS_PATH = Path("data/predictions.json")
PAST_EVENTS_PATH = Path("data/past_events.json")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot_predictions(predictions: list[dict] | None = None) -> Path:
    """
    Save the current predictions with a datestamp (for evaluation later)
    AND merge any past-dated bouts into data/past_events.json so the
    dashboard can show them after they fall out of upcoming.
    """
    if predictions is None:
        if not PREDICTIONS_PATH.exists():
            raise FileNotFoundError(f"No predictions at {PREDICTIONS_PATH}")
        with open(PREDICTIONS_PATH) as f:
            predictions = json.load(f)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    path = HISTORY_DIR / f"predictions_{ts}.json"
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    log.info("Prediction snapshot saved: %s (%d bouts)", path, len(predictions))

    # Also persist any past-dated bouts to past_events.json (committed to repo).
    _accumulate_past_events(predictions)
    return path


def _accumulate_past_events(predictions: list[dict]) -> None:
    """
    Merge any past-dated predictions into data/past_events.json keyed by
    upcoming_bout_id. Existing entries are kept so we never lose a
    prediction once an event passes.
    """
    today_str = date.today().isoformat()

    # Load existing past events (keyed by bout_id for dedup)
    existing: dict[str, dict] = {}
    if PAST_EVENTS_PATH.exists():
        try:
            arr = json.loads(PAST_EVENTS_PATH.read_text(encoding="utf-8"))
            existing = {p.get("upcoming_bout_id", ""): p for p in arr if p.get("upcoming_bout_id")}
        except (json.JSONDecodeError, KeyError):
            existing = {}

    # Accumulate strictly-past events (event_date < today) into past_events.json.
    # Today's events are LEFT in upcoming so that mid-card the user still sees
    # the bouts that haven't happened yet. The dashboard handles per-bout
    # completion within a live event using ESPN status (see upcoming_poller).
    for p in predictions:
        ev_date = str(p.get("event_date", ""))
        bid = p.get("upcoming_bout_id", "")
        if ev_date and bid and ev_date < today_str:
            existing[bid] = p

    # Save back as a sorted list, newest events first
    out = sorted(existing.values(),
                 key=lambda p: str(p.get("event_date", "")),
                 reverse=True)
    PAST_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAST_EVENTS_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("Past events accumulated: %d total in %s", len(out), PAST_EVENTS_PATH)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _brier(prob: float, outcome: int) -> float:
    return (prob - outcome) ** 2


def _log_loss(prob: float, outcome: int) -> float:
    p = max(1e-6, min(1 - 1e-6, prob))
    return -(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))


def evaluate_past_predictions(db_url: str | None = None) -> dict:
    """
    Match archived predictions against fight outcomes now in the DB.

    For each archived prediction with a fight date in the past, looks up the
    actual result in fights table and computes:
      - accuracy          (correct winner predicted)
      - mean Brier score  (lower = better calibrated)
      - mean log-loss
      - realised EV       (if sportsbet_odds recorded, did value bets win?)

    Results are written to data/model_performance.json and returned.
    """
    from sqlalchemy import text

    from ufc_predict.db.session import get_session_factory

    factory  = get_session_factory(db_url)
    today    = date.today()
    records  = []
    seen_keys: set[tuple] = set()

    def _ingest(p: dict, source: str) -> None:
        event_date_str = p.get("event_date")
        if not event_date_str:
            return
        try:
            event_date = date.fromisoformat(str(event_date_str)[:10])
        except ValueError:
            return
        if event_date >= today:
            return  # fight hasn't happened yet
        bid = p.get("upcoming_bout_id") or ""
        key = (str(event_date), p.get("fighter_a_name"), p.get("fighter_b_name"), bid)
        if key in seen_keys:
            return
        seen_keys.add(key)
        records.append({
            "snap_file":      source,
            "event_date":     event_date,
            "event_name":     p.get("event_name"),
            "fighter_a":      p.get("fighter_a_name"),
            "fighter_b":      p.get("fighter_b_name"),
            "prob_a_wins":    float(p.get("prob_a_wins") or 0.5),
            "sportsbet_odds": p.get("sportsbet_odds"),
            "bet_analysis":   p.get("bet_analysis") or [],
        })

    # Primary source: past_events.json — accumulated across runs and committed
    # to the repo, so it survives ephemeral CI runners. Snapshot files in
    # HISTORY_DIR are a local-only fallback.
    if PAST_EVENTS_PATH.exists():
        try:
            for p in json.loads(PAST_EVENTS_PATH.read_text(encoding="utf-8")):
                _ingest(p, PAST_EVENTS_PATH.name)
        except json.JSONDecodeError:
            log.warning("past_events.json unreadable — falling back to snapshot dir")

    for snap_path in sorted(HISTORY_DIR.glob("predictions_*.json")):
        with open(snap_path) as f:
            preds = json.load(f)
        for p in preds:
            _ingest(p, snap_path.name)

    if not records:
        log.info("No past predictions to evaluate.")
        return {}

    # Batch-fetch outcomes from DB by fighter names
    with factory() as session:
        results = _fetch_outcomes(records, session)

    if not results:
        log.info("No matching fight outcomes found in DB.")
        return {}

    briers, log_losses, correct, total = [], [], 0, 0
    ev_realised, ev_n = 0.0, 0

    for r in results:
        prob_a = r["prob_a_wins"]
        won_a  = int(r["a_won"])
        briers.append(_brier(prob_a, won_a))
        log_losses.append(_log_loss(prob_a, won_a))
        predicted_winner = "a" if prob_a >= 0.5 else "b"
        actual_winner    = "a" if won_a else "b"
        correct += int(predicted_winner == actual_winner)
        total   += 1

        # Realised EV: did the value bets cash?
        for bet in r.get("bet_analysis") or []:
            if not bet.get("is_value"):
                continue
            # Only moneyline realisable without knowing exact method/round outcome
            if bet.get("bet_type") != "moneyline":
                continue
            desc = (bet.get("description") or "").lower()
            if r["fighter_a"].split()[-1].lower() in desc:
                bet_won = won_a
            elif r["fighter_b"].split()[-1].lower() in desc:
                bet_won = 1 - won_a
            else:
                continue
            odds = bet.get("sb_odds", 1.0)
            ev_realised += (odds - 1) if bet_won else -1.0
            ev_n += 1

    perf = {
        "evaluated_at":      datetime.now(timezone.utc).isoformat(),
        "n_bouts":           total,
        "accuracy":          round(correct / total, 4) if total else None,
        "mean_brier":        round(float(np.mean(briers)), 4) if briers else None,
        "mean_log_loss":     round(float(np.mean(log_losses)), 4) if log_losses else None,
        "value_bets_evaluated": ev_n,
        "realised_ev_per_bet":  round(ev_realised / ev_n, 4) if ev_n else None,
        "fight_details":     results,
    }

    PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if PERFORMANCE_PATH.exists():
        with open(PERFORMANCE_PATH) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]
    existing.append(perf)
    with open(PERFORMANCE_PATH, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    log.info(
        "Performance: %d bouts | accuracy=%.1f%% | brier=%.4f | log_loss=%.4f",
        total,
        (correct / total * 100) if total else 0,
        float(np.mean(briers)) if briers else 0,
        float(np.mean(log_losses)) if log_losses else 0,
    )
    return perf


def _fetch_outcomes(records: list[dict], session) -> list[dict]:
    """
    Match prediction records to fight rows in the DB using fighter names.
    Returns records enriched with a_won, fighter_a, fighter_b.
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            f.fight_id, f.date,
            fa.full_name AS name_a,
            fb.full_name AS name_b,
            f.winner_fighter_id,
            f.red_fighter_id,
            f.blue_fighter_id,
            f.method
        FROM fights f
        JOIN fighters fa ON fa.canonical_fighter_id = f.red_fighter_id
        JOIN fighters fb ON fb.canonical_fighter_id = f.blue_fighter_id
        WHERE f.date < :today
          AND f.method IS NOT NULL
          AND f.winner_fighter_id IS NOT NULL
        ORDER BY f.date DESC
        LIMIT 500
    """)
    db_fights = session.execute(sql, {"today": date.today()}).fetchall()

    from rapidfuzz import fuzz

    def _score(a: str, b: str) -> int:
        return max(
            fuzz.token_set_ratio(a, b),
            fuzz.partial_ratio(a.split()[-1], b),
            fuzz.partial_ratio(a, b.split()[-1]),
        )

    results = []
    for rec in records:
        pa = rec["fighter_a"]
        pb = rec["fighter_b"]
        best_score, best_row = 0, None
        for row in db_fights:
            s = min(_score(pa, row.name_a), _score(pb, row.name_b))
            if s > best_score:
                best_score, best_row = s, row
        if best_row is None or best_score < 70:
            continue

        # Determine if fighter_a (red corner mapped by name) won
        winner_is_red = best_row.winner_fighter_id == best_row.red_fighter_id
        # Check which of our a/b maps to red
        score_a_red = _score(pa, best_row.name_a)
        score_a_blue = _score(pa, best_row.name_b)
        a_is_red = score_a_red >= score_a_blue
        a_won = int(winner_is_red == a_is_red)

        results.append({
            **rec,
            "a_won":    a_won,
            "db_date":  str(best_row.date),
            "method":   best_row.method,
            "match_score": best_score,
        })

    return results


# ---------------------------------------------------------------------------
# Backtesting on historical data
# ---------------------------------------------------------------------------

def backtest(db_url: str | None = None, since_year: int = 2020) -> dict:
    """
    Run the current model on historical fights and measure accuracy.
    Uses OOF (out-of-fold) predictions from the training run if available,
    otherwise re-runs predictions on the full historical dataset.

    Returns accuracy, Brier score, log-loss across all evaluated fights.
    """
    oof_path = Path("models/oof_predictions.parquet")
    if not oof_path.exists():
        log.warning("OOF predictions not found at %s — run weekly_retrain first", oof_path)
        return {}

    import pandas as pd
    oof = pd.read_parquet(oof_path)
    # train.run_cv writes the OOF prediction column as `pred_prob`. Older
    # snapshots may have used `oof_prob`; accept either to stay compatible.
    prob_col = "pred_prob" if "pred_prob" in oof.columns else (
        "oof_prob" if "oof_prob" in oof.columns else None
    )
    if oof.empty or "label" not in oof.columns or prob_col is None:
        log.warning("OOF parquet missing required columns: label, pred_prob/oof_prob")
        return {}

    oof = oof[oof["date"].dt.year >= since_year].copy() if "date" in oof.columns else oof

    labels = oof["label"].values
    probs  = oof[prob_col].values

    accuracy   = float(((probs >= 0.5).astype(int) == labels).mean())
    brier      = float(np.mean((probs - labels) ** 2))
    ll_vals    = [_log_loss(p, y) for p, y in zip(probs, labels)]
    log_loss_m = float(np.mean(ll_vals))

    result = {
        "since_year": since_year,
        "n_fights":   int(len(oof)),
        "accuracy":   round(accuracy, 4),
        "brier":      round(brier, 4),
        "log_loss":   round(log_loss_m, 4),
    }
    log.info(
        "Backtest (%d+): %d fights | accuracy=%.1f%% | brier=%.4f | log_loss=%.4f",
        since_year, len(oof), accuracy * 100, brier, log_loss_m,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Evaluating past predictions ===")
    perf = evaluate_past_predictions()
    if perf:
        print(f"Accuracy:   {perf.get('accuracy', 'N/A')}")
        print(f"Brier:      {perf.get('mean_brier', 'N/A')}")
        print(f"Log-loss:   {perf.get('mean_log_loss', 'N/A')}")
    print("\n=== Backtest on historical OOF data ===")
    bt = backtest()
    if bt:
        print(f"Accuracy:   {bt['accuracy']}")
        print(f"Brier:      {bt['brier']}")
        print(f"Log-loss:   {bt['log_loss']}")
