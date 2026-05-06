# Data Shapes

These are the actual JSON shapes the Moirai frontend consumes today, derived from `data/predictions.json`, `data/past_events.json`, `data/upcoming_schedule.json`, `data/fighter_metadata.json`, and `data/model_performance.json`. Examples use real UFC fighters with realistic values.

---

## 1. Upcoming Fights screen

The page receives a list of **events**, each with an array of **bouts**. The shape below is one bout from `data/predictions.json` with light renaming for clarity.

```json
{
  "upcoming_bout_id": "a31f8e0c4d2b9117",
  "event_date": "2026-05-09",
  "event_name": "UFC 328: Chimaev vs. Strickland",
  "is_main_event": true,
  "is_title_bout": true,
  "is_five_round": true,
  "scheduled_rounds": 5,
  "weight_class": "Middleweight",
  "card_segment": "Main Card",
  "start_time_iso": "2026-05-10T05:00Z",
  "fighter_a": {
    "name": "Khamzat Chimaev",
    "record": "14-0-0",
    "nationality": "United Arab Emirates",
    "stance": "Orthodox",
    "age": 31,
    "image_url": "fighter-images/khamzat-chimaev.png",
    "ufc_style": "Wrestler",
    "win_streak": 14,
    "loss_streak": 0,
    "l3_win_rate": 1.0,
    "ko_rate": 0.43,
    "sub_rate": 0.36,
    "finish_rate": 0.79
  },
  "fighter_b": {
    "name": "Sean Strickland",
    "record": "29-6-0",
    "nationality": "United States",
    "stance": "Southpaw",
    "age": 35,
    "image_url": "fighter-images/sean-strickland.png",
    "ufc_style": "Striker",
    "win_streak": 0,
    "loss_streak": 1,
    "l3_win_rate": 0.33,
    "ko_rate": 0.21,
    "sub_rate": 0.04,
    "finish_rate": 0.25
  },
  "prediction": {
    "prob_a_wins": 0.738,
    "prob_b_wins": 0.262,
    "ci_90_lo": 0.62,
    "ci_90_hi": 0.84,
    "uncertainty_std": 0.058,
    "confidence_tier": "High",
    "data_quality": "good",
    "headline_method": "KO/TKO",
    "headline_round": "R2",
    "method_summary": "Chimaev to win by submission or TKO inside the distance"
  },
  "props": {
    "prob_a_wins_ko_tko": 0.241,
    "prob_a_wins_sub": 0.367,
    "prob_a_wins_dec": 0.130,
    "prob_b_wins_ko_tko": 0.118,
    "prob_b_wins_sub": 0.012,
    "prob_b_wins_dec": 0.132,
    "prob_finish": 0.738,
    "prob_decision": 0.262,
    "prob_rounds": { "R1": 0.21, "R2": 0.27, "R3": 0.14, "R4": 0.07, "R5": 0.05 }
  },
  "drivers": [
    { "feature": "Grappling control differential", "impact": 0.18, "favors": "A" },
    { "feature": "Takedown defence rating",         "impact": 0.11, "favors": "A" },
    { "feature": "Cardio (round 4–5 output)",       "impact": 0.06, "favors": "B" },
    { "feature": "Chin / durability score",         "impact": 0.05, "favors": "B" },
    { "feature": "Reach",                            "impact": 0.04, "favors": "B" }
  ],
  "value_bet_count": 2,
  "has_edge": true
}
```

### Field notes
- `prob_a_wins == 0.5` exactly is a **sentinel**: model didn't run for this bout. Render as "preview only" / ungraded.
- `confidence_tier` ∈ `"High" | "Medium" | "Low"` — derived from `uncertainty_std`.
- `data_quality` ∈ `"good" | "limited" | "sparse"` — based on minimum of fighters' UFC fight counts.
- `drivers` is the model's local feature attribution for this bout, sorted by `|impact|`. Top 5 is plenty for a row preview; full list goes on the detail screen.
- `headline_method` / `headline_round` are derived: argmax over the per-fighter method probs, then argmax over `prob_rounds`. Used for the one-line method summary on the row.

---

## 2. Fight Detail screen

Same bout object as above, plus:

```json
{
  "sportsbet_odds": {
    "source": "sportsbet.com.au",
    "moneyline_a": 1.62,
    "moneyline_b": 2.35,
    "method": {
      "Khamzat Chimaev by KO/TKO": 3.2,
      "Khamzat Chimaev by Submission": 2.6,
      "Khamzat Chimaev by Points": 6.5,
      "Sean Strickland by KO/TKO": 9.0,
      "Sean Strickland by Points": 4.4,
      "Sean Strickland by Submission": 41.0
    },
    "method_neutral": { "KO/TKO": 1.95, "Submission": 2.4, "Points": 3.5 },
    "distance": { "Yes": 3.2, "No": 1.32 },
    "total_rounds": { "Over": 2.55, "Under": 1.48 },
    "winning_round": {
      "Khamzat Chimaev to Win in Round 1": 4.5,
      "Khamzat Chimaev to Win in Round 2": 5.0,
      "Khamzat Chimaev to Win by Decision": 7.5,
      "Sean Strickland to Win by Decision": 5.0
    },
    "round_survival": { "Yes": 2.4, "No": 1.55 }
  },
  "bet_analysis": [
    {
      "bet_type": "method",
      "description": "Khamzat Chimaev wins by Submission",
      "our_prob": 0.367,
      "sb_odds": 2.60,
      "implied_prob": 0.385,
      "edge": -0.018,
      "ev_pct": -4.6,
      "kelly_pct": 0.0,
      "is_value": false,
      "historical_roi_pct": 7.8,
      "historical_n_bets": 412,
      "market_backtested": true
    },
    {
      "bet_type": "winning_round",
      "description": "Khamzat Chimaev to Win in Round 2",
      "our_prob": 0.193,
      "sb_odds": 5.00,
      "implied_prob": 0.20,
      "edge": -0.007,
      "ev_pct": -3.5,
      "kelly_pct": 0.0,
      "is_value": false,
      "historical_roi_pct": 14.2,
      "historical_n_bets": 188,
      "market_backtested": true
    }
  ]
}
```

### Field notes
- `is_value` is the **empirical gate**: the model has a positive edge AND that bet_type/edge bucket has shown positive ROI in backtest. The dashboard MUST NOT recommend value on `moneyline` bets historically (negative ROI in every edge bucket).
- `historical_roi_pct` / `historical_n_bets` come from the per-market backtest — surface them in the bet card to give credibility.
- The full `bet_analysis` array can be 20–40 entries per bout; treat as a sortable, filterable table.

---

## 3. Top Bets screen

Computed from `bet_analysis` across all upcoming bouts. Shape:

```json
{
  "bankroll": 1000,
  "strategy": "kelly_quarter",
  "summary": {
    "total_stake": 142.50,
    "expected_return": 38.20,
    "expected_ev_pct": 26.8,
    "n_bets": 9,
    "events_covered": 2
  },
  "bets": [
    {
      "event_name": "UFC 328: Chimaev vs. Strickland",
      "fighter_a": "Carlos Prates",
      "fighter_b": "Belal Muhammad",
      "bet_type": "method_combo",
      "description": "Carlos Prates wins by KO or Submission",
      "our_prob": 0.418,
      "sb_odds": 3.40,
      "implied_prob": 0.294,
      "edge": 0.124,
      "ev_pct": 42.1,
      "kelly_pct": 5.2,
      "stake": 52.00,
      "expected_return": 21.90,
      "historical_roi_pct": 11.4,
      "historical_n_bets": 287
    }
  ]
}
```

---

## 4. Past Events screen

Same bout object as Upcoming Fights, plus result fields persisted from ESPN:

```json
{
  "is_completed": true,
  "live_winner": "Ilia Topuria",
  "live_method": "KO/TKO",
  "live_round": 2,
  "result_text": "Ilia Topuria by KO/TKO R2",
  "pred_correct": true
}
```

Plus event-level rollup (from `data/model_performance.json`):

```json
{
  "evaluated_at": "2026-05-02T11:26:52Z",
  "n_bouts": 12,
  "accuracy": 0.667,
  "mean_brier": 0.198,
  "mean_log_loss": 0.612,
  "value_bets_evaluated": 6,
  "realised_ev_per_bet": 0.31
}
```

Plus a rolling-trend series for a sparkline:

```json
{
  "rolling": [
    { "event_date": "2026-03-22", "accuracy": 0.583, "log_loss": 0.681 },
    { "event_date": "2026-03-29", "accuracy": 0.625, "log_loss": 0.654 },
    { "event_date": "2026-04-05", "accuracy": 0.692, "log_loss": 0.622 },
    { "event_date": "2026-04-12", "accuracy": 0.667, "log_loss": 0.611 },
    { "event_date": "2026-04-19", "accuracy": 0.700, "log_loss": 0.598 },
    { "event_date": "2026-04-25", "accuracy": 0.667, "log_loss": 0.612 }
  ]
}
```
