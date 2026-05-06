# Repo Context

A short technical summary so Claude Design doesn't need to crawl the whole codebase.

## Project structure (top level)

```
ufc_predict/
  ingest/      Scrapers: UFC stats (greco), SportsBet odds, ESPN schedule, UFC.com fighter metadata, Wikidata
  features/    Feature builders: rolling stats, Elo ratings, age curves, SOS, style mismatch, chin/durability
  models/      LightGBM training, isotonic calibration, split-conformal intervals, prop models, totals models
  eval/        Backtests, bet analysis (with empirical-ROI gating), data audit, performance tracking
  serve/
    build_dashboard.py       Reads predictions/past_events/schedule/metadata → Jinja2 → docs/index.html
    templates/dashboard.html ~2,000-line single-file Jinja2 template (HTML + CSS + JS inline)
  cli.py       `ufc-predict`, `ufc-predict dashboard`, `ufc-ingest sportsbet`
data/
  ufc.db, feature_matrix.parquet, predictions.json, past_events.json, upcoming_schedule.json,
  fighter_metadata.json, fighter_images.json, model_performance.json
models/
  lgbm_model.pkl, isotonic.pkl, conformal_quantiles*.json, prop_models.pkl, ensemble.pkl, meta_blender.pkl
docs/
  index.html, fighter-images/<slug>.png   (GitHub Pages site root)
.github/workflows/
  daily refresh + weekly retrain (no n8n; CI handles scheduling)
```

## Prediction pipeline (one paragraph)

A daily GitHub Actions workflow scrapes upcoming UFC bouts from ESPN and live odds from SportsBet AU, joins them onto fighter records resolved via Wikidata, builds a feature matrix from career history (Elo, EWMA-decayed rolling stats, age curves, SOS, takedown defence, chin/durability scores, style-mismatch indicators), and feeds the matrix into a LightGBM model. Outputs are isotonically calibrated, wrapped with a split-conformal 90% interval, and post-processed by a meta-blender for prop probabilities (method × round). A separate bet-analysis pass compares model probabilities against SportsBet odds, sizes Kelly fractions, and flags `is_value: true` ONLY for `(bet_type, edge_bucket)` combos that have shown positive ROI in a 26k-bet historical backtest. Predictions are written to `data/predictions.json`; the dashboard generator reads them, merges in fighter metadata + ESPN live status, and renders `docs/index.html`. A weekly job retrains the model from scratch on the latest data.

## Existing styling / theme

The current dashboard defines its theme as inline CSS variables in `templates/dashboard.html`:

```css
:root {
  --red:    #e63946;       /* fighter A corner */
  --blue:   #4a90d9;       /* fighter B corner */
  --matrix: #00ff88;       /* current accent (Matrix-green, being replaced) */
  --purple: #c8a0fa;       /* secondary accent */
  --dark:   #0a0a0c;       /* background */
  --mid:    #131316;
  --card:   #101013;
  --border: #1f1f25;
  --light:  #eef1f5;       /* primary text */
  --muted:  #6f6f78;       /* secondary text */
  --gold:   #f4a261;
  --green:  #2dc653;       /* win/correct prediction */
  --yellow: #f7c948;
}
```

**The redesign drops most of this.** The new variables we want:

```css
:root {
  --bg:        #0A0A0B;    /* page background */
  --surface:   #141416;    /* card / elevated surface */
  --surface-2: #1B1B1F;    /* hover / secondary elevation */
  --border:    #25252B;    /* hairline */
  --accent:    #FFB300;    /* electric amber — used sparingly */
  --text:      #F5F2EC;    /* warm white, primary */
  --text-2:    #8A8A92;    /* muted grey, secondary */
  --text-3:    #4A4A52;    /* dim grey, tertiary / labels */
}
```

Fonts: Inter (UI), JetBrains Mono (numbers / stats).

## Frontend constraints to respect on re-import

1. **Static-site delivery.** The build target is a single HTML file written to `docs/`, served by GitHub Pages. Whatever Claude Design ships (React/Next/Astro), the user will port the markup + styles into `templates/dashboard.html` (Jinja2). Component-style outputs are fine; we'll inline them.
2. **No backend at runtime.** All data is baked into the page at build time from JSON files. There is no API. Charts/tables must work from inline data.
3. **Asset paths.** Fighter images live at `fighter-images/<slug>.png` relative to the page. The image URL is already in the data (`fighter.image_url`).
4. **Build command.** `python -m ufc_predict.serve.build_dashboard` (also exposed as `ufc-predict dashboard`).
5. **One page, multiple "screens".** Today the screens are tabs in a single document. The redesign can keep that or move to client-side routing — but it must remain a single static HTML deliverable.

## Things the data has that the current dashboard wastes

- `drivers` (per-bout feature attribution) — currently buried in a stats table. Could be the *hero* of the Fight Detail screen.
- `historical_roi_pct` per bet type — currently a small footnote. Should be a credibility-anchor in the Top Bets view.
- `ci_90_lo` / `ci_90_hi` — currently rendered as plain text. Could be the visual signature of the win-probability hero.
- `confidence_tier` — currently a tiny badge. Should drive the prominence of the prediction.

## What the user (Ainsley) cares about

- Restraint over flash. The look should feel earned, not loud.
- Density. He's reading 12 bouts, not browsing.
- Empirical credibility. Every probability should be visibly tied to a CI; every bet to a backtest result.
- One "wow" moment per screen, not five.
