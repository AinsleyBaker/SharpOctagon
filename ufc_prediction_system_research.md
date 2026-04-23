# UFC Fight Prediction System — Research & Design Brief
 
**Prepared for:** Ainsley
**Date:** 2026-04-21
**Status:** Research complete. Implementation blocked on the open decisions in §10.
 
---
 
## 0. TL;DR
 
You want a system that produces **calibrated win probabilities with confidence intervals** for **upcoming UFC cards**, refreshed on a schedule. This brief answers the four questions Claude Code will need settled before it writes a line of code:
 
1. **Where does the data come from?** Primary: scrape UFCStats.com (use `Greco1899/scrape_ufc_stats` as the base — it publishes daily-refreshed CSVs). Supplementary: Sherdog (pre-UFC + DOB), Tapology (earliest upcoming-card announcements), Kaggle only as a sanity cross-check, not the source of truth.
2. **What's the data model?** Six canonical tables — `events`, `fights`, `fight_stats_round`, `fighters`, `fighter_snapshots`, `upcoming_bouts` — keyed on canonical fighter IDs bootstrapped from Wikidata + fuzzy-match.
3. **What's the model?** LightGBM (log-loss objective) on difference features + Elo/Glicko-2 ratings + rolling recent-form stats. Isotonic calibration on a held-out temporal fold. Bootstrap-bag + split-conformal for confidence intervals.
4. **How do we predict live?** Daily poll of Tapology + UFC.com's undocumented `api/v3/events/live/{id}.json` endpoint; increase to 6-hourly in fight week; predictions written to a view that refreshes on every poll.
The **non-negotiable disciplines** that most public UFC models fail at: chronological train/val/test splits, no post-fight-stat leakage, randomized corner assignment during training, and benchmarking log-loss against the Vegas closing line. Getting these right is what separates a toy from something usable.
 
---
 
## 1. Scope & Target (confirmed with you)
 
- **Prediction target:** calibrated win probability P(A beats B), optimized for Brier/log-loss, with a confidence interval (bootstrap + conformal).
- **Use case:** live predictions for upcoming UFC cards, refreshed on a schedule.
- **Data sources approved:** UFCStats.com (primary), Kaggle (supplementary), Sherdog + Tapology (supplementary). **No betting odds as features** — but see §7 where I recommend using closing lines for **evaluation only** as a benchmark we try to beat. Flagging for your call in §10.
- **Deliverable from research phase:** this document.
---
 
## 2. Data Sources Evaluation
 
### 2.1 UFCStats.com — PRIMARY source
 
Official UFC stats. HTTP-only site, no official API. Scraping is the only access method.
 
**URL structure** (verified from multiple scrapers):
 
| Page | Pattern | Example |
|---|---|---|
| Events (completed) | `/statistics/events/completed?page=all` | http://ufcstats.com/statistics/events/completed?page=all |
| Events (upcoming) | `/statistics/events/upcoming` | http://ufcstats.com/statistics/events/upcoming |
| Event detail | `/event-details/{16-hex-id}` | — |
| Fight detail | `/fight-details/{16-hex-id}` | — |
| Fighter detail | `/fighter-details/{16-hex-id}` | — |
| Fighter directory | `/statistics/fighters?char=[a-z]&page=all` | — |
 
IDs are opaque 16-char hex hashes; must be discovered by crawling from events → event-details → fight-details. **Join on these IDs, never on fighter name** — names have diacritics, nicknames, and historical renames.
 
**Fields captured per fight:**
- **Totals per fighter:** KD, SIG.STR. (landed/attempted), SIG.STR.%, TOTAL STR. (landed/attempted), TD (landed/attempted), TD%, SUB.ATT, REV., CTRL (mm:ss).
- **Significant strike splits** by target (HEAD / BODY / LEG) and position (DISTANCE / CLINCH / GROUND), landed-of-attempted.
- **Per-round breakdowns** of everything above.
- **Fight metadata:** weight class, method (KO/TKO/SUB/Decision/DQ), round ended, time, time format (3×5 or 5×5), referee, performance bonuses, title-bout flag.
- **Fighter profile:** name, nickname, DOB, height, reach, stance, weight class, career aggregates (SLpM, Str. Acc., SApM, Str. Def., TD Avg, TD Acc., TD Def., Sub. Avg.), full fight history with links.
**Recommended scraper:** **`Greco1899/scrape_ufc_stats`** (https://github.com/Greco1899/scrape_ufc_stats). ~135 stars, last manual refresh Nov 2025, runs a daily Cloud Run + Cloud Scheduler refresh. Output is **six CSVs committed directly to the repo** (`ufc_events.csv`, `ufc_fight_details.csv`, `ufc_fight_results.csv`, `ufc_fight_stats.csv`, `ufc_fighter_details.csv`, `ufc_fighter_tott.csv`). We can `git pull` daily rather than running our own crawl — enormous time-saver. Fallback: we can still run our own scrape against UFCStats directly using its code as reference.
 
**Alternative scrapers worth knowing:**
- `remypereira99/UFC-Web-Scraping` — Scrapy-based, Docker-ready, schema via dataclasses.
- `natebuel29/ufc-stats-scraper` — notable for also scraping *upcoming* fights.
- `ufcscraper` on PyPI (v1.1.0, Aug 2025) — packaged; also scrapes BestFightOdds and matches odds to fighters.
**Scraping etiquette:** robots.txt not directly verified from this environment. Community convention: desktop User-Agent, 0.5–2 s delay per request. No reports of rate-limiting or IP bans. Risk profile low for research use.
 
**Known gotchas:**
- **Early UFC events missing per-round data** (pre-UFC 31, ~2001). Expect NaNs.
- **Pride data** imported, but without per-round breakdowns.
- **NC / DQ bouts** present; exclude from win/loss learning but keep in fighter-history ledger.
- **Title-bout flag** was historically missed by some scrapers; confirm it's in your extract.
- **Control time** is `mm:ss` strings — convert to seconds on ingest.
- **Cancelled bouts** sometimes remain on event pages with null outcomes — filter.
- **Women's divisions** fully covered from UFC 157 (Feb 2013) onward, identical schema.
### 2.2 Sherdog.com — Supplementary (pre-UFC & DOB)
 
**What it adds over UFCStats:** pre-UFC professional records, non-UFC fights (Bellator, ONE, PFL, regional, amateur), reliable DOBs, nationality, training camp, nicknames. Critical for fighter-history features that start before their UFC debut.
 
**URL structure:** `https://www.sherdog.com/fighter/{Name-Hyphenated}-{NumericID}` (e.g. `/fighter/Conor-McGregor-29688`).
 
**Scraping:** No maintained PyPI package. Best starting points (all likely need selector patches):
- `jc0n/sherdog-scraper` (Python, minimal).
- `Montanaz0r/MMA-parser-for-Sherdog-and-UFC-data`.
- `ufc-api` on PyPI (v0.0.1, July 2023) — stale but functional.
- Hosted alternative: Apify `sherdog-profile-scraper` (paid, maintained).
**Difficulty:** low-to-moderate. `requests` + `BeautifulSoup` with a real UA and 1–2 s delays typically works. ToS prohibits scraping — respect robots.txt and throttle aggressively. *Unverified* whether Cloudflare bot management has been added recently; assume it could change.
 
### 2.3 Tapology.com — Supplementary (upcoming cards & regional history)
 
**What it adds:** earliest posting of announced/rumored bouts (often weeks before UFC.com confirms), deep non-UFC history, rankings, training camps, aggregated betting odds summaries.
 
**URL structure:**
- Fighter: `https://www.tapology.com/fightcenter/fighters/{id}-{slug}`
- Event: `https://www.tapology.com/fightcenter/events/{id}-{slug}`
- UFC promotion hub: `https://www.tapology.com/fightcenter/promotions/1-ultimate-fighting-championship-ufc`
**Scraping:** **HARD**. Every existing scraper's README warns about aggressive defenses: IP banning after sustained requests, JS-based pagination, non-standard HTML, and ToS that explicitly forbids bots/scrapers.
- `ehan03/Tapology-Scraper` — best starting point, but **explicitly unmaintained**; author warns the `all` mode "will 100% get your IP blocked."
- `angel-721/tapology-python-parser` — minimal.
- `CianHub/python-tapology-events-web-scraper` — upcoming events only, to spreadsheet.
**Mitigations needed if we use it:** residential proxies with rotation, ≤1 req / 3–5 s, randomized UA + Accept-Language, TLS-fingerprint match via `curl_cffi` or `cloudscraper`. Honestly, **Tapology is only worth the effort for upcoming-card announcements earlier than UFC.com** — if UFC.com's lead time is acceptable, skip Tapology.
 
### 2.4 Kaggle — Cross-check only
 
Use as a sanity check against our own UFCStats scrape, not as source of truth (licensing, staleness, possible leakage in pre-engineered features).
 
- **`neelagiriaditya/ufc-datasets-1994-2025`** — most current; through UFC 319, last update Sept 2025.
- **`mdabbert/ultimate-ufc-dataset`** — cleanest schema (pre-engineered `R_`/`B_` differentials) but last update Dec 2024 and **the pre-engineered rolling features have been repeatedly flagged for leakage**. Re-generate features yourself from UFCStats.
- `aminealibi/ufc-fights-fighters-and-events-dataset` — 2025 refresh in the ultimate-dataset format.
- `asaniczka/ufc-fighters-statistics` — fighter-level only, from UFCStats.
- `rajeevw/ufcdata` — stale (stops at 2021), but it's the baseline for `WarrierRajeev/UFC-Predictions`.
### 2.5 Live / upcoming card data sources
 
| Source | How | Lead time | Reliability | Use |
|---|---|---|---|---|
| **UFCStats "upcoming"** | HTML scrape | Late, schema-consistent with historical | High | Cross-check only |
| **UFC.com event pages** | `https://www.ufc.com/event/ufc-{N}` (slug) + undocumented `https://www.ufc.com/api/v3/events/live/{eventid}.json` | Official confirmation, moderate lead | High for final confirmation; fragile API (undocumented) | **Confirmation source** |
| **ESPN MMA** | `https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard` + event/competition endpoints | Competitive with UFC.com | High, no auth | **Cross-check** |
| **Tapology** | HTML scrape | **Earliest** — weeks ahead | Mid (rumored vs confirmed mixed) | **Earliest heads-up** (only if scraping budget allows) |
 
**Debunked:** there is no public `api.ufc.com`. Community projects all hit `www.ufc.com/api/v3/...` or scrape HTML.
 
**Typical lead time:** numbered PPV cards announced 8–12 weeks out, but individual bouts fill in across that window. Locks ~4–6 weeks out. **Last-minute changes are routine** — injuries, weight misses, visa issues. Plan for same-day card deltas.
 
**Recommended refresh schedule:**
- Historical (completed) data: **weekly**, post-event (Monday).
- Upcoming cards baseline: **daily** from ~8 weeks out.
- Fight week (T−7 days → T−0): **every 6 hours**, plus a final pull at **T−2 hours** to catch day-of replacements.
### 2.6 Fighter ID resolution
 
The cross-source join problem: UFCStats uses 16-char hex, Sherdog uses `{Name}-{intID}`, Tapology uses `{intID}-{slug}`, Kaggle keys on string names. Same fighter can appear as "Tony Ferguson" / "Anthony Ferguson" / "El Cucuy".
 
**Bootstrap strategy:**
1. Query Wikidata SPARQL for all entities with a `UFC athlete ID` (+ Sherdog ID P2818 + Tapology ID P9728). This gives a partial but high-quality mapping for well-known fighters.
2. For remaining UFCStats fighters, fuzzy-match to Sherdog with `rapidfuzz` token-sort ratio ≥ 90 **gated on `|dob_diff| ≤ 1 day`** (Sherdog has reliable DOBs; UFCStats has DOBs). DOB gating prevents the "two Michael Johnsons" problem.
3. Manual review of the long tail (probably <50 fighters).
4. Persist `canonical_fighter_id` table: `{canonical_id, ufcstats_hash, sherdog_id, tapology_id, wikidata_qid, name_variants[]}`.
No existing public project releases a canonical mapping CSV, so we own this table. Good news: we maintain it incrementally after the initial bootstrap.
 
---
 
## 3. Canonical Data Schema
 
Store as SQLite in dev, Postgres if we ever deploy. All timestamps UTC. All times in seconds.
 
### 3.1 `fighters` — one row per fighter (slow-changing attributes)
 
| Column | Type | Notes |
|---|---|---|
| `canonical_fighter_id` | TEXT PK | Our own UUID |
| `ufcstats_id` | TEXT | 16-hex hash |
| `sherdog_id` | TEXT | `{Name}-{NumericID}` |
| `tapology_id` | TEXT | `{id}-{slug}` |
| `wikidata_qid` | TEXT | Q-ID |
| `full_name` | TEXT | Canonical form |
| `nickname` | TEXT | |
| `name_variants` | JSON | All observed spellings |
| `dob` | DATE | |
| `nationality` | TEXT | |
| `stance` | TEXT | Current; time-versioned history in `fighter_snapshots` |
| `height_cm` | REAL | |
| `reach_cm` | REAL | |
| `primary_weight_class` | TEXT | Current; historical in snapshots |
 
### 3.2 `fighter_snapshots` — fighter attributes at a point in time
 
Needed because fighters change stance, weight class, and camps.
 
| Column | Type | Notes |
|---|---|---|
| `canonical_fighter_id` | TEXT FK | |
| `as_of_date` | DATE | |
| `stance` | TEXT | |
| `weight_class` | TEXT | |
| `camp` | TEXT | From Sherdog |
| `is_active` | BOOL | |
 
### 3.3 `events` — one row per UFC event
 
| Column | Type | Notes |
|---|---|---|
| `event_id` | TEXT PK | UFCStats hex |
| `ufc_event_number` | TEXT | `UFC 300`, `UFC Fight Night: X vs Y`, etc. |
| `date` | DATE | |
| `location` | TEXT | |
| `country` | TEXT | Parsed — used for home-country feature |
| `altitude_m` | REAL | Looked up by city |
 
### 3.4 `fights` — one row per bout
 
Corner assignment (red/blue) is a **source-of-truth quirk** — red is historically the betting favorite. To avoid leaking that signal into training, we randomize corner at feature-extraction time. Here we store raw.
 
| Column | Type | Notes |
|---|---|---|
| `fight_id` | TEXT PK | UFCStats hex |
| `event_id` | TEXT FK | |
| `date` | DATE | Denormalized for as-of queries |
| `red_fighter_id` | TEXT FK canonical | |
| `blue_fighter_id` | TEXT FK canonical | |
| `weight_class` | TEXT | |
| `is_title_bout` | BOOL | |
| `is_five_round` | BOOL | |
| `winner_fighter_id` | TEXT FK canonical | `NULL` if draw / NC |
| `method` | TEXT | KO / TKO / SUB / Decision (U/S/M) / DQ / NC |
| `method_detail` | TEXT | Ref stoppage, punches, RNC, etc. |
| `round_ended` | INT | |
| `time_ended_sec` | INT | |
| `referee` | TEXT | |
| `bonus_awards` | JSON | POTN, FOTN |
| `red_is_short_notice` | BOOL | Derived |
| `blue_is_short_notice` | BOOL | Derived |
| `red_missed_weight` | BOOL | |
| `blue_missed_weight` | BOOL | |
| `closing_odds_red` | REAL | American odds; **evaluation only**, never a feature |
| `closing_odds_blue` | REAL | |
 
### 3.5 `fight_stats_round` — one row per (fight, fighter, round)
 
| Column | Type | Notes |
|---|---|---|
| `fight_id` | TEXT FK | |
| `fighter_id` | TEXT FK canonical | |
| `round` | INT | 0 = fight total, 1..N = per-round |
| `knockdowns` | INT | |
| `sig_strikes_landed` | INT | |
| `sig_strikes_attempted` | INT | |
| `total_strikes_landed` | INT | |
| `total_strikes_attempted` | INT | |
| `head_landed` | INT | |
| `head_attempted` | INT | |
| `body_landed` | INT | |
| `body_attempted` | INT | |
| `leg_landed` | INT | |
| `leg_attempted` | INT | |
| `distance_landed` | INT | |
| `distance_attempted` | INT | |
| `clinch_landed` | INT | |
| `clinch_attempted` | INT | |
| `ground_landed` | INT | |
| `ground_attempted` | INT | |
| `takedowns_landed` | INT | |
| `takedowns_attempted` | INT | |
| `submission_attempts` | INT | |
| `reversals` | INT | |
| `control_time_sec` | INT | |
 
### 3.6 `upcoming_bouts` — scheduled but not yet fought
 
Drives live prediction. Written by the live-data pipeline.
 
| Column | Type | Notes |
|---|---|---|
| `upcoming_bout_id` | TEXT PK | Hash of (event + fighters) |
| `event_date` | DATE | |
| `event_name` | TEXT | |
| `red_fighter_id` | TEXT FK canonical | Null if not yet resolved |
| `blue_fighter_id` | TEXT FK canonical | |
| `red_name_raw` | TEXT | Before canonicalization |
| `blue_name_raw` | TEXT | |
| `weight_class` | TEXT | |
| `is_title_bout` | BOOL | |
| `is_five_round` | BOOL | |
| `source` | TEXT | `ufc.com` / `tapology` / `espn` |
| `first_seen_at` | TIMESTAMP | |
| `last_updated_at` | TIMESTAMP | |
| `is_confirmed` | BOOL | `false` if only rumored (Tapology) |
| `is_cancelled` | BOOL | |
 
### 3.7 Derived: `fighter_aso_features` — as-of snapshots
 
**This is the key anti-leakage table.** For each (fighter, target_date) we compute features using **only fights with `date < target_date`**. Never use cumulative career stats from UFCStats fighter pages directly — those include the fight you're trying to predict.
 
Implemented as a view or materialized table. Features per row listed in §4.
 
---
 
## 4. Feature Engineering Plan
 
Organized by bucket. Features marked **CORE** consistently move the needle in published time-split models; **std** are commonly included; **edge** are promising but under-explored. I'm recommending we ship with CORE + std at v1, then A/B the edge features.
 
### 4.1 Static attributes (per fighter, point-in-time)
- **Age at fight date** (CORE)
- **Years since pro debut** (std) — proxy for decline curves
- Reach (weak on its own)
- Height (weak)
- Stance (std)
- Weight class at fight (std)
### 4.2 Career aggregates (as-of, pre-fight only)
- SLpM, SApM, Sig-strike accuracy, Sig-strike defense (CORE)
- TD avg/15, TD accuracy, TD defense (CORE)
- Sub attempts / 15 min (CORE)
- Career W / L / D, KO rate, SUB rate (std)
- **With shrinkage** toward weight-class-division means for fighters with <5 UFC fights (avoids rookie-bias).
### 4.3 Rolling recent form (last-N, exponentially weighted)
**CORE** — outperforms career aggregates in every published time-split model.
- L3 and L5 fight striking-diff, TD-diff, control-time-diff
- L3 KD differential
- L3 finish rate
- **Opponent-adjusted** rolling stats (strike diff adjusted by opponent's defense rating) — edge, strong theoretical case, implement in v1 if time allows.
### 4.4 Rating systems
- **Elo per weight class** with tuned K-factor (CORE baseline; 58–62% standalone)
- **Glicko-2** (std; slightly beats Elo; the RD term itself is useful as an uncertainty feature)
- Multi-dimensional Elo (separate striking-Elo and grappling-Elo) — **edge**, under-explored, worth a spike
### 4.5 Style / stylistic matchup
- Striker vs grappler (derived from TD-attempt rate + strike volume via k-means cluster) (std/edge)
- Stance cross-term (southpaw-vs-orthodox) (std) — ~1–2 pp lift
- Style-cluster one-hots (edge)
### 4.6 Layoff / activity
- Days since last fight (std, nonlinear) — 60–180 d optimal; >400 d mildly negative
- Fight frequency in last 24 months (std)
### 4.7 Age curves
- Age (CORE)
- Age relative to weight-class peak (edge) — Tandfonline 2023 curves: LW/WW 26–31, MW/LHW 27–32, HW 28–34
- Years-since-debut (std) — proxies ~9–10 yr decline
### 4.8 Contextual
- **Short-notice flag + days-of-notice** (CORE)
- **Weight-miss flag** (std) — misser underperforms, counterintuitively
- **Home-country × Brazilian** interaction (std, regional) — the only home-cage effect that replicates
- Title-bout flag (std)
- 5-round-vs-3-round flag (std) — interacts with age + cardio
### 4.9 Difference features (always include)
- `age_diff`, `reach_diff`, `height_diff`, `SLpM_diff`, `TDacc_diff`, `TDdef_diff`, `Elo_diff`, `Glicko_diff`, rolling-form diffs (CORE)
- Ratio features (e.g. `SLpM_A / (SLpM_A + SLpM_B)`) — edge stabilizer
### 4.10 Symmetry handling
 
The A-vs-B vs B-vs-A problem. Three approaches; we're going with **symmetric swap augmentation** for training + **difference-only features at inference**.
 
1. **Symmetric swap augmentation (chosen for training):** include each fight twice with labels flipped. Keep swapped pairs in the same split (avoid leak). At inference, average `p(A>B)` with `1 − p(B>A)` to enforce consistency.
2. **Difference-only features (chosen for inference):** model input is anti-symmetric so predictions are naturally consistent.
3. Siamese networks: overkill for ~8k fights; we're doing GBMs.
### 4.11 Features explicitly **NOT** included
- Betting odds — excluded per your brief (only used in evaluation benchmark).
- Corner (red/blue) — would leak favorite signal; we randomize during training.
- Referee as main effect — underpowered; only include as hierarchical prior if signal emerges.
- Event attendance / PPV-ness — noise.
---
 
## 5. Modeling Approach
 
### 5.1 Primary model
**LightGBM with `objective='binary'`, metric `binary_logloss`.**
 
Why: gradient boosting consistently wins on published UFC benchmarks (65–68% accuracy, ~0.60 log-loss on proper time-splits). Beats NNs on this data size. LightGBM over XGBoost for speed and built-in categorical handling, but XGBoost is a fine substitute — we can abstract via sklearn estimator interface.
 
### 5.2 Baseline models (sanity floors)
- **Logistic regression on difference features** — interpretable baseline, ~61% accuracy.
- **Pick-the-favorite** (closing-odds-based, evaluation only) — the public-market baseline to beat.
- **Elo-only** — 58–62% accuracy; anything below this in the production model is broken.
### 5.3 Training discipline (critical)
- **Chronological walk-forward cross-validation.** Never `train_test_split(random_state=...)` — that leaks future fights. Example folds:
  - Train <2022, val 2022H1, test 2022H2
  - Train <2023, val 2023H1, test 2023H2
  - Train <2024, val 2024H1, test 2024H2
- **Symmetric-swap augmentation** applied *after* splitting so both orientations of a fight stay in the same fold.
- **Corner randomization** applied during feature extraction so corner has no residual signal.
- **No class rebalancing** (no SMOTE, no oversample) — distorts calibration. Base rate is ~55/45 red/blue after randomization ≈ 50/50.
### 5.4 Calibration
Fit on a **separate temporal calibration fold** (e.g., fights from year T−1), then evaluate on test (year T).
 
- **Isotonic regression** — preferred when calibration set ≥1k fights. UFC has ~600 fights/year, so pooling 2–3 years gives enough data.
- **Platt / sigmoid** — fallback if calibration set <500.
- **Beta calibration** — good middle ground; fewer parametric assumptions.
### 5.5 Uncertainty (the "confidence" in "win probability + confidence")
 
Two layers:
 
1. **Bootstrap ensemble:** 20 bagged LightGBM models (different seeds, sampled fight subsets). Predictive distribution → mean + std of `p(A>B)`. Gives us a natural "I'm 76% ± 5%" band.
2. **Split conformal prediction on calibrated probabilities.** Gives distribution-free coverage guarantees (e.g., "90% prediction intervals"). Simple implementation; Anderson-Sutton NCAA 2023 is the template.
Both combine cleanly — conformal intervals over the ensemble-averaged calibrated prob.
 
### 5.6 Handling method-of-victory, round, etc.
**Not in v1.** Your brief specified calibrated win probability + confidence as the target. If we later want method or round heads, add them as separate multi-class models on the same feature pipeline — don't try to joint-predict (curse of dimensionality on 8k fights).
 
---
 
## 6. Evaluation Plan
 
### 6.1 Metrics
- **Log-loss** (primary). Vegas closing-line implied-prob log-loss on UFC ≈ 0.61. Beating 0.63 would be competitive.
- **Brier score** (secondary) — diagnoses calibration vs sharpness tradeoff.
- **ROC-AUC** — ranking quality, insensitive to calibration.
- **Reliability diagram** (10 deciles) — visual calibration check.
- **Accuracy** (reported but not optimized — misleading on probability targets).
### 6.2 Benchmark against closing line
Even though we're not using odds as features, **we should scrape closing moneylines** (from BestFightOdds.com or `betmma.tips`) and compare our log-loss to Vegas implied-prob log-loss on the same held-out fights. Beating the closing line is the only honest validation.
 
**This is an open decision — see §10, Q2.**
 
### 6.3 ROI simulation (optional but recommended)
Kelly-fraction bet sizing on historical closing lines. Ties probability quality directly to dollars. If the model has EV, this shows it; if it doesn't, the headline log-loss can still look fine.
 
### 6.4 Target performance (what "good" looks like)
- Log-loss ≤ 0.63 on a rolling chronological test set
- Brier ≤ 0.22
- AUC ≥ 0.68
- Calibration: 10-decile Hosmer-Lemeshow p > 0.1 or reliability slope within [0.9, 1.1]
- ROI positive at 1% Kelly on historical closing lines (stretch)
---
 
## 7. Known Pitfalls (all observed in public UFC models)
 
The single biggest quality gap in public UFC prediction repos: they mostly get these wrong. Getting them right is the main technical discipline of the project.
 
1. **Post-fight stats leaking as pre-fight features.** UFCStats fighter pages show cumulative career stats that *include* the fight you're predicting. We must reconstruct each fighter's stat snapshot as of the fight date, using only prior fights. → Enforced by the `fighter_aso_features` table in §3.7.
2. **Random train/test splits on a time-series problem.** Same opponent on both sides. → Enforced by chronological splits in §5.3.
3. **Corner leakage.** Red corner ≈ betting favorite in UFC scheduling. A model that learns "red → win" is learning Vegas's signal. → Enforced by corner randomization in §5.3.
4. **Oversampling / class rebalancing distorts calibration.** Don't. → §5.3.
5. **Survivorship bias.** Dropping fighters with <N fights throws out rookies. → Use shrinkage toward division means instead.
6. **Retired-fighter contamination in opponent lookups.** Always use as-of dates when computing opponent's features, not their current DB snapshot.
7. **Stance/weight-class label churn.** Fighters change weight classes and stances. → `fighter_snapshots` table in §3.2.
8. **Name-based joins.** Never. → Canonical IDs everywhere.
9. **Over-interpreting small-N features.** A fighter's first UFC fight has no rolling-L5. → Shrinkage + missing-data indicators, not imputation with zero.
10. **Control time in `mm:ss` format** silently treated as numeric — scrapers have shipped this bug. → Convert to seconds at ingest.
---
 
## 8. Proposed System Architecture
 
```
┌─────────────────────┐   ┌────────────────────┐
│ Scheduled ingesters │   │ Upcoming-card      │
│ (daily / weekly /   │   │ pollers            │
│ fight-week)         │   │ UFC.com / ESPN /   │
│ Greco1899 CSV pull  │   │ Tapology           │
│ + UFCStats diff     │   └──────────┬─────────┘
│ + Sherdog DOB fill  │              │
└──────────┬──────────┘              │
           ▼                          ▼
   ┌───────────────────────────────────────┐
   │ SQLite / Postgres                     │
   │ events, fights, fight_stats_round,    │
   │ fighters, fighter_snapshots,          │
   │ upcoming_bouts, canonical_ids         │
   └──────────┬────────────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Feature builder      │
   │ fighter_aso_features │
   │ difference features  │
   │ Elo / Glicko update  │
   └──────────┬───────────┘
              │
       ┌──────┴───────┐
       ▼              ▼
  ┌──────────┐  ┌─────────────────┐
  │ Trainer  │  │ Live predictor  │
  │ chron CV │  │ batches upcoming│
  │ LightGBM │──▶ bouts → probs   │
  │ isotonic │  │ + conformal CI  │
  │ ensemble │  └────────┬────────┘
  └──────────┘           │
                         ▼
                 ┌──────────────────┐
                 │ Predictions view │
                 │ (served how?)    │
                 └──────────────────┘
```
 
**Retraining cadence:** retrain weekly after each event ingest, OR only when model drift alert fires. Cheap enough to just do weekly.
 
**Serving:** **open decision — see §10, Q4.** Options range from writing a CSV nightly → static HTML dashboard → FastAPI service → webapp.
 
**Automation:** **open decision — see §10, Q5.** Options: local cron, GitHub Actions, small cloud VM, Cloud Run.
 
---
 
## 9. Proposed Build Sequence for Claude Code
 
Once you confirm the §10 decisions, here's the recommended order for Claude Code to implement. Each step ends in something verifiable.
 
1. **Repo scaffold + deps.** `pyproject.toml` with `lightgbm`, `scikit-learn`, `pandas`, `pyarrow`, `rapidfuzz`, `requests`, `beautifulsoup4`, `sqlalchemy`, `alembic`, `pytest`. Pre-commit + ruff.
2. **DB schema.** Alembic migrations for the seven tables in §3.
3. **UFCStats ingest.** Start by vendoring `Greco1899/scrape_ufc_stats`'s CSVs (clone their repo nightly). Write CSV→DB loader. Verify row counts vs an event we hand-check.
4. **Fighter ID resolution.** Wikidata SPARQL bootstrap → fuzzy-match Sherdog → manual review of long tail → `canonical_fighter_ids` table.
5. **Sherdog DOB/camp enrichment.** Fill in what UFCStats lacks for pre-UFC fighters.
6. **As-of feature builder.** `fighter_aso_features` view or materialized table. Unit-tested with hand-verified fixtures for 5 fights.
7. **Elo + Glicko-2 rating pipelines.** Incremental update; persisted.
8. **Training pipeline.** Chronological walk-forward CV, symmetric-swap augmentation, corner randomization, LightGBM, isotonic calibration, bootstrap ensemble.
9. **Evaluation harness.** Log-loss, Brier, AUC, reliability diagram. Closing-line benchmark (only if §10 Q2 = yes).
10. **Upcoming-card poller.** UFC.com + ESPN cross-check → `upcoming_bouts` table.
11. **Live predictor.** Joins upcoming bouts to features → predictions + conformal CIs.
12. **Serving layer.** Per §10 Q4.
13. **Scheduled jobs.** Per §10 Q5.
14. **Monitoring.** Log-loss drift on recent events; alert if degrades past threshold.
Reasonable estimate: 6–10 weekends for a careful solo build, depending on serving ambition.
 
---
 
## 10. OPEN DECISIONS — need your call before Claude Code starts
 
These are the architectural forks I deliberately did *not* resolve for you. Each one meaningfully changes what Claude Code builds.
 
**Q1. Historical time scope.**
Full history (1993 → today, ~8k fights) vs modern unified-rules era (2005+, ~6.5k fights) vs just women-era-and-after (2013+, ~5k). Older fights have worse data (no per-round breakdowns before ~2001, no Pride per-round). My recommendation: **train on 2001+, evaluate on 2016+**, with a toggle to experiment with 1993–2000 as extra training data that gets automatically downweighted or excluded.
 
**Q2. Betting odds for evaluation (not features)?**
Even though we excluded odds as features, I strongly recommend **scraping closing lines for evaluation-only benchmarking**. Without it, we can't tell if the model actually beats the market. My recommendation: **yes, scrape BestFightOdds or betmma.tips closing lines, store in `fights.closing_odds_*`, never touch them at train/inference time, use them only in the evaluation notebook and ROI simulation.** Your call.
 
**Q3. Tapology: scrape it or skip it?**
Tapology gives us earlier visibility into announced cards but scraping it is **hard** (IP bans, ToS-forbidden). UFC.com + ESPN give us the same card data ~1–2 weeks later with no hassle. My recommendation: **skip Tapology in v1**, revisit if weekly lead time proves too short. Saves significant engineering effort.
 
**Q4. Serving layer: how do you want to consume predictions?**
Options:
- **(a) CSV/Parquet file** updated on schedule (cheapest; no serving).
- **(b) Static HTML dashboard** generated on schedule (e.g., MkDocs/Observable/Quarto) — nice for browsing upcoming card predictions.
- **(c) FastAPI JSON endpoint** — useful if you want to query ad-hoc or integrate elsewhere.
- **(d) Full webapp** (FastAPI + a React/Svelte front-end) — fight card views, fighter lookup, history.
My recommendation: **(b) static HTML dashboard** as the v1 deliverable — it's the best effort/value ratio and is perfectly adequate for weekly browsing. (c) is a fast follow.
 
**Q5. Automation host.**
- **(a) Local cron on your machine** — fine for research use; breaks if laptop is off.
- **(b) GitHub Actions scheduled workflows** — free, reliable, and conveniently version-controls the refresh. Best for CSV/static-dashboard serving.
- **(c) Small cloud VM (Fly.io / Render / DO $5/mo)** — needed if you want the FastAPI option.
- **(d) Cloud Run / Lambda** — overkill for this scale.
My recommendation: **(b) GitHub Actions**, assuming we go with serving option (a) or (b). If we go with (c)/(d) serving, pair with host (c).
 
**Q6. Betting workflow (out-of-scope sanity check).**
Are you planning to use this to actually bet, or is this purely a modeling exercise? That changes my recommendations on (a) whether Q2 is optional vs required, (b) whether we should add Kelly sizing and bankroll tracking to the outputs, and (c) how paranoid we need to be about data freshness on fight day. **Tell me even if it's "maybe"** — the architecture cost of leaving room for it is near zero.
 
**Q7. Corner-case features (pick any to add in v1).**
I'm proposing a feature set in §4. A few "edge" features I flagged are promising but take extra engineering. Want any of these in v1 or all v1.1?
- Opponent-adjusted rolling stats
- Multi-dimensional Elo (striking / grappling separately)
- K-means style clusters
- Age-relative-to-weight-class-peak
My recommendation: **defer all of them to v1.1**. Ship the CORE + std features first, measure, then add.
 
---
 
## 11. Risks & Unknowns
 
- **UFCStats site changes.** If they change their HTML, scrapers break. Mitigation: vendor Greco1899's CSVs as primary path; our own scraper as backup.
- **UFC.com undocumented API changes.** No SLA. Mitigation: ESPN as cross-check.
- **Tapology legal risk.** If we decide to scrape it, respect ToS and robots.txt, throttle aggressively.
- **Distribution shift.** UFC rules, fighter skill distributions, and judging tendencies drift. Mitigate with rolling windows in rating systems and periodic retraining.
- **Small sample size.** ~8k fights total is small for deep models — reinforces the LightGBM choice.
- **Women's divisions sample size** is smaller; the model may under-fit. Consider per-weight-class calibration.
- **Dataset license considerations.** UFCStats data has no clear license; Kaggle datasets each have their own. Personal/research use is the safe posture; redistribution would need review.
---
 
## 12. Sources
 
### Scraping / data source references
- [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)
- [WarrierRajeev/UFC-Predictions](https://github.com/WarrierRajeev/UFC-Predictions)
- [remypereira99/UFC-Web-Scraping](https://github.com/remypereira99/UFC-Web-Scraping)
- [DavesAnalytics/UFC-Analytics-Scraper](https://github.com/DavesAnalytics/UFC-Analytics-Scraper)
- [natebuel29/ufc-stats-scraper](https://github.com/natebuel29/ufc-stats-scraper)
- [ufcscraper on PyPI](https://pypi.org/project/ufcscraper/)
- [jackinthebox52/ufc-rest-py](https://github.com/jackinthebox52/ufc-rest-py)
- [ESPN hidden API docs (akeaswaran gist)](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b)
- [Public-ESPN-API (pseudo-r)](https://github.com/pseudo-r/Public-ESPN-API)
- [UFC events index](https://www.ufc.com/events)
- [UFCStats completed events](http://www.ufcstats.com/statistics/events/completed?page=all)
- [UFCStats upcoming events](http://ufcstats.com/statistics/events/upcoming)
- [Tapology FightCenter](https://www.tapology.com/fightcenter)
- [ehan03/Tapology-Scraper](https://github.com/ehan03/Tapology-Scraper)
- [Sherdog Fight Finder](https://www.sherdog.com/stats/fightfinder)
- [Montanaz0r/MMA-parser-for-Sherdog-and-UFC-data](https://github.com/Montanaz0r/MMA-parser-for-Sherdog-and-UFC-data)
- [ufc-api on PyPI](https://pypi.org/project/ufc-api/)
- [Wikidata Property P2818 – Sherdog fighter ID](https://www.wikidata.org/wiki/Property:P2818)
- [Wikidata Property P9728 – Tapology fighter ID](https://www.wikidata.org/wiki/Property:P9728)
### Kaggle datasets
- [Ultimate UFC Dataset (mdabbert)](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)
- [UFC 2025 Dataset (aminealibi)](https://www.kaggle.com/datasets/aminealibi/ufc-fights-fighters-and-events-dataset)
- [UFC DATASETS 1994-2025 (neelagiriaditya)](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
- [UFC Fighters' Statistics (asaniczka)](https://www.kaggle.com/datasets/asaniczka/ufc-fighters-statistics)
- [UFC-Fight historical data (rajeevw)](https://www.kaggle.com/datasets/rajeevw/ufcdata)
- [shortlikeafox/ultimate_ufc_dataset GitHub mirror](https://github.com/shortlikeafox/ultimate_ufc_dataset)
### Academic / methodology
- [Holmes et al. (2023) — Markov chain model for MMA forecasting](https://www.sciencedirect.com/science/article/pii/S0169207022000073)
- [Hitkul et al. — Comparative Study of ML for UFC Fights (Springer)](https://link.springer.com/chapter/10.1007/978-981-13-0761-4_7)
- [Johnson (2012) — Predicting MMA with novel fight-space features](https://getd.libs.uga.edu/pdfs/johnson_jeremiah_d_201208_ms.pdf)
- [Ho (2013) — Does MMA Math Work? (CS229)](https://cs229.stanford.edu/proj2013/Ho-DoesMMAMathWorkAStudyonSportsPredictionAppliedtoMixedMartialArts.pdf)
- [CS229 (2019) — Applying ML to Predict UFC Fight Outcomes](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647731.pdf)
- [AI in UFC Outcome Prediction (ACM IIP 2024)](https://dl.acm.org/doi/10.1145/3696952.3696966)
- [Anderson & Sutton (2023) — Conformal win probability for NCAA](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2283199)
- [5-Year Analysis of Age, Stature, Armspan in MMA (Tandfonline 2023)](https://www.tandfonline.com/doi/full/10.1080/02701367.2023.2252473)
- [The Southpaw Advantage (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3834302/)
### Modeling & calibration
- [scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Brier Score and Model Calibration (Neptune)](https://neptune.ai/blog/brier-score-and-model-calibration)
- [Log Loss vs Brier Score (DRatings)](https://www.dratings.com/log-loss-vs-brier-score/)
- [Gentle Intro to Conformal Prediction (arXiv 2107.07511)](https://arxiv.org/abs/2107.07511)
- [ML for sports betting (arXiv 2303.06021)](https://arxiv.org/pdf/2303.06021)
### Prediction repos (state-of-the-art public references)
- [dgonzap30/ufc-predictor](https://github.com/dgonzap30/ufc-predictor) — *only public repo with explicit chronological splits*
- [jansen88/ufc-match-predictor](https://github.com/jansen88/ufc-match-predictor)
- [naity/DeepUFC](https://github.com/naity/DeepUFC)
- [jasonchanhku/UFC-MMA-Predictor](https://github.com/jasonchanhku/UFC-MMA-Predictor)
- [jazyz/UFCFightPredictor](https://github.com/jazyz/UFCFightPredictor)
- [Turbash/Ufc-Fight-Predictor](https://github.com/Turbash/Ufc-Fight-Predictor)
### Rating systems & context
- [Cage Calculus — How our Elo model works](https://cagecalculus.com/2021/09/16/how-our-model-works/)
- [Fightomic — ranking system explained](https://fightomic.com/fightomic-ufc-ranking-system-explained/)
- [Fight Matrix FAQ on Elo limitations](https://www.fightmatrix.com/faq/)
- [Ranking MMA fighters using Elo (Pinheiro, Medium)](https://medium.com/geekculture/ranking-mma-fighters-using-the-elo-rating-system-2704adbf0c94)
- [Ranking MMA fighters with Glicko (Pinheiro, Medium)](https://medium.com/geekculture/ranking-mma-fighters-part-2-the-glicko-rating-system-1d450e0703d8)
- [Does home-cage advantage exist in MMA? (ESPN)](https://www.espn.com/mma/story/_/id/23585786/does-home-cage-advantage-exist-mma)
- [Stats and Metrics for UFC Betting — Fight Matrix](https://www.fightmatrix.com/2025/01/02/stats-and-metrics-to-use-when-betting-on-ufc-fights/)
- [Leak-free UFC model article (Medium)](https://medium.com/data-science-collective/can-data-science-predict-ufc-fights-building-a-leak-free-model-with-random-forest-4b6a1cf0945e)