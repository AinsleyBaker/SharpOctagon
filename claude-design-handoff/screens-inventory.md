# Screens Inventory

Four screens. Build in this order — visual language is set by screen 1, the rest inherit it.

---

## 1. Upcoming Fights (home) — must-have

**Purpose.** Default landing view. The user opens Moirai on a Friday or Saturday to see "what's on this weekend, who does the model favour, where's the edge." It must read at a glance and reward a closer look.

**Data displayed** (per-bout shape lives in `data-shapes.md` §1):
- Event header: name, date, venue, "Main Card / Prelims / Early Prelims" segments, model version + last-refresh timestamp.
- For each bout: both fighters (name, record, nationality, image), weight class, title/main-event flags, model's win-probability for each, the **headline method** + round, the **confidence interval** (ci_90_lo–ci_90_hi), a confidence tier ("High / Medium / Low"), and a `value_bet_count` badge if the bet engine flagged anything.
- Bouts where `prob_a_wins == 0.5` are **preview-only** — render distinctly (greyed, "no model" tag), not as a 50/50 chart.

**Key user actions.**
- Click a fight row → expand inline OR navigate to Fight Detail (designer's call; current dashboard expands inline).
- Filter: by event (when multiple events on one card), by "fights with edge only".
- Search fighters by name.
- Switch tabs to Top Bets / Past Events.

**Wow moment — the slow-sliding hero bar (committed).**
The screen opens with a slow horizontally auto-scrolling bar of featured fight cards, sitting above the dense list. Premium ticker, not sportsbook carousel. Specifics:
- 5–8 cards visible across the row, partially clipped at the edges.
- Auto-scrolls left at ~30–45s per card-width. Linear easing, no snap.
- Pauses on hover and on focus. Seamless wrap (clone + offset).
- Each card is more spacious than the dense rows below: portraits, oversized monospace win-prob, headline method, event date.
- Edge fades on left/right (background colour, not a heavy gradient).
- The amber accent appears on at most one slider card in view at a time — typically the highest-edge bout.
- Respects `prefers-reduced-motion`: falls back to a static row.

**Other motion candidates (subordinate to the slider).**
- A horizontal probability bar that animates from 50/50 to the predicted split on viewport entry, with percentages counting up.
- The next-up event card in the dense list has slightly more breathing room than later events, with a quiet sticky countdown timer in the sidebar.

**Density target.** Roughly 8–12 visible bouts on a desktop viewport without scrolling — Linear's issue list, not a marketing site. But each row breathes; this isn't a spreadsheet.

---

## 2. Fight Detail — must-have

**Purpose.** Drill-down for one bout. The user wants to understand *why* the model favours a fighter, what method it expects, and whether there's a bet worth taking.

**Data displayed.**
- Hero: both fighters side-by-side (image, name, record, nationality, age, stance, height/reach, UFC style).
- The headline win-probability split with confidence interval.
- A method breakdown (KO/TKO, Submission, Decision per fighter — six bars or a stacked treatment).
- A round-by-round finish distribution chart (R1–R5, plus "goes to decision").
- Top 5–10 model **drivers**: feature, magnitude, which fighter it favours.
- The bet card: list of `bet_analysis` rows, sortable by EV%. Show `our_prob` vs. `implied_prob` (SportsBet), `edge`, `kelly_pct`, and the **historical-ROI badge** (`historical_roi_pct` over `historical_n_bets` from the per-market backtest).
- A "data quality" pill — `good / limited / sparse` — driven by sample size.

**Key user actions.**
- Back to fight list.
- Sort/filter the bet table.
- Toggle: show only `is_value: true` bets vs. all markets.
- Copy bet to clipboard / mark as taken (nice-to-have).

**Wow moment.** The probability ring/bar transition from the list view into a larger, fuller treatment that animates the CI band into place.

---

## 3. Top Bets — must-have

**Purpose.** Cross-event portfolio view. "Across this whole weekend, where should my money go?" Shaped by Kelly fractional sizing on a user-set bankroll.

**Data displayed** (shape in `data-shapes.md` §3):
- Bankroll input (number, with persisted value).
- Strategy toggle: Quarter-Kelly / Half-Kelly / Full-Kelly.
- Portfolio summary: total stake, expected return, expected EV%, n_bets, events covered.
- Bets table: event, fight, bet_type, description, our_prob, sb_odds, edge, kelly_pct, stake, expected_return, historical_roi_pct (n_bets).
- Filter: bet_type (moneyline / method / method_combo / winning_round / distance / total_rounds / round_survival).
- Filter: jump-to-event (when the user clicks an event from screen 1 and wants only that event's bets).

**Key user actions.**
- Edit bankroll → all stake values recompute.
- Switch strategy → Kelly fractions and stakes recompute.
- Filter to one event / one bet type.
- Click a bet row → open the relevant Fight Detail.

**Empirical-ROI gate (important).** Only bet types with positive historical ROI in their edge bucket appear with `is_value: true`. Moneyline is never marked `is_value` (Vegas beats us on moneyline). The UI should make this gating *visible* — e.g., a small "backtested" pill or the historical ROI rendered next to the EV%.

---

## 4. Past Events — nice-to-have (but design language must extend cleanly)

**Purpose.** Credibility check. "Has this thing actually been right?" Shows completed events, pred-vs-actual for each bout, and rolling model-quality metrics.

**Data displayed** (shape in `data-shapes.md` §4):
- Per-event card with each completed bout: fighters, our pick + probability, actual winner, method/round, hit/miss indicator (no neon green/red — see anti-patterns).
- Per-event scorecard: accuracy, mean Brier, mean log-loss, value-bets evaluated, realised EV per bet.
- A small rolling-metrics sparkline (last 6–12 events): accuracy and log-loss trend.

**Key user actions.**
- Expand event to see all bouts.
- Toggle: "model picks only" vs. "value bets only" view.

**Wow moment.** A single quiet hero stat at the top — e.g., "67% accuracy across 412 graded bouts since v1" — set in oversized monospace, with a subtle sparkline behind it.

---

## Cross-screen requirements

- **Sticky header** with current event name, model version, and a quick-jump nav (Upcoming / Top Bets / Past). Linear-style.
- **Search** is global — it filters the current screen.
- **Empty / loading / preview-only** states must be designed (not generic spinners).
- The accent (electric amber) appears at most twice per visible viewport: typically the highest-confidence pick and the primary CTA. Everything else is neutral.
- **No flag emojis.** If nationality matters, render as a small monospace ISO code (e.g., `AUS`, `BRA`) in muted grey, not a flag.
