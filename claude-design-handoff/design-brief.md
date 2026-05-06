# Moirai — Design Brief (paste into Claude Design)

I'm redesigning the frontend of **Moirai** (site: **moirai.gg**), a UFC fight prediction system that produces calibrated win probabilities, predicted method (KO/TKO, Submission, Decision), per-round finish distributions, confidence intervals, and edge-vs-bookmaker analysis for every bout on every upcoming UFC card.

The current frontend is a static HTML dashboard with a "terminal/Matrix" aesthetic — gradients, neon green, red/blue corners, flag emojis. I'm replacing it. Attached files: `sample-data.json` (a full event payload), `data-shapes.md` (per-screen JSON shapes), `screens-inventory.md` (every screen broken down). Reference screenshots and "before" shots of the current dashboard are in `screenshots/` (filenames prefixed `before-` show what's being replaced; `ref-` show the target aesthetic).

## Feeling

**Bloomberg terminal for fight nerds**, with the production polish of NVIDIA, Linear, Vercel, Stripe, and Anthropic. Technical and intelligent, never gambling-flashy. Confident enough that a stat geek leans in, restrained enough that it doesn't look like FanDuel.

## Visual direction

**Background:** near-black `#0A0A0B`. Elevated card surfaces `#141416`. Subtle 1px hairlines on borders, no heavy shadows.

**Accent:** electric amber `#FFB300`. Used sparingly — at most twice per visible viewport. Reserve it for: the highest-confidence win probability number, the primary CTA, and one moment per screen (e.g., a sparkline endpoint, a value-bet indicator).

**Text:** warm white `#F5F2EC` for primary, muted grey `#8A8A92` for secondary, dim grey `#4A4A52` for tertiary/labels.

**No** neon green or neon red for win probabilities. Win/loss states use neutral typography weight + the amber accent for highlight, not colour-coding.

## Typography

- UI: **Inter** (or system sans) — tight tracking, confident headlines.
- Numbers, probabilities, stats, monospace tables: **JetBrains Mono**.
- Hierarchy: oversized hero numbers (e.g., `73.8%` win probability set in 64–96px monospace), confident H1s, generous body line-height.

## Motion

- Probability numbers count up when they enter the viewport (~600ms ease-out).
- Probability bars/rings animate from 50/50 baseline to predicted split on load.
- Cards lift 2–4px on hover with a subtle amber inner-edge glow (no thick borders).
- Fade-and-rise on scroll for cards and chart elements.
- Smooth scroll. Sticky header with a subtle backdrop blur.
- One "wow" moment per screen — a hero stat, an animated chart, an interactive comparison.

## Anti-patterns to avoid

- Gambling-app gradients (purple-to-pink, green-to-yellow, anything that looks like a slot machine).
- Glassmorphism overkill — at most one subtle blurred surface per screen.
- Octagon graphics, cage textures, or chain-link motifs in backgrounds.
- Neon green / neon red for probabilities or pred-vs-actual results.
- Flag emojis next to fighter names. (If nationality matters, use a small monospace ISO code in muted grey.)
- Generic Bootstrap-looking cards with rounded `12px` borders and grey backgrounds.
- Score-ticker / sportsbook-style red and green pills.

## Reference

Linear's issue list crossed with a Bloomberg terminal: dense but elegant, every row carries information, nothing is decoration. NVIDIA-level confidence with the data-density of Bloomberg and the restraint of Anthropic.

## First screen to build

**Upcoming Fights overview.** A list view of fights for one or two events. Each row shows both fighters (name, record), the predicted winner with probability, a one-line method breakdown, a confidence indicator, and an edge/value badge if relevant. Tappable for drill-down. Use the attached `sample-data.json` — render a real card with real fighters (Chimaev vs. Strickland headlines).

### Hero treatment — slow-sliding fight-card bar

Above the dense bout list, the screen opens with a **slow horizontally auto-scrolling bar of upcoming fight cards** (think a quiet, premium ticker — not a sportsbook carousel). Behaviour:

- One card per featured bout (5–8 visible across the row, partially clipped at the edges so the motion reads as continuous).
- Auto-scrolls left at a **slow, deliberate pace** (~30–45s for one card-width), pauses on hover and on focus. No abrupt transitions — linear easing, no snap.
- Cards are larger and more spacious than the dense list rows below: fighter portraits, win-probability hero number in monospace, headline method, event date.
- Wraps seamlessly (clone + offset, not a jarring loop).
- Respects `prefers-reduced-motion` — falls back to a static row when set.
- Subtle edge fades on the left/right (background colour, not a heavy gradient) so cards dissolve in/out rather than hard-clipping.
- The amber accent appears on **at most one card** in view at a time — typically the bout flagged with the highest model edge.

The dense bout list lives below the slider and is the workhorse view. The slider is the "wow" moment for this screen.

For the first version, give me:
1. The full Upcoming Fights screen (slow-sliding hero bar + event header + bout rows + sticky nav).
2. Three variations of just the bout-row component (in the dense list), showing how density vs. spaciousness trade off.
3. The default expanded state (one row drilled down inline) so I can see how detail unfolds.

Once I sign off on the visual language, I'll ask for Fight Detail, Top Bets, and Past Events — same typography, same accent discipline, same density.

## Data quirks worth knowing

- Some bouts have `prob_a_wins == 0.5` exactly. That's a sentinel: model didn't run. Render as a "preview only" state — not a 50/50 bar.
- `confidence_tier` is `"High" | "Medium" | "Low"` — it's the headline interpretation of `uncertainty_std`. The CI (`ci_90_lo`, `ci_90_hi`) tells you how wide the band is.
- `value_bet_count` and `has_edge` flag bouts where the bet engine found something. The engine is empirically gated — moneyline is never `is_value` in this codebase (the model loses to Vegas on win/loss bets, but beats it on method/round/distance markets).
- `drivers` is the model's top features for that bout. It's the closest thing to "explainability" — each driver has an impact magnitude and which fighter it favours.

## What success looks like

A frontend that someone who follows fights for a living would want to keep open. Not flashy, not generic. The user opens it on Saturday morning and the first impression is: *this is built by someone who actually understands the sport and respects the data*.
