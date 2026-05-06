# Screenshots — manual capture checklist

Drop your screenshots into this folder before sending the handoff to Claude Design. Aim for **PNG**, viewport ~1440×900 desktop (laptop-sized), full-quality. Filenames matter — Claude Design's prompts read better when references are named.

---

## A. "Before" — current Moirai frontend (3–5 shots)

These show what's being replaced. Claude Design uses them to understand the current information density and what *not* to ship.

Open the live dashboard (`docs/index.html` in a browser, or the GitHub Pages URL) and capture:

- [ ] `before-01-upcoming-fights.png` — the **Upcoming Fights** tab, full event card with several bouts visible.
- [ ] `before-02-fight-row-expanded.png` — one bout drilled-down (click a fight row to expand). Show the stats grid + props + bet card.
- [ ] `before-03-top-bets.png` — the **Top Bets** tab with a populated portfolio table.
- [ ] `before-04-past-events.png` — the **Past Events** tab with the predicted-vs-actual scorecard.
- [ ] `before-05-header-and-nav.png` (optional) — close-up of the header + tab nav so the existing brand treatment is unambiguous.

---

## B. References — sites with the target aesthetic (5–8 shots)

Capture the regions described, not full marketing pages. Smaller crops > whole-page screenshots.

- [ ] `ref-01-nvidia-hero.png` — **nvidia.com** homepage. The dark hero with oversized headline + product imagery. We want the typographic confidence and the disciplined accent use.
- [ ] `ref-02-linear-issues.png` — **linear.app** product screenshots (e.g. linear.app/homepage or their product page hero showing the issue list). The dense-row treatment is the closest reference for the Upcoming Fights screen.
- [ ] `ref-03-vercel-dashboard.png` — **vercel.com/dashboard** (logged-in) or screenshots from vercel.com/templates showing their project list. The way vercel handles cards, status pills, and sticky headers.
- [ ] `ref-04-stripe-docs.png` — **stripe.com/docs** or **stripe.com/payments**. The precision of their tables, code blocks, and the way they pair sans-serif UI with monospace data.
- [ ] `ref-05-anthropic-home.png` — **anthropic.com**. The restraint, the warm white text on near-black, the way a single accent moment carries the eye.
- [ ] `ref-06-vercel-analytics.png` (optional) — **vercel.com/analytics** product page — chart treatment is a strong fit for our probability/round distribution charts.
- [ ] `ref-07-bloomberg-terminal.png` (optional but valuable) — any reference shot of a Bloomberg terminal showing the dense-row + monospace data aesthetic. Search "Bloomberg terminal screenshot" or use a stock image.
- [ ] `ref-08-linear-keyboard.png` (optional) — Linear's keyboard-shortcut overlay or command palette. Reinforces the "tool for power users" vibe.

---

## C. Tips for capturing

- Use a viewport around 1440×900 — desktop-typical without being huge.
- Crop to the relevant region. Whole-page screenshots dilute the reference signal.
- Light-mode screenshots are useless for this brief. All references should be dark-mode where the site supports it.
- If a site is light-mode only (e.g. some Stripe docs pages), choose pages that still demonstrate **typographic discipline** rather than colour.
- Don't include any UI overlays (browser dev-tools, tab bars beyond the address bar).

---

## When done

The folder should look like:

```
screenshots/
  INSTRUCTIONS.md             ← this file (leave it)
  before-01-upcoming-fights.png
  before-02-fight-row-expanded.png
  before-03-top-bets.png
  before-04-past-events.png
  ref-01-nvidia-hero.png
  ref-02-linear-issues.png
  ref-03-vercel-dashboard.png
  ref-04-stripe-docs.png
  ref-05-anthropic-home.png
  ...
```

Then drag the entire `claude-design-handoff/` folder onto Claude Design with the contents of `design-brief.md` as the prompt.
