"""
Stage 12 — Static HTML dashboard generator.

Reads data/predictions.json and renders a fight-card dashboard to docs/index.html.
Designed to be committed to GitHub Pages (branch deploy from /docs) or any static host.
Regenerated after every prediction run.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

log = logging.getLogger(__name__)

PREDICTIONS_PATH  = Path("data/predictions.json")
FIGHTER_IMGS_PATH = Path("data/fighter_images.json")
TEMPLATES_DIR     = Path(__file__).parent / "templates"
OUTPUT_DIR        = Path("docs")


def load_predictions() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)


def load_fighter_images() -> dict[str, str]:
    if not FIGHTER_IMGS_PATH.exists():
        return {}
    try:
        return json.loads(FIGHTER_IMGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _group_by_event(predictions: list[dict]) -> list[dict]:
    events: dict[str, dict] = {}
    for p in predictions:
        key = f"{p.get('event_date')}|{p.get('event_name', '')}"
        if key not in events:
            events[key] = {
                "event_name": p.get("event_name", "Unknown Event"),
                "event_date": p.get("event_date"),
                "bouts": [],
            }
        events[key]["bouts"].append(p)
    return sorted(events.values(), key=lambda e: e["event_date"] or "")


def _format_prob(p) -> str:
    if p is None or (isinstance(p, float) and p != p):  # NaN check
        return "—"
    return f"{float(p)*100:.1f}%"


def _format_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return ""
    return f"{float(lo)*100:.0f}–{float(hi)*100:.0f}%"


def _confidence_tier(std) -> str:
    if std is None or (isinstance(std, float) and std != std):
        return "—"
    s = float(std)
    if s < 0.07:
        return "High"
    if s < 0.12:
        return "Medium"
    return "Low"


def _data_quality(a_n, b_n) -> str:
    n = min(int(a_n or 0), int(b_n or 0))
    if n >= 10:
        return "good"
    if n >= 5:
        return "limited"
    return "sparse"


def _streak_display(win_s, loss_s) -> str:
    win_s  = int(win_s  or 0)
    loss_s = int(loss_s or 0)
    if win_s > 0:
        return f"{win_s}W"
    if loss_s > 0:
        return f"{loss_s}L"
    return "—"


def _kelly_display(frac) -> str:
    if frac is None or (isinstance(frac, float) and frac != frac) or frac == 0:
        return "—"
    return f"{float(frac)*100:.1f}% of bankroll"


# Country name → ISO 3166-1 alpha-2 for flag emoji
_COUNTRY_ISO: dict[str, str] = {
    "Australia": "AU", "Brazil": "BR", "United States": "US", "USA": "US",
    "Russia": "RU", "Georgia": "GE", "England": "GB", "United Kingdom": "GB",
    "Ireland": "IE", "Mexico": "MX", "China": "CN", "Canada": "CA",
    "New Zealand": "NZ", "Poland": "PL", "Netherlands": "NL", "Sweden": "SE",
    "Nigeria": "NG", "France": "FR", "Italy": "IT", "Spain": "ES",
    "Japan": "JP", "South Korea": "KR", "South Africa": "ZA", "Jamaica": "JM",
    "Kazakhstan": "KZ", "Kyrgyzstan": "KG", "Uzbekistan": "UZ", "Ukraine": "UA",
    "Azerbaijan": "AZ", "Armenia": "AM", "Cameroon": "CM", "DR Congo": "CD",
    "Senegal": "SN", "Morocco": "MA", "Ghana": "GH", "Kenya": "KE",
    "Argentina": "AR", "Colombia": "CO", "Venezuela": "VE", "Peru": "PE",
    "Chile": "CL", "Ecuador": "EC", "Bolivia": "BO", "Paraguay": "PY",
    "Cuba": "CU", "Dominican Republic": "DO", "Puerto Rico": "PR",
    "Germany": "DE", "Austria": "AT", "Switzerland": "CH", "Belgium": "BE",
    "Portugal": "PT", "Czech Republic": "CZ", "Slovakia": "SK", "Hungary": "HU",
    "Romania": "RO", "Bulgaria": "BG", "Serbia": "RS", "Croatia": "HR",
    "Slovenia": "SI", "Bosnia and Herzegovina": "BA", "Albania": "AL",
    "Greece": "GR", "Turkey": "TR", "Israel": "IL", "Iran": "IR",
    "Saudi Arabia": "SA", "UAE": "AE", "Philippines": "PH", "Thailand": "TH",
    "Indonesia": "ID", "Malaysia": "MY", "Vietnam": "VN", "Myanmar": "MM",
    "Scotland": "GB-SCT", "Wales": "GB-WLS", "Northern Ireland": "GB-NIR",
}


def _flag_emoji(country: str | None) -> str:
    if not country:
        return ""
    iso = _COUNTRY_ISO.get(country, "")
    if not iso or len(iso) != 2:
        return ""
    # Convert ISO to regional indicator symbols (Unicode flag emoji)
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso.upper())


def _stat_colors(a_val, b_val, higher_is_better: bool = True) -> tuple[str, str]:
    """Return (class_for_a, class_for_b) — 'stat-better', 'stat-worse', or ''."""
    try:
        a, b = float(a_val or 0), float(b_val or 0)
    except (TypeError, ValueError):
        return "", ""
    if abs(a - b) < 0.001:
        return "", ""
    if higher_is_better:
        return ("stat-better", "stat-worse") if a > b else ("stat-worse", "stat-better")
    return ("stat-better", "stat-worse") if a < b else ("stat-worse", "stat-better")


def _fighter_type(ko_rate, sub_rate, td_per_min, slpm, sig_acc) -> str:
    """Classify fighting style from computed stats."""
    ko  = float(ko_rate  or 0)
    sub = float(sub_rate or 0)
    td  = float(td_per_min or 0)
    sp  = float(slpm or 0)
    acc = float(sig_acc or 0)

    if ko > 0.30:
        return "KO Artist"
    if sub > 0.30:
        return "Submission"
    if td > 2.5 and sub > 0.10:
        return "Wrestler"
    if td > 1.5:
        return "Grappler"
    if sp > 5.5 or (acc > 0.48 and sp > 4.0):
        return "Striker"
    if ko > 0.15 and sp > 4.0:
        return "Power Striker"
    return "Complete"


def _enrich_props(bout: dict) -> None:
    """Add formatted prop probability strings to a bout dict."""
    props = bout.get("props") or {}
    if not props:
        bout["has_props"] = False
        return

    bout["has_props"] = True
    fa = bout.get("fighter_a_name", "A")
    fb = bout.get("fighter_b_name", "B")

    def pct(key: str) -> str:
        v = props.get(key)
        return f"{float(v)*100:.1f}%" if v is not None else "—"

    bout["prop_rows"] = [
        {"label": f"{fa} wins by KO/TKO",  "prob": pct("prob_a_wins_ko_tko")},
        {"label": f"{fa} wins by Submission", "prob": pct("prob_a_wins_sub")},
        {"label": f"{fa} wins by Decision",  "prob": pct("prob_a_wins_dec")},
        {"label": f"{fb} wins by KO/TKO",  "prob": pct("prob_b_wins_ko_tko")},
        {"label": f"{fb} wins by Submission", "prob": pct("prob_b_wins_sub")},
        {"label": f"{fb} wins by Decision",  "prob": pct("prob_b_wins_dec")},
        {"label": "Goes to Decision",       "prob": pct("prob_decision")},
    ]

    # prob_rounds values are P(finish in Rx) — they sum to prob_finish, not 1.
    # Show finish-round rows then a separate "Goes to Decision" row so it's clear
    # that "Ends in Round 3" means a KO/Sub in R3, NOT a decision.
    prob_rounds = props.get("prob_rounds") or {}
    prob_dec = props.get("prob_decision", 0)
    bout["round_rows"] = [
        {"label": f"Finish in Round {i}", "prob": f"{float(prob_rounds.get(f'R{i}', 0))*100:.1f}%"}
        for i in range(1, 6)
        if prob_rounds.get(f"R{i}", 0) > 0.005
    ]
    if prob_dec and prob_dec > 0.005:
        bout["round_rows"].append(
            {"label": "Goes to Decision", "prob": f"{float(prob_dec)*100:.1f}%"}
        )


def _enrich_bet_analysis(bout: dict) -> None:
    """Format bet_analysis list for display — top 10 value bets only."""
    bets = bout.get("bet_analysis") or []
    # Take value bets first, then best EV — cap at 10
    value_bets = [b for b in bets if b.get("is_value")][:10]

    formatted = []
    for bet in value_bets:
        ev       = float(bet.get("ev_pct", 0))
        kelly_p  = float(bet.get("kelly_pct", 0))
        odds     = float(bet.get("sb_odds", 1.0))
        our_p    = float(bet.get("our_prob", 0))
        # Est. profit: for every $100 wagered, expected net gain
        est_profit = round((odds - 1) * 100)
        formatted.append({
            "description": bet.get("description", ""),
            "our_prob":    f"{our_p * 100:.0f}%",
            "sb_odds":     f"{odds:.2f}",
            "est_profit":  f"+${est_profit}",
            "stake_pct":   f"{kelly_p:.1f}%" if kelly_p > 0 else "<1%",
            "is_value":    True,
            "ev_raw":      ev,
        })
    bout["bet_rows"] = formatted
    bout["has_bets"] = bool(formatted)
    bout["value_bet_count"] = len(formatted)


def build(output_dir: Path = OUTPUT_DIR) -> None:
    predictions = load_predictions()
    if not predictions:
        log.warning("No predictions found at %s", PREDICTIONS_PATH)

    fighter_imgs = load_fighter_images()
    events = _group_by_event(predictions)

    # Enrich each bout for display
    for event in events:
        for bout in event["bouts"]:
            bout["prob_a_pct"] = _format_prob(bout.get("prob_a_wins"))
            bout["prob_b_pct"] = _format_prob(bout.get("prob_b_wins"))

            lo_key = next((k for k in bout if k.startswith("ci_") and k.endswith("_lo")), None)
            hi_key = next((k for k in bout if k.startswith("ci_") and k.endswith("_hi")), None)
            bout["ci_display"] = _format_ci(
                bout.get(lo_key), bout.get(hi_key)
            ) if lo_key and hi_key else ""

            bout["kelly_display"] = _kelly_display(bout.get("kelly_fraction"))
            bout["std_pct"] = _format_prob(bout.get("uncertainty_std"))

            # Determine favourite
            pa = bout.get("prob_a_wins") or 0.5
            if abs(pa - 0.5) < 0.03:
                bout["favourite"] = "toss-up"
            elif pa > 0.5:
                bout["favourite"] = "a"
            else:
                bout["favourite"] = "b"

            # Confidence tier and data quality
            bout["confidence_tier"] = _confidence_tier(bout.get("uncertainty_std"))
            bout["data_quality"] = _data_quality(
                bout.get("a_n_fights"), bout.get("b_n_fights")
            )
            bout["a_streak"] = _streak_display(
                bout.get("a_win_streak"), bout.get("a_loss_streak")
            )
            bout["b_streak"] = _streak_display(
                bout.get("b_win_streak"), bout.get("b_loss_streak")
            )

            def _nan_pct(v, decimals=0):
                if v is None or (isinstance(v, float) and v != v):
                    return "—"
                return f"{float(v)*100:.{decimals}f}%"

            def _nan_f(v, fmt=".1f"):
                if v is None or (isinstance(v, float) and v != v):
                    return "—"
                return format(float(v), fmt)

            bout["a_l3"] = _nan_pct(bout.get("a_l3_win_rate"))
            bout["b_l3"] = _nan_pct(bout.get("b_l3_win_rate"))
            bout["a_n_fights"] = int(bout.get("a_n_fights") or 0)
            bout["b_n_fights"] = int(bout.get("b_n_fights") or 0)
            bout["a_slpm"]        = _nan_f(bout.get("a_slpm"))
            bout["b_slpm"]        = _nan_f(bout.get("b_slpm"))
            bout["a_sapm"]        = _nan_f(bout.get("a_sapm"))
            bout["b_sapm"]        = _nan_f(bout.get("b_sapm"))
            bout["a_finish_rate"] = _nan_pct(bout.get("a_finish_rate"))
            bout["b_finish_rate"] = _nan_pct(bout.get("b_finish_rate"))
            bout["a_ko_rate"]     = _nan_pct(bout.get("a_ko_rate"))
            bout["b_ko_rate"]     = _nan_pct(bout.get("b_ko_rate"))
            bout["a_sub_rate"]    = _nan_pct(bout.get("a_sub_rate"))
            bout["b_sub_rate"]    = _nan_pct(bout.get("b_sub_rate"))
            bout["a_td_per_min"]  = _nan_f(bout.get("a_td_per_min"), ".2f")
            bout["b_td_per_min"]  = _nan_f(bout.get("b_td_per_min"), ".2f")

            # Country flags
            a_nat = bout.get("fighter_a_nationality") or ""
            b_nat = bout.get("fighter_b_nationality") or ""
            bout["a_flag"] = _flag_emoji(a_nat)
            bout["b_flag"] = _flag_emoji(b_nat)
            bout["a_nationality"] = a_nat
            bout["b_nationality"] = b_nat

            # Fighter type — use raw numeric values before they're formatted below
            bout["a_fighter_type"] = _fighter_type(
                bout.get("a_ko_rate") if not isinstance(bout.get("a_ko_rate"), str) else None,
                bout.get("a_sub_rate") if not isinstance(bout.get("a_sub_rate"), str) else None,
                bout.get("a_td_per_min") if not isinstance(bout.get("a_td_per_min"), str) else None,
                bout.get("a_slpm") if not isinstance(bout.get("a_slpm"), str) else None,
                bout.get("a_sig_acc") if not isinstance(bout.get("a_sig_acc"), str) else None,
            )
            bout["b_fighter_type"] = _fighter_type(
                bout.get("b_ko_rate") if not isinstance(bout.get("b_ko_rate"), str) else None,
                bout.get("b_sub_rate") if not isinstance(bout.get("b_sub_rate"), str) else None,
                bout.get("b_td_per_min") if not isinstance(bout.get("b_td_per_min"), str) else None,
                bout.get("b_slpm") if not isinstance(bout.get("b_slpm"), str) else None,
                bout.get("b_sig_acc") if not isinstance(bout.get("b_sig_acc"), str) else None,
            )

            # Initials for avatar fallback
            def _initials(name: str) -> str:
                parts = (name or "").split()
                return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else (name[:2].upper() or "?")
            bout["a_initials"] = _initials(bout.get("fighter_a_name", ""))
            bout["b_initials"] = _initials(bout.get("fighter_b_name", ""))

            # Fighter images from cache
            bout["a_img"] = fighter_imgs.get(bout.get("fighter_a_name", ""), "")
            bout["b_img"] = fighter_imgs.get(bout.get("fighter_b_name", ""), "")

            # Stance from DB (factual UFC data)
            bout["a_stance"] = (bout.get("fighter_a_stance") or "").replace("Switch", "Switch Stance")
            bout["b_stance"] = (bout.get("fighter_b_stance") or "").replace("Switch", "Switch Stance")

            # Fighter style badge — only if data quality is good enough
            n_min = min(bout.get("a_n_fights") or 0, bout.get("b_n_fights") or 0)
            bout["show_fighter_type"] = n_min >= 5
            bout["a_fighter_type"] = bout.get("a_fighter_type", "")
            bout["b_fighter_type"] = bout.get("b_fighter_type", "")

            # Stat color classes for the comparison table (raw values before formatting)
            _raw = lambda k: bout.get(k)  # raw value from JSON (still numeric here)
            bout["col_streak"]   = _stat_colors(bout.get("a_win_streak"), bout.get("b_win_streak"))
            bout["col_l3"]       = _stat_colors(bout.get("a_l3_win_rate"), bout.get("b_l3_win_rate"))
            bout["col_finish"]   = _stat_colors(bout.get("a_finish_rate"), bout.get("b_finish_rate"))
            bout["col_ko"]       = _stat_colors(bout.get("a_ko_rate"), bout.get("b_ko_rate"))
            bout["col_sub"]      = _stat_colors(bout.get("a_sub_rate"), bout.get("b_sub_rate"))
            bout["col_slpm"]     = _stat_colors(bout.get("a_slpm"), bout.get("b_slpm"))
            bout["col_sapm"]     = _stat_colors(bout.get("a_sapm"), bout.get("b_sapm"), higher_is_better=False)
            bout["col_td"]       = _stat_colors(bout.get("a_td_per_min"), bout.get("b_td_per_min"))

            # Prop probabilities
            _enrich_props(bout)
            # EV / Kelly bet analysis
            _enrich_bet_analysis(bout)

            # SportsBet moneyline odds for display
            sb = bout.get("sportsbet_odds") or {}
            bout["sb_odds_a"] = f"{sb['moneyline_a']:.2f}" if sb.get("moneyline_a") else None
            bout["sb_odds_b"] = f"{sb['moneyline_b']:.2f}" if sb.get("moneyline_b") else None

    from ufc_predict.eval.bet_analysis import top_value_bets
    top_bets = top_value_bets(predictions, n=30)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("dashboard.html")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = template.render(
        events=events,
        top_bets=top_bets,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_bouts=len(predictions),
    )

    out_path = output_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    log.info("Dashboard written to %s (%d bouts, %d events)", out_path, len(predictions), len(events))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
