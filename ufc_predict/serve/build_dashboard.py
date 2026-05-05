"""
Stage 12 — Static HTML dashboard generator.

Reads data/predictions.json and renders a fight-card dashboard to docs/index.html.
Designed to be committed to GitHub Pages (branch deploy from /docs) or any static host.
Regenerated after every prediction run.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def _norm_name(s: str | None) -> str:
    """
    Normalize a fighter name for cross-source matching. ESPN preserves
    diacritics ("Mateusz Rębecki", "Joel Álvarez") but predictions.json
    strips them — without normalization, sched_order lookups miss those
    bouts and dump them at the end of the card.
    """
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.strip().lower()


log = logging.getLogger(__name__)

PREDICTIONS_PATH  = Path("data/predictions.json")
PAST_EVENTS_PATH  = Path("data/past_events.json")
FIGHTER_IMGS_PATH = Path("data/fighter_images.json")
FIGHTER_META_PATH = Path("data/fighter_metadata.json")
SCHEDULE_PATH     = Path("data/upcoming_schedule.json")
TEMPLATES_DIR     = Path(__file__).parent / "templates"
OUTPUT_DIR        = Path("docs")


def load_predictions() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH) as f:
        preds = json.load(f)
    # Re-run bet analysis on every load. predictions.json may be stale w.r.t.
    # the current bet_analysis logic (e.g. new fields like outcome_keys); the
    # raw inputs (sportsbet_odds, props, fighter probs) are still authoritative,
    # so recomputing keeps the dashboard in lockstep with the code.
    from ufc_predict.eval.bet_analysis import analyze_all_fights
    return analyze_all_fights(preds)


def load_fighter_images() -> dict[str, str]:
    if not FIGHTER_IMGS_PATH.exists():
        return {}
    try:
        raw = json.loads(FIGHTER_IMGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    # Mirror the fighter_metadata behaviour: alias every entry under its
    # normalised name so unaccented predictions hit accented image entries.
    aliased = dict(raw)
    for k, v in raw.items():
        if not v:
            continue
        nkey = _norm_meta_key(k)
        if nkey and nkey not in aliased:
            aliased[nkey] = v
    return aliased


def load_schedule() -> list[dict]:
    if not SCHEDULE_PATH.exists():
        return []
    try:
        return json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _norm_meta_key(s: str | None) -> str:
    """Aggressive name normalisation for cross-source metadata lookups.

    Strips diacritics AND punctuation so "Joel Álvarez", "Joel Alvarez",
    and "Waldo Cortes-Acosta" / "Waldo Cortes Acosta" all collapse onto
    the same key. Without this, the predictions side (which carries the
    DB's unaccented name) misses metadata records keyed under the
    UFC.com-scraped accented name and the dashboard renders without
    flag/image/style.
    """
    import re
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    stripped = re.sub(r"[^a-z0-9]+", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def load_fighter_metadata() -> dict[str, dict]:
    """Country, official UFC fighting style, image — keyed by fighter name.

    The on-disk JSON sometimes contains *duplicate logical entries* for the
    same fighter under accented vs. unaccented spellings (the scraper
    re-keys when UFC.com updates a profile). One copy may be richly
    populated (Joel Álvarez → Spain, Striker, png) while the other is a
    near-empty stub (Joel Alvarez → "", "", jpg). We merge such
    duplicates here, preferring non-empty fields, then expose the result
    under BOTH the original keys and a normalised-name alias so callers
    that look up either spelling get the merged best record.
    """
    if not FIGHTER_META_PATH.exists():
        return {}
    try:
        raw = json.loads(FIGHTER_META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    # Group records by normalised key so duplicates can be merged.
    groups: dict[str, list[tuple[str, dict]]] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        groups.setdefault(_norm_meta_key(k), []).append((k, v))

    merged: dict[str, dict] = {}
    for nkey, items in groups.items():
        if not nkey:
            continue
        # Field-wise merge: first non-empty wins. Iterate in reverse-length
        # order so the most-populated record's nested dicts (stats) win
        # ties when both sides have a value.
        items_sorted = sorted(
            items,
            key=lambda kv: sum(1 for x in kv[1].values() if x),
            reverse=True,
        )
        rec: dict = {}
        for _orig_name, src in items_sorted:
            for fld, val in src.items():
                if val in (None, "", {}, []):
                    continue
                if fld not in rec or rec[fld] in (None, "", {}, []):
                    rec[fld] = val
        # Expose under every original key in this group …
        for orig_name, _src in items:
            merged[orig_name] = rec
        # … and under the normalised alias so unaccented/punctuation-
        # stripped lookups also resolve.
        merged[nkey] = rec

    return merged


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


def _format_fight_time_aest(start_time, event_date: str) -> str:
    """
    Format fight start time in AEST. SportsBet provides Unix timestamps
    OR ISO strings; ESPN provides ISO. Returns short display like
    "Sat 9:00 AM AEST" for fights this week, or "Sat 3 May" for further out.
    """
    if not start_time:
        return ""
    try:
        from datetime import datetime, timedelta
        # Parse: int (Unix sec/ms) or ISO string
        if isinstance(start_time, (int, float)):
            ts = int(start_time)
            if ts > 1e12:  # milliseconds
                ts = ts // 1000
            dt_utc = datetime.fromtimestamp(ts, tz=UTC)
        else:
            s = str(start_time).rstrip("Z")
            if s.endswith("+0000"):
                s = s[:-5]
            dt_utc = datetime.fromisoformat(s).replace(tzinfo=UTC) if "T" in s else None
            if dt_utc is None:
                return ""
        # Convert to AEST (UTC+10, no DST for simplicity — Sydney is mostly +10/+11)
        aest = dt_utc + timedelta(hours=10)
        # Cross-platform 12-hour format (no leading zero)
        hour = aest.hour % 12 or 12
        ampm = "AM" if aest.hour < 12 else "PM"
        weekday = aest.strftime("%a")
        days_out = (aest.date() - date.today()).days
        # Always include AEST so users know the timezone
        if -2 <= days_out <= 7:
            return f"{weekday} {hour}:{aest.minute:02d} {ampm} AEST"
        return f"{weekday} {aest.day} {aest.strftime('%b')} · {hour}:{aest.minute:02d} {ampm} AEST"
    except Exception:
        return ""


def _streak_display(win_s, loss_s) -> tuple[str, str]:
    """Return (label, css_class). Plain English: '3-fight win streak'."""
    win_s  = int(win_s  or 0)
    loss_s = int(loss_s or 0)
    if win_s >= 1:
        return (f"{win_s}-fight win streak" if win_s > 1 else "1 win", "streak-w")
    if loss_s >= 1:
        return (f"{loss_s}-fight loss streak" if loss_s > 1 else "1 loss", "streak-l")
    return ("", "")


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
    """
    Return a URL to a flag image (24x18 PNG) for the given country name.

    Switched from Unicode regional-indicator characters because Windows browsers
    don't ship flag glyphs in their default font set — those characters render
    as bare two-letter ASCII (e.g. "AU"). flagcdn.com serves stable flag PNGs
    by ISO 3166-1 alpha-2 code; subdivisions (Scotland etc.) fall back to GB.
    Function name kept for backward compat — many template/build callsites
    reference `bout.a_flag` etc.
    """
    if not country:
        return ""
    iso = _COUNTRY_ISO.get(country, "")
    if not iso:
        return ""
    base = iso.split("-")[0].lower()
    if len(base) != 2:
        return ""
    return f"https://flagcdn.com/24x18/{base}.png"


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
    """
    Classify a clear fighting style from computed stats.
    Returns "" when no specific style stands out — caller hides the badge
    rather than falsely labelling everyone "Complete".
    """
    try:
        ko  = float(ko_rate  or 0)
        sub = float(sub_rate or 0)
        td  = float(td_per_min or 0)
        sp  = float(slpm or 0)
        acc = float(sig_acc or 0)
    except (TypeError, ValueError):
        return ""

    if ko > 0.30:
        return "KO Artist"
    if sub > 0.30:
        return "Submission"
    if td > 2.5 and sub > 0.10:
        return "Wrestler"
    if td > 1.5:
        return "Grappler"
    if sp > 5.5 or (acc > 0.48 and sp > 4.5):
        return "Striker"
    if ko > 0.15 and sp > 4.0:
        return "Power Striker"
    return ""  # no clear style — hide badge in template


_LIVE_TERMINAL = {"STATUS_FINAL", "STATUS_SCHEDULED", "STATUS_CANCELED",
                  "STATUS_POSTPONED", "", None}


def _is_live_status(status: str) -> bool:
    """
    Treat any ESPN status that isn't terminal (final/scheduled/canceled) and
    isn't empty as 'live'. This catches STATUS_IN_PROGRESS, STATUS_PRE_FIGHT,
    STATUS_FIGHTERS_WALKING, STATUS_FIGHTER_INTRODUCTIONS, STATUS_END_OF_PERIOD,
    etc. — all of which mean the bout is happening *right now*.
    """
    return bool(status) and status not in _LIVE_TERMINAL


def _was_prediction_correct(
    actual_winner: str,
    fighter_a_name: str,
    fighter_b_name: str,
    prob_a_wins: float | None,
) -> bool | None:
    """
    Compare ESPN's actual winner against our model's pick.
    Returns True/False, or None when the result isn't decidable
    (no probability, or the winner string doesn't match either fighter).
    """
    if prob_a_wins is None or actual_winner is None:
        return None
    try:
        prob_a = float(prob_a_wins)
    except (TypeError, ValueError):
        return None
    # Placeholder probability (0.5 — model never ran for this fight): treat
    # as ungraded. Otherwise we'd be scoring a coin-flip pick against the
    # actual outcome, producing ~50% noise that looks like real accuracy.
    if abs(prob_a - 0.5) < 0.005:
        return None

    aw = (actual_winner or "").strip().lower()
    fa = (fighter_a_name or "").strip().lower()
    fb = (fighter_b_name or "").strip().lower()
    if not aw:
        return None

    # Match on full name OR last-name token (ESPN sometimes returns just one
    # name segment for fighters with diacritics or compound names).
    def _matches(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return True
        a_last = a.split()[-1] if a.split() else a
        b_last = b.split()[-1] if b.split() else b
        return bool(a_last) and bool(b_last) and (a_last == b_last)

    if _matches(aw, fa):
        a_won = True
    elif _matches(aw, fb):
        a_won = False
    else:
        return None

    predicted_a = prob_a >= 0.5
    return predicted_a == a_won


def _format_result_text(winner: str, method_raw: str, round_num: int | None) -> str:
    """
    Build a clean "Winner by METHOD R#" string from ESPN status fields.
    `method_raw` may be a real finish method ("KO/TKO", "Decision - Unanimous"),
    or just "Final" — in which case we drop the "by METHOD" portion entirely.
    """
    if not winner:
        return ""
    method = (method_raw or "").strip()
    method_upper = method.upper()
    method_short = ""
    if "KO" in method_upper or "TKO" in method_upper:
        method_short = "KO/TKO"
    elif "SUB" in method_upper:
        method_short = "SUB"
    elif "DEC" in method_upper or "DECISION" in method_upper:
        method_short = "DEC"
    elif method_upper and method_upper != "FINAL":
        method_short = method  # unrecognised but non-empty — show raw

    rnd_part = f" R{int(round_num)}" if round_num else ""
    if method_short:
        return f"{winner} by {method_short}{rnd_part}".strip()
    # No usable method — just winner (+ round if known)
    return f"{winner}{rnd_part}".strip()


def _build_preview_bout(
    sb: dict,
    fighter_meta: dict,
    fighter_imgs: dict,
    status_payload: dict | None = None,
) -> dict | None:
    """
    Construct a preview-style bout dict from a schedule entry.
    Used both for fully schedule-only events (no predictions yet) and to
    fill in completed/upcoming bouts on a predicted event when those bouts
    are missing from predictions.json (typical for early prelims that
    finished before the predict step ran).
    """
    fa = (sb.get("fighter_a") or "").strip()
    fb = (sb.get("fighter_b") or "").strip()
    if not fa or not fb:
        return None

    def _initials(n: str) -> str:
        parts = (n or "").split()
        return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else (n[:2].upper() or "?")

    a_meta = fighter_meta.get(fa) or fighter_meta.get(_norm_meta_key(fa)) or {}
    b_meta = fighter_meta.get(fb) or fighter_meta.get(_norm_meta_key(fb)) or {}
    a_stats = a_meta.get("stats", {}) or {}
    b_stats = b_meta.get("stats", {}) or {}

    a_lbl, a_cls = _streak_display(a_stats.get("win_streak"), a_stats.get("loss_streak"))
    b_lbl, b_cls = _streak_display(b_stats.get("win_streak"), b_stats.get("loss_streak"))

    def _pct(v):
        return f"{float(v)*100:.0f}%" if v is not None else "—"
    def _f(v, fmt=".1f"):
        return format(float(v), fmt) if v is not None else "—"

    _is_five = bool(sb.get("is_five_round") or sb.get("is_title_bout"))
    bout = {
        "is_preview":     True,
        "fighter_a_name": fa,
        "fighter_b_name": fb,
        "weight_class":   sb.get("weight_class") or "",
        "is_title_bout":  sb.get("is_title_bout", False),
        "is_five_round":  _is_five,
        "scheduled_rounds": 5 if _is_five else 3,
        "favourite":      "toss-up",
        "prob_a_pct":     "—",
        "prob_b_pct":     "—",
        "prob_a_wins":    0.5,
        "a_initials":     _initials(fa),
        "b_initials":     _initials(fb),
        "a_img":          a_meta.get("image_url") or fighter_imgs.get(fa, ""),
        "b_img":          b_meta.get("image_url") or fighter_imgs.get(fb, ""),
        "a_streak":       a_lbl,
        "b_streak":       b_lbl,
        "a_streak_class": a_cls,
        "b_streak_class": b_cls,
        "a_flag":         _flag_emoji(a_meta.get("country", "")),
        "b_flag":         _flag_emoji(b_meta.get("country", "")),
        "a_nationality":  a_meta.get("country", ""),
        "b_nationality":  b_meta.get("country", ""),
        "a_ufc_style":    a_meta.get("style", ""),
        "b_ufc_style":    b_meta.get("style", ""),
        "a_record":       a_meta.get("record", ""),
        "b_record":       b_meta.get("record", ""),
        "a_stance":       a_stats.get("stance", ""),
        "b_stance":       b_stats.get("stance", ""),
        "a_n_fights":     a_stats.get("n_fights", 0),
        "b_n_fights":     b_stats.get("n_fights", 0),
        "a_l3":           _pct(a_stats.get("l3_win_rate")),
        "b_l3":           _pct(b_stats.get("l3_win_rate")),
        "a_finish_rate":  _pct(a_stats.get("finish_rate")),
        "b_finish_rate":  _pct(b_stats.get("finish_rate")),
        "a_ko_rate":      _pct(a_stats.get("ko_rate")),
        "b_ko_rate":      _pct(b_stats.get("ko_rate")),
        "a_sub_rate":     _pct(a_stats.get("sub_rate")),
        "b_sub_rate":     _pct(b_stats.get("sub_rate")),
        "a_slpm":         _f(a_stats.get("slpm")),
        "b_slpm":         _f(b_stats.get("slpm")),
        "col_l3":         _stat_colors(a_stats.get("l3_win_rate"), b_stats.get("l3_win_rate")),
        "col_finish":     _stat_colors(a_stats.get("finish_rate"), b_stats.get("finish_rate")),
        "col_ko":         _stat_colors(a_stats.get("ko_rate"), b_stats.get("ko_rate")),
        "col_sub":        _stat_colors(a_stats.get("sub_rate"), b_stats.get("sub_rate")),
        "col_slpm":       _stat_colors(a_stats.get("slpm"), b_stats.get("slpm")),
        "a_fighter_type": "",
        "b_fighter_type": "",
        "show_fighter_type": False,
        "value_bet_count": 0,
        "has_bets":       False,
        "has_props":      False,
        "has_stats":      bool(a_stats or b_stats),
    }

    # Live status (LIVE pill, RESULT pill + result text)
    sp = status_payload or {}
    espn_status = sp.get("status", "")
    bout["is_live"]      = _is_live_status(espn_status)
    bout["is_completed"] = espn_status == "STATUS_FINAL"
    bout["live_winner"]  = sp.get("winner", "")
    bout["live_method"]  = sp.get("method", "")
    bout["live_round"]   = int(sp.get("round") or 0)
    if bout["is_completed"] and bout["live_winner"]:
        bout["result_text"] = _format_result_text(
            bout["live_winner"], bout["live_method"], bout["live_round"]
        )
    else:
        bout["result_text"] = ""

    return bout


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


_LIVE_STATE_FIELDS = (
    "is_completed", "is_live", "live_winner", "live_method", "live_round",
    "result_text", "pred_correct",
)


def _persist_live_results(events: list[dict]) -> None:
    """
    Merge any bouts with live ESPN results (is_completed=True + live_winner)
    into data/past_events.json, keyed by upcoming_bout_id. This captures the
    state *before* ESPN's scoreboard drops the event from its current
    window, so the dashboard can keep showing actual results in the Past
    Events panel even after the event falls out of the live schedule.

    Called during the dashboard build, after bouts have been enriched with
    live status from upcoming_schedule.json.
    """
    if not events:
        return

    # Read current past_events.json. Key by (event_date, lower(name_a),
    # lower(name_b)) — sorted so swapped corner orders dedupe cleanly. The
    # prediction's upcoming_bout_id and synthetic preview-bout IDs vary
    # across runs, so keying purely on bout_id leaves duplicates.
    def _bout_key(p: dict) -> tuple:
        a = (p.get("fighter_a_name") or "").strip().lower()
        b = (p.get("fighter_b_name") or "").strip().lower()
        names = tuple(sorted([a, b]))
        return (str(p.get("event_date", "")), names)

    existing: dict[tuple, dict] = {}
    if PAST_EVENTS_PATH.exists():
        try:
            arr = json.loads(PAST_EVENTS_PATH.read_text(encoding="utf-8"))
            for p in arr:
                if not p.get("event_date"):
                    continue
                key = _bout_key(p)
                if key[1][0] and key[1][1]:  # both fighter names present
                    existing[key] = p
        except json.JSONDecodeError:
            existing = {}

    changed = False
    for event in events:
        for bout in event.get("bouts", []):
            if not bout.get("is_completed") or not bout.get("live_winner"):
                continue
            key = _bout_key(bout)
            if not (key[0] and key[1][0] and key[1][1]):
                continue  # missing event date or fighter names — can't key
            bid = bout.get("upcoming_bout_id") or ""
            if not bid:
                # Preview-only bouts (early prelims missing from predictions)
                # don't have a stable bout_id. Synthesise one so the entry has
                # a stable identifier downstream, but dedup is by name+date.
                bid = "live-" + str(abs(hash((
                    str(bout.get("event_date", "")),
                    (bout.get("fighter_a_name") or "").lower(),
                    (bout.get("fighter_b_name") or "").lower(),
                ))))[:16]
            entry = dict(existing.get(key, {}))
            # Carry forward any prediction data the bout already has, plus
            # the live state fields so the past-events template can render
            # winner/method/round without a DB lookup.
            for k in (
                "event_date", "event_name", "fighter_a_name", "fighter_b_name",
                "weight_class", "is_title_bout", "prob_a_wins", "prob_b_wins",
                "props", "sportsbet_odds", "bet_analysis",
                "fighter_a_nationality", "fighter_b_nationality",
                "uncertainty_std", "ci_90_lo", "ci_90_hi",
            ):
                if bout.get(k) is not None:
                    entry[k] = bout.get(k)
            for k in _LIVE_STATE_FIELDS:
                if bout.get(k) is not None:
                    entry[k] = bout.get(k)
            entry["upcoming_bout_id"] = bid
            if existing.get(key) != entry:
                existing[key] = entry
                changed = True

    if not changed:
        return

    out = sorted(existing.values(),
                 key=lambda p: str(p.get("event_date", "")),
                 reverse=True)
    PAST_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAST_EVENTS_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("Persisted live results: %d bouts in %s", len(out), PAST_EVENTS_PATH)


def _load_persisted_past_events(fighter_meta: dict, fighter_imgs: dict) -> list[dict]:
    """
    Read data/past_events.json (accumulated by track_predictions.snapshot)
    and group its bouts back into the same event-card structure the
    template expects. Enriches with metadata so flags, images, records
    are present.
    """
    if not PAST_EVENTS_PATH.exists():
        return []
    try:
        flat = json.loads(PAST_EVENTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(flat, list) or not flat:
        return []

    # Re-run bet analysis so past entries pick up the current bet schema —
    # in particular `outcome_keys`, which earlier persists predate. Keeps
    # historical bet grading working without a manual backfill.
    try:
        from ufc_predict.eval.bet_analysis import analyze_all_fights
        flat = analyze_all_fights(flat)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Re-running bet_analysis on past events failed: %s", exc)

    by_event: dict[tuple, dict] = {}
    for p in flat:
        key = (str(p.get("event_date", "")), p.get("event_name", "Unknown Event"))
        ev  = by_event.setdefault(key, {
            "event_name": p.get("event_name", "Unknown Event"),
            "event_date": str(p.get("event_date", "")),
            "bouts":      [],
            "is_past":    True,
        })

        a_name = p.get("fighter_a_name", "")
        b_name = p.get("fighter_b_name", "")
        a_meta = fighter_meta.get(a_name) or fighter_meta.get(_norm_meta_key(a_name)) or {}
        b_meta = fighter_meta.get(b_name) or fighter_meta.get(_norm_meta_key(b_name)) or {}

        def _initials(n: str) -> str:
            parts = (n or "").split()
            if len(parts) >= 2:
                return (parts[0][0] + parts[-1][0]).upper()
            return n[:2].upper() or "?"

        sb = p.get("sportsbet_odds") or {}
        bout = {
            **p,  # keep all original prediction data (probs, props, etc.)
            "a_initials":   _initials(a_name),
            "b_initials":   _initials(b_name),
            "a_img":        a_meta.get("image_url") or fighter_imgs.get(a_name, ""),
            "b_img":        b_meta.get("image_url") or fighter_imgs.get(b_name, ""),
            "a_flag":       _flag_emoji(a_meta.get("country", "")),
            "b_flag":       _flag_emoji(b_meta.get("country", "")),
            "a_nationality": a_meta.get("country", ""),
            "b_nationality": b_meta.get("country", ""),
            "a_record":     a_meta.get("record", ""),
            "b_record":     b_meta.get("record", ""),
            "fight_time_aest": _format_fight_time_aest(
                sb.get("start_time"), str(p.get("event_date", ""))
            ),
        }
        ev["bouts"].append(bout)

    return list(by_event.values())


def _enrich_past_events(past_events: list[dict]) -> list[dict]:
    """
    For each past event, try to look up actual fight outcomes in the DB
    and tag each bout's prediction as ✓ (correct) / ✗ (wrong) / ? (unknown).
    Also computes per-event accuracy + total correct.
    """
    if not past_events:
        return []

    try:
        from sqlalchemy import text

        from ufc_predict.db.session import get_session_factory
        factory = get_session_factory()
    except Exception:
        # No DB available — just mark events as past with no outcomes.
        for ev in past_events:
            ev["is_past"]    = True
            ev["n_correct"]  = 0
            ev["n_resolved"] = 0
        return past_events

    with factory() as session:
        # Map (last_name_a.lower(), last_name_b.lower(), event_date) → result row
        sql = text("""
            SELECT f.fight_id, f.date, fa.full_name AS name_a, fb.full_name AS name_b,
                   f.method, f.round_ended, f.winner_fighter_id,
                   f.red_fighter_id, f.blue_fighter_id
            FROM fights f
            JOIN fighters fa ON fa.canonical_fighter_id = f.red_fighter_id
            JOIN fighters fb ON fb.canonical_fighter_id = f.blue_fighter_id
            WHERE f.date >= :since
        """)
        from datetime import timedelta
        since = (date.today() - timedelta(days=365)).isoformat()
        rows = session.execute(sql, {"since": since}).fetchall()

        # Bulk-load fight totals (round=0) for every fight in the window so we
        # can attach actual strikes/takedowns to each past-event bout without
        # an N+1 query. Keyed by (fight_id, fighter_id).
        stats_sql = text("""
            SELECT s.fight_id, s.fighter_id,
                   s.sig_strikes_landed, s.sig_strikes_attempted,
                   s.total_strikes_landed, s.total_strikes_attempted,
                   s.takedowns_landed, s.takedowns_attempted,
                   s.knockdowns, s.submission_attempts, s.control_time_sec
            FROM fight_stats_round s
            JOIN fights f ON f.fight_id = s.fight_id
            WHERE f.date >= :since AND s.round = 0
        """)
        stats_lookup: dict[tuple, dict] = {}
        for sr in session.execute(stats_sql, {"since": since}).fetchall():
            stats_lookup[(sr.fight_id, sr.fighter_id)] = {
                "sig_strikes_landed":    sr.sig_strikes_landed or 0,
                "sig_strikes_attempted": sr.sig_strikes_attempted or 0,
                "total_strikes_landed":  sr.total_strikes_landed or 0,
                "takedowns_landed":      sr.takedowns_landed or 0,
                "takedowns_attempted":   sr.takedowns_attempted or 0,
                "knockdowns":            sr.knockdowns or 0,
                "submission_attempts":   sr.submission_attempts or 0,
                "control_time_sec":      sr.control_time_sec or 0,
            }

        from rapidfuzz import fuzz
        def _score(a: str, b: str) -> int:
            return max(
                fuzz.token_set_ratio(a, b),
                fuzz.partial_ratio((a or "").split()[-1] if a else "", b or ""),
            )

        for ev in past_events:
            ev["is_past"]    = True
            ev_date          = str(ev.get("event_date", ""))
            n_correct, n_resolved = 0, 0
            for bout in ev.get("bouts", []):
                pa = bout.get("fighter_a_name", "")
                pb = bout.get("fighter_b_name", "")
                # Find best matching fight row near this date — try both
                # orientations because DB red/blue order may not match our A/B.
                best_score, best_row = 0, None
                for row in rows:
                    try:
                        row_date = date.fromisoformat(str(row.date))
                        if abs((row_date - date.fromisoformat(ev_date)).days) > 3:
                            continue
                    except ValueError:
                        continue
                    s_normal  = min(_score(pa, row.name_a), _score(pb, row.name_b))
                    s_flipped = min(_score(pa, row.name_b), _score(pb, row.name_a))
                    s = max(s_normal, s_flipped)
                    if s > best_score:
                        best_score, best_row = s, row
                if best_row is None or best_score < 70:
                    # No DB match — but live ESPN status may already have a
                    # winner (typical for bouts that finished in the last few
                    # hours and haven't been ingested into the fights table yet).
                    if bout.get("is_completed") and bout.get("live_winner"):
                        bout["actual_winner_name"] = bout["live_winner"]
                        # Map winner name back to A/B side for the template.
                        lw = (bout["live_winner"] or "").strip().lower()
                        a_lc = (pa or "").strip().lower()
                        b_lc = (pb or "").strip().lower()
                        if a_lc and (lw == a_lc or lw in a_lc or a_lc in lw):
                            bout["actual_winner_side"] = "a"
                        elif b_lc and (lw == b_lc or lw in b_lc or b_lc in lw):
                            bout["actual_winner_side"] = "b"
                        bout["actual_method"] = bout.get("live_method", "")
                        bout["actual_round"]  = bout.get("live_round", 0)
                        # Override stale persisted pred_correct when the input
                        # prob is a 0.5 placeholder — those grades are bogus
                        # coin flips. See _was_prediction_correct rationale.
                        _pa_raw = bout.get("prob_a_wins")
                        try:
                            _is_placeholder = (
                                _pa_raw is None or abs(float(_pa_raw) - 0.5) < 0.005
                            )
                        except (TypeError, ValueError):
                            _is_placeholder = True
                        if _is_placeholder:
                            bout["pred_correct"] = None
                        elif bout.get("pred_correct") is True:
                            n_correct  += 1
                            n_resolved += 1
                        elif bout.get("pred_correct") is False:
                            n_resolved += 1
                    else:
                        bout["actual_winner"] = "?"
                        bout["actual_method"] = ""
                        bout["pred_correct"]  = None
                    continue
                # Determine which side our fighter A is on in the DB row
                a_score_red  = _score(pa, best_row.name_a)
                a_score_blue = _score(pa, best_row.name_b)
                a_is_red = a_score_red >= a_score_blue
                winner_is_red = best_row.winner_fighter_id == best_row.red_fighter_id
                a_won = (winner_is_red == a_is_red)
                bout["actual_winner_name"] = pa if a_won else pb
                bout["actual_winner_side"] = "a" if a_won else "b"
                bout["actual_method"]      = best_row.method or ""
                bout["actual_round"]       = best_row.round_ended
                # Attach per-fighter fight totals (sig strikes, takedowns, etc.)
                # using the red/blue → A/B mapping resolved above.
                a_fid = best_row.red_fighter_id if a_is_red else best_row.blue_fighter_id
                b_fid = best_row.blue_fighter_id if a_is_red else best_row.red_fighter_id
                bout["actual_stats_a"] = stats_lookup.get((best_row.fight_id, a_fid))
                bout["actual_stats_b"] = stats_lookup.get((best_row.fight_id, b_fid))
                # Was our prediction correct? Skip grading for 0.5 placeholders
                # (model never ran for this fight) — same reason as above.
                _pa_raw = bout.get("prob_a_wins")
                if _pa_raw is None or abs(float(_pa_raw) - 0.5) < 0.005:
                    bout["pred_correct"] = None
                else:
                    pa_prob = float(_pa_raw)
                    predicted_a = pa_prob >= 0.5
                    bout["pred_correct"] = (predicted_a == a_won)
                    n_resolved += 1
                    if bout["pred_correct"]:
                        n_correct += 1

            ev["n_correct"]  = n_correct
            ev["n_resolved"] = n_resolved
            ev["accuracy_pct"] = (
                round(100 * n_correct / n_resolved) if n_resolved > 0 else None
            )

            # After the actual outcome is resolved, grade every value bet
            # we'd have placed and attach it to the bout so the past-events
            # template can render a "what we said vs what happened" table.
            for bout in ev.get("bouts", []):
                _attach_graded_bets(bout)

    return past_events


# ---------------------------------------------------------------------------
# Bet grading for past events
# ---------------------------------------------------------------------------

def _canonical_method(method: str) -> str | None:
    """Reduce a free-text method string to KO / SUB / DEC, or None if unknown."""
    m = (method or "").upper()
    if not m or "FINAL" == m.strip():
        return None
    if "KO" in m or "TKO" in m:
        return "KO"
    if "SUB" in m:
        return "SUB"
    if "DEC" in m or "POINT" in m or "DRAW" in m:
        return "DEC"
    return None


def _actual_outcome_key(bout: dict) -> str | None:
    """
    Build the atomic outcome key ("A|KO|2", "B|DEC|DEC", …) that bet_analysis
    emits in `outcome_keys`. Returns None when we don't have enough data to
    grade (no winner yet, or method not classifiable).
    """
    side = (bout.get("actual_winner_side") or "").upper()
    if side not in {"A", "B"}:
        return None
    method = _canonical_method(bout.get("actual_method") or "")
    if method is None:
        return None
    if method == "DEC":
        return f"{side}|DEC|DEC"
    rnd = bout.get("actual_round")
    try:
        rnd_int = int(rnd) if rnd else 0
    except (TypeError, ValueError):
        rnd_int = 0
    if rnd_int < 1 or rnd_int > 5:
        return None
    return f"{side}|{method}|{rnd_int}"


def _grade_bet(bet: dict, actual_key: str | None) -> str | None:
    """Return 'won' / 'lost' / None for a single bet given the actual outcome key."""
    if not actual_key:
        return None
    keys = bet.get("outcome_keys") or []
    if not keys:
        return None
    return "won" if actual_key in keys else "lost"


def _attach_graded_bets(bout: dict) -> None:
    """Attach graded bet lists (top picks + all bets) to a past-event bout."""
    bets = bout.get("bet_analysis") or []
    if not bets:
        bout["graded_top_bets"] = []
        bout["graded_all_bets"] = []
        return

    actual_key = _actual_outcome_key(bout)
    graded: list[dict] = []
    for bet in bets:
        result = _grade_bet(bet, actual_key)
        graded.append({
            "description":  bet.get("description") or "",
            "bet_type":     bet.get("bet_type") or "",
            "our_prob":     bet.get("our_prob") or 0.0,
            "sb_odds":      bet.get("sb_odds") or 0.0,
            "ev_pct":       bet.get("ev_pct") or 0.0,
            "is_value":     bool(bet.get("is_value")),
            "result":       result,  # 'won' | 'lost' | None
        })

    # Top picks = top-EV value bets only, capped to 5. Mirrors the
    # "Top picks" portfolio mode used elsewhere in the dashboard.
    value_only = [g for g in graded if g["is_value"]]
    value_only.sort(key=lambda g: g["ev_pct"], reverse=True)
    bout["graded_top_bets"] = value_only[:5]

    # All bets = every analysed bet (value or not), sorted by EV desc.
    graded_sorted = sorted(graded, key=lambda g: g["ev_pct"], reverse=True)
    bout["graded_all_bets"] = graded_sorted

    # Counts for the toggle labels
    won_count = sum(1 for g in value_only if g["result"] == "won")
    bout["graded_top_won"]   = won_count
    bout["graded_top_total"] = len(value_only)


def _enrich_bet_analysis(bout: dict) -> None:
    """Format bet_analysis list for display — top 10 value bets only."""
    bets = bout.get("bet_analysis") or []
    value_bets = [b for b in bets if b.get("is_value")][:10]

    formatted = []
    for bet in value_bets:
        ev_pct  = float(bet.get("ev_pct", 0))
        kelly_p = float(bet.get("kelly_pct", 0))
        odds    = float(bet.get("sb_odds", 1.0))
        our_p   = float(bet.get("our_prob", 0))
        # Two distinct numbers:
        #   win_payout    — net profit IF bet wins ((odds-1) * stake)
        #   expected_pnl  — long-run average per bet (probability-weighted), == EV
        win_payout   = round((odds - 1) * 100)
        expected_pnl = round((our_p * odds - 1) * 100)
        formatted.append({
            "description":   bet.get("description", ""),
            "our_prob":      f"{our_p * 100:.0f}%",
            "sb_odds":       f"{odds:.2f}",
            "win_payout":    f"+${win_payout}",
            "expected_pnl":  f"+${expected_pnl}" if expected_pnl > 0 else f"-${abs(expected_pnl)}",
            "expected_sign": "pos" if expected_pnl > 0 else "neg",
            "stake_pct":     f"{kelly_p:.1f}%" if kelly_p > 0 else "<1%",
            "is_value":      True,
            "ev_raw":        ev_pct,
        })
    bout["bet_rows"] = formatted
    bout["has_bets"] = bool(formatted)
    bout["value_bet_count"] = len(formatted)


def build(output_dir: Path = OUTPUT_DIR) -> None:
    predictions = load_predictions()
    if not predictions:
        log.warning("No predictions found at %s", PREDICTIONS_PATH)

    fighter_imgs = load_fighter_images()
    fighter_meta = load_fighter_metadata()
    events = _group_by_event(predictions)

    # Per-bout start times AND live status from the ESPN-derived schedule.
    # We use them as fallbacks when SportsBet has nothing (prelims often
    # missing from SportsBet's listing until close to fight day) and to
    # render LIVE / RESULT state during live cards.
    schedule_for_lookup = load_schedule()
    schedule_time_lookup: dict[tuple, str] = {}
    schedule_status_lookup: dict[tuple, dict] = {}
    # ESPN's weight_class lives on the schedule, not the prediction (which is
    # generated from the DB's upcoming_bouts table). Backfill so the dashboard
    # picks it up on rebuild without a full re-predict cycle.
    schedule_wc_lookup: dict[tuple, str] = {}
    # is_five_round / is_title_bout from the schedule, used to derive
    # scheduled_rounds for the dashboard chip even when the prediction
    # row was generated before the upcoming_poller learned the flag.
    schedule_5r_lookup: dict[tuple, bool] = {}
    schedule_title_lookup: dict[tuple, bool] = {}
    for _ev in schedule_for_lookup:
        ev_date = str(_ev.get("event_date", ""))
        for _b in _ev.get("bouts") or []:
            fa = _norm_name(_b.get("fighter_a"))
            fb = _norm_name(_b.get("fighter_b"))
            if not (fa and fb):
                continue
            t = _b.get("start_time_iso") or ""
            if t:
                schedule_time_lookup[(ev_date, fa, fb)] = t
                schedule_time_lookup[(ev_date, fb, fa)] = t
            wc = (_b.get("weight_class") or "").strip()
            if wc:
                schedule_wc_lookup[(ev_date, fa, fb)] = wc
                schedule_wc_lookup[(ev_date, fb, fa)] = wc
            if _b.get("is_five_round"):
                schedule_5r_lookup[(ev_date, fa, fb)] = True
                schedule_5r_lookup[(ev_date, fb, fa)] = True
            if _b.get("is_title_bout"):
                schedule_title_lookup[(ev_date, fa, fb)] = True
                schedule_title_lookup[(ev_date, fb, fa)] = True
            status_payload = {
                "status":  _b.get("espn_status") or "",
                "winner":  _b.get("espn_winner_name") or "",
                "method":  _b.get("espn_method") or "",
                "round":   int(_b.get("espn_round") or 0),
            }
            if status_payload["status"]:
                schedule_status_lookup[(ev_date, fa, fb)] = status_payload
                schedule_status_lookup[(ev_date, fb, fa)] = status_payload

    def _meta_lookup(name: str, key: str, default: str = "") -> str:
        # Try the exact name first, then fall back to the normalised alias
        # so unaccented predictions ("Joel Alvarez") still hit the
        # accented metadata record ("Joel Álvarez").
        meta = fighter_meta.get(name) or fighter_meta.get(_norm_meta_key(name)) or {}
        val = meta.get(key) or default
        return val

    # Today (for past/upcoming event split + past-event result fetch)
    today_str = date.today().isoformat()

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
            a_label, a_class = _streak_display(
                bout.get("a_win_streak"), bout.get("a_loss_streak")
            )
            b_label, b_class = _streak_display(
                bout.get("b_win_streak"), bout.get("b_loss_streak")
            )
            bout["a_streak"]       = a_label   # plain English label
            bout["b_streak"]       = b_label
            bout["a_streak_class"] = a_class   # css class for colour
            bout["b_streak_class"] = b_class

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

            # Country flags — prefer fighter_metadata (scraped from UFC.com bio)
            # over the DB nationality field, which may be sparse.
            a_name = bout.get("fighter_a_name", "")
            b_name = bout.get("fighter_b_name", "")
            a_nat = _meta_lookup(a_name, "country") or bout.get("fighter_a_nationality") or ""
            b_nat = _meta_lookup(b_name, "country") or bout.get("fighter_b_nationality") or ""
            bout["a_flag"] = _flag_emoji(a_nat)
            bout["b_flag"] = _flag_emoji(b_nat)
            bout["a_nationality"] = a_nat
            bout["b_nationality"] = b_nat
            # Official UFC fighting style (e.g. "Brazilian Jiu-Jitsu", "Kickboxer")
            bout["a_ufc_style"] = _meta_lookup(a_name, "style")
            bout["b_ufc_style"] = _meta_lookup(b_name, "style")
            # Career record (W-L-D)
            bout["a_record"] = _meta_lookup(a_name, "record")
            bout["b_record"] = _meta_lookup(b_name, "record")

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
                if len(parts) >= 2:
                    return (parts[0][0] + parts[-1][0]).upper()
                return name[:2].upper() or "?"
            bout["a_initials"] = _initials(bout.get("fighter_a_name", ""))
            bout["b_initials"] = _initials(bout.get("fighter_b_name", ""))

            # Fighter images: prefer metadata's image_url (already downloaded
            # to docs/fighter-images/). Fall back to legacy fighter_images cache.
            a_img = (
                _meta_lookup(a_name, "image_url")
                or fighter_imgs.get(a_name)
                or fighter_imgs.get(_norm_meta_key(a_name), "")
            )
            b_img = (
                _meta_lookup(b_name, "image_url")
                or fighter_imgs.get(b_name)
                or fighter_imgs.get(_norm_meta_key(b_name), "")
            )
            bout["a_img"] = a_img
            bout["b_img"] = b_img

            # Stance from DB (factual UFC data)
            bout["a_stance"] = (
                (bout.get("fighter_a_stance") or "").replace("Switch", "Switch Stance")
            )
            bout["b_stance"] = (
                (bout.get("fighter_b_stance") or "").replace("Switch", "Switch Stance")
            )

            # Fighter style badge — only if data quality is good enough
            n_min = min(bout.get("a_n_fights") or 0, bout.get("b_n_fights") or 0)
            bout["show_fighter_type"] = n_min >= 5
            bout["a_fighter_type"] = bout.get("a_fighter_type", "")
            bout["b_fighter_type"] = bout.get("b_fighter_type", "")

            # Stat color classes for the comparison table (raw values before formatting)
            bout["col_streak"] = _stat_colors(
                bout.get("a_win_streak"), bout.get("b_win_streak")
            )
            bout["col_l3"] = _stat_colors(
                bout.get("a_l3_win_rate"), bout.get("b_l3_win_rate")
            )
            bout["col_finish"] = _stat_colors(
                bout.get("a_finish_rate"), bout.get("b_finish_rate")
            )
            bout["col_ko"] = _stat_colors(bout.get("a_ko_rate"), bout.get("b_ko_rate"))
            bout["col_sub"] = _stat_colors(bout.get("a_sub_rate"), bout.get("b_sub_rate"))
            bout["col_slpm"] = _stat_colors(bout.get("a_slpm"), bout.get("b_slpm"))
            bout["col_sapm"] = _stat_colors(
                bout.get("a_sapm"), bout.get("b_sapm"), higher_is_better=False
            )
            bout["col_td"] = _stat_colors(
                bout.get("a_td_per_min"), bout.get("b_td_per_min")
            )

            # Prop probabilities
            _enrich_props(bout)
            # EV / Kelly bet analysis
            _enrich_bet_analysis(bout)

            # SportsBet moneyline odds for display
            sb = bout.get("sportsbet_odds") or {}
            bout["sb_odds_a"] = f"{sb['moneyline_a']:.2f}" if sb.get("moneyline_a") else None
            bout["sb_odds_b"] = f"{sb['moneyline_b']:.2f}" if sb.get("moneyline_b") else None
            # Fight time in AEST. Prefer SportsBet's per-event start_time;
            # fall back to ESPN's per-competition date from the schedule when
            # SportsBet hasn't listed this fight yet (typical for prelims).
            ev_date_str = str(bout.get("event_date", ""))
            fa_key = _norm_name(bout.get("fighter_a_name"))
            fb_key = _norm_name(bout.get("fighter_b_name"))
            start_time = sb.get("start_time") or schedule_time_lookup.get(
                (ev_date_str, fa_key, fb_key)
            )
            bout["fight_time_aest"] = _format_fight_time_aest(start_time, ev_date_str)

            # Backfill weight class from the schedule when the prediction's
            # field is empty (predictions.json may have been generated before
            # the poller started capturing this).
            if not (bout.get("weight_class") or "").strip():
                wc_from_sched = schedule_wc_lookup.get((ev_date_str, fa_key, fb_key))
                if wc_from_sched:
                    bout["weight_class"] = wc_from_sched

            # Scheduled rounds chip. Trust the prediction's flag first, then
            # fall back to the schedule (lets a freshly-polled main-event
            # 5R flag show on the dashboard even before re-prediction).
            is_five = bool(
                bout.get("is_five_round")
                or bout.get("is_title_bout")
                or schedule_5r_lookup.get((ev_date_str, fa_key, fb_key))
                or schedule_title_lookup.get((ev_date_str, fa_key, fb_key))
            )
            bout["scheduled_rounds"] = 5 if is_five else 3

            # Live status — drives LIVE / RESULT pills on the bout row and the
            # event-level LIVE EVENT header. Status is empty for everything
            # not on the current/next-day window.
            sched_status = schedule_status_lookup.get((ev_date_str, fa_key, fb_key)) or {}
            espn_status = sched_status.get("status", "")
            bout["is_live"]      = _is_live_status(espn_status)
            bout["is_completed"] = espn_status == "STATUS_FINAL"
            bout["live_winner"]  = sched_status.get("winner", "")
            bout["live_method"]  = sched_status.get("method", "")
            bout["live_round"]   = sched_status.get("round", 0)
            if bout["is_completed"] and bout["live_winner"]:
                bout["result_text"] = _format_result_text(
                    bout["live_winner"], bout["live_method"], bout["live_round"]
                )
                bout["pred_correct"] = _was_prediction_correct(
                    bout["live_winner"],
                    bout.get("fighter_a_name", ""),
                    bout.get("fighter_b_name", ""),
                    bout.get("prob_a_wins"),
                )
            else:
                bout["result_text"] = ""
                bout["pred_correct"] = None

    # Some bouts on a predicted event may be missing from predictions.json
    # (typical for early prelims that finished before the predict step ran,
    # or for fights too far out to score). Merge in any schedule bouts not
    # already present so the card shows ALL fights with live/result state.
    schedule_by_event: dict[tuple, dict] = {
        (str(_e.get("event_date", "")), _e.get("event_name", "")): _e
        for _e in schedule_for_lookup
    }
    from rapidfuzz import fuzz

    def _bout_already_covered(fa: str, fb: str, pairs: list[tuple[str, str]]) -> bool:
        """Fuzzy match a (fighter_a, fighter_b) pair against existing bouts.

        Strict equality misses 1-letter typos that the data sources disagree
        on — e.g. "Bernardo Sopaj" (UFC.com / ESPN) vs "Benardo Sopaj" (DB).
        Without this, the schedule version is treated as a new bout and the
        dashboard renders Sopaj/Cuamba twice for the same event.
        """
        for ea, eb in pairs:
            score_normal  = min(fuzz.ratio(fa, ea), fuzz.ratio(fb, eb))
            score_flipped = min(fuzz.ratio(fa, eb), fuzz.ratio(fb, ea))
            if max(score_normal, score_flipped) >= 88:
                return True
        return False

    for event in events:
        ekey = (str(event.get("event_date", "")), event.get("event_name", ""))
        sched_event = schedule_by_event.get(ekey)
        if not sched_event:
            continue
        existing_pairs: list[tuple[str, str]] = []
        for b in event["bouts"]:
            a = _norm_name(b.get("fighter_a_name"))
            bn = _norm_name(b.get("fighter_b_name"))
            existing_pairs.append((a, bn))
        for sb in sched_event.get("bouts", []):
            fa = _norm_name(sb.get("fighter_a"))
            fb = _norm_name(sb.get("fighter_b"))
            if _bout_already_covered(fa, fb, existing_pairs):
                continue
            status_payload = {
                "status":  sb.get("espn_status") or "",
                "winner":  sb.get("espn_winner_name") or "",
                "method":  sb.get("espn_method") or "",
                "round":   sb.get("espn_round") or 0,
            }
            pb = _build_preview_bout(sb, fighter_meta, fighter_imgs, status_payload)
            if pb is None:
                continue
            pb["event_date"] = str(event.get("event_date", ""))
            pb["event_name"] = event.get("event_name", "")
            pb["fight_time_aest"] = _format_fight_time_aest(
                sb.get("start_time_iso"), str(event.get("event_date", ""))
            )
            event["bouts"].append(pb)

        # Order bouts to match the schedule (ESPN lists earliest prelim first
        # → main event last). Bouts not in the schedule fall to the end.
        # Keep the schedule in list form so the sort key can fuzzy-match
        # name typos (e.g. Bernardo vs Benardo Sopaj).
        sched_pairs = [
            (_norm_name(sb.get("fighter_a")), _norm_name(sb.get("fighter_b")))
            for sb in sched_event.get("bouts", [])
        ]

        def _bout_sort_key(b: dict) -> int:
            a = _norm_name(b.get("fighter_a_name"))
            bn = _norm_name(b.get("fighter_b_name"))
            for i, (sa, sb_) in enumerate(sched_pairs):
                score_normal  = min(fuzz.ratio(a, sa), fuzz.ratio(bn, sb_))
                score_flipped = min(fuzz.ratio(a, sb_), fuzz.ratio(bn, sa))
                if max(score_normal, score_flipped) >= 88:
                    return i
            return 10_000

        event["bouts"].sort(key=_bout_sort_key)

    # Mark each event as live if ANY of its bouts are currently in progress;
    # the template renders a "LIVE EVENT" badge in place of "Next Event".
    for event in events:
        event["is_live"] = any(b.get("is_live") for b in event.get("bouts", []))

    from ufc_predict.eval.bet_analysis import build_portfolio, top_value_bets
    top_bets = top_value_bets(predictions, n=200)  # show all value bets; UI filters by event

    # Group top bets by event so the Top Value Bets page can offer an event filter.
    # The "all" key holds every bet so the user can also see them combined.
    bets_by_event: dict[str, list] = {"__all__": list(top_bets)}
    for bet in top_bets:
        ev_label = bet.get("event") or "Other"
        bets_by_event.setdefault(ev_label, []).append(bet)
    event_filter_options = [
        {"key": k, "label": k, "count": len(v)}
        for k, v in bets_by_event.items() if k != "__all__"
    ]
    event_filter_options.sort(key=lambda x: x["label"])

    # Bankroll-allocation portfolios for the Top Value Bets page.
    # One pair per event filter (incl. "__all__") so when the user picks an
    # event the allocation rebalances correctly — Kelly across 30 bets is
    # very different from Kelly across 4 bets in a single card.
    portfolios_by_event: dict[str, dict] = {}
    for ev_key, blist in bets_by_event.items():
        portfolios_by_event[ev_key] = {
            "kelly":        build_portfolio(blist, strategy="kelly"),
            "flat":         build_portfolio(blist, strategy="flat"),
            "concentrated": build_portfolio(blist, strategy="concentrated"),
        }

    # Build a carousel of all known upcoming events.
    # Merges predicted events (have data) with the full ESPN schedule
    # (covers events too far out for predictions, e.g. 2-3 weeks ahead).
    # Past events (now included in the schedule for results-persistence)
    # are excluded — the carousel is "what's coming up" only.
    schedule = load_schedule()
    pred_events_by_key = {
        (str(ev["event_date"]), ev["event_name"]): ev for ev in events
    }

    # Combined list: schedule first (broader coverage), enrich with prediction data
    seen_keys: set = set()
    event_carousel = []
    all_events = []

    # Schedule entries (broad) — drop anything earlier than today
    for s in schedule:
        if str(s.get("event_date", "")) < today_str:
            continue
        key = (s["event_date"], s["event_name"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_events.append({
            "event_date": s["event_date"],
            "event_name": s["event_name"],
            "n_bouts":    len(s.get("bouts", [])),
            "from_schedule": True,
        })

    # Prediction events (may include extras the schedule missed) — same filter
    for key, ev in pred_events_by_key.items():
        if str(ev.get("event_date", "")) < today_str:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_events.append({
            "event_date": str(ev["event_date"]),
            "event_name": ev["event_name"],
            "n_bouts":    len(ev["bouts"]),
            "from_schedule": False,
        })

    # Sort by date and enrich with prediction stats where available
    all_events.sort(key=lambda e: e["event_date"])
    for ae in all_events:
        key = (ae["event_date"], ae["event_name"])
        pred_ev = pred_events_by_key.get(key)
        if pred_ev:
            n_value     = sum(b.get("value_bet_count", 0) for b in pred_ev["bouts"])
            n_with_odds = sum(1 for b in pred_ev["bouts"] if b.get("sportsbet_odds"))
            n_bouts     = len(pred_ev["bouts"])
            has_preds   = True
        else:
            n_value, n_with_odds = 0, 0
            n_bouts = ae["n_bouts"]
            has_preds = False

        anchor = "event-" + (ae["event_date"] or "").replace(":", "").replace(" ", "-")[:12]
        event_carousel.append({
            "event_name": ae["event_name"],
            "event_date": ae["event_date"],
            "n_bouts":    n_bouts,
            "n_value":    n_value,
            "has_odds":   n_with_odds > 0,
            "has_preds":  has_preds,
            "anchor":     anchor,
        })

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("dashboard.html")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Add schedule-only events as preview entries (basic fighter info, no predictions)
    # so the user can browse/expand them.
    pred_keys = {(str(ev["event_date"]), ev["event_name"]) for ev in events}
    preview_events = []
    for s in schedule:
        key = (s["event_date"], s["event_name"])
        if key in pred_keys or not s.get("bouts"):
            continue
        anchor = "event-" + (s["event_date"] or "").replace(":", "").replace(" ", "-")[:12]
        preview_bouts = []
        for sb in s["bouts"]:
            status_payload = {
                "status":  sb.get("espn_status") or "",
                "winner":  sb.get("espn_winner_name") or "",
                "method":  sb.get("espn_method") or "",
                "round":   sb.get("espn_round") or 0,
            }
            pb = _build_preview_bout(sb, fighter_meta, fighter_imgs, status_payload)
            if pb is not None:
                pb["event_date"] = str(s["event_date"])
                pb["event_name"] = s.get("event_name", "")
                # Time display for preview bouts (predicted events get this in the
                # main loop, but preview events skipped that path entirely).
                pb["fight_time_aest"] = _format_fight_time_aest(
                    sb.get("start_time_iso"), s["event_date"]
                )
                preview_bouts.append(pb)
        if preview_bouts:
            preview_events.append({
                "event_name": s["event_name"],
                "event_date": s["event_date"],
                "bouts":      preview_bouts,
                "is_preview": True,
                "is_live":    any(b.get("is_live") for b in preview_bouts),
                "anchor":     anchor[:18],
            })

    # Combine: predicted events first (sorted by date), then preview events
    all_events_for_template = sorted(events, key=lambda e: str(e.get("event_date", "")))
    all_events_for_template.extend(preview_events)

    # Persist any completed bouts (with ESPN-derived winner/method/round) to
    # data/past_events.json BEFORE we lose them. Once a fight night ends and
    # the date rolls over, ESPN's scoreboard drops the event so the live
    # state vanishes; this snapshot is what keeps the Past Events panel
    # populated with results next build.
    _persist_live_results(all_events_for_template)

    # Split events into upcoming and past, with per-bout granularity for
    # today's live card: completed bouts migrate to a Past Events row for
    # the same event the moment they finish, while live + scheduled bouts
    # stay on the upcoming card. Once every bout has finished, the upcoming
    # entry drops away entirely (the past row already has them all).
    upcoming_events: list[dict] = []
    past_events_today: list[dict] = []
    for e in all_events_for_template:
        ev_date = str(e.get("event_date", ""))
        if ev_date < today_str:
            past_events_today.append(e)
            continue
        bouts = e.get("bouts") or []
        completed = [b for b in bouts if b.get("is_completed")]
        live_or_scheduled = [b for b in bouts if not b.get("is_completed")]
        if completed:
            # Split: completed bouts become a Past Events row for this event.
            past_events_today.append({
                "event_name": e.get("event_name", ""),
                "event_date": ev_date,
                "bouts":      completed,
                "is_past":    True,
                # Tag so the template can render a "Live event in progress"
                # hint — distinguishes from fully-finished historical events.
                "is_partial": bool(live_or_scheduled),
                "anchor":     "past-" + (ev_date or "").replace("-", "")[:8],
            })
        if live_or_scheduled:
            # Strip completed ones from the upcoming entry but keep the rest.
            e_upcoming = dict(e)
            e_upcoming["bouts"] = live_or_scheduled
            # Recompute is_live for the trimmed list
            e_upcoming["is_live"] = any(b.get("is_live") for b in live_or_scheduled)
            upcoming_events.append(e_upcoming)
        # If both are empty (no bouts at all) the event simply isn't shown.

    # Pull historical past events from data/past_events.json (persisted across
    # workflow runs). Predictions.json only contains upcoming events, so once
    # an event passes the entry is dropped — past_events.json keeps them.
    #
    # Prefer the persisted version: schedule-only "preview" bouts lack props,
    # probabilities, and bet analysis, so when both sources cover the same
    # event the persisted entry is the richer one. Fall back to today's
    # split-from-upcoming entries only when the event isn't in past_events.json.
    persisted_past = _load_persisted_past_events(fighter_meta, fighter_imgs)
    persisted_keys = {(str(ev["event_date"]), ev["event_name"]) for ev in persisted_past}
    past_events_today = [
        e for e in past_events_today
        if (str(e["event_date"]), e["event_name"]) not in persisted_keys
    ]
    past_events_today.extend(persisted_past)

    # Sort past events by date descending (most recent first)
    past_events = sorted(
        past_events_today,
        key=lambda e: str(e.get("event_date", "")),
        reverse=True,
    )
    # Try to enrich each past event's bouts with actual outcomes from the fights table
    past_events = _enrich_past_events(past_events)

    rendered = template.render(
        events=upcoming_events,
        past_events=past_events,
        top_bets=top_bets,
        bets_by_event=bets_by_event,
        event_filter_options=event_filter_options,
        portfolios_by_event=portfolios_by_event,
        event_carousel=event_carousel,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        n_bouts=len(predictions),
    )

    out_path = output_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    log.info(
        "Dashboard written to %s (%d bouts, %d events)",
        out_path,
        len(predictions),
        len(events),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
