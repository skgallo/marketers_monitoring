"""
views.py — builds the three dashboard views from prepared submissions
======================================================================
Pure pandas (no Streamlit), so every view can be tested locally:

    from rollup import prepare_submissions
    subs = prepare_submissions(raw_submissions)
    ledger, css = ride_ledger(subs, batch_codes)

Unit of analysis per view
-------------------------
Tab 1  Ride verification       one row per PLANNED batch code
Tab 2  Implementation fidelity one row per E1 form (the ride as carried out)
Tab 3  Enumerator performance  one row per enumerator (× week for trends)

Builders that feed styled tables return (table, css): css has the same shape
as the table and holds a CSS string per cell ("" = no style).
"""

import numpy as np
import pandas as pd

from rollup import (
    parse_batch_code, _parse_time, _diff_minutes, _code, pitch_pause_check,
    NORMAL_PITCH_MIN, PEDDLING_PITCH_MIN, PAUSE_END_TOL_MIN,
    DELAY_MAX_MIN, ANN_DUR_MIN, ANN_DUR_MAX, SIGNUP_DIFF_MAX,
    PITCH_TARGET_MIN, PITCH_TOL_MIN, RIDE_MIN_MIN, SIGNUP_MIN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEWS_VERSION = "2026-11-07n"   # app.py checks this to catch an out-of-date views.py

E1, SUP = "Enumerator 1", "Supervisor"
MISSING = "Missing/not assessed"

RED  = "color: #CC0000; font-weight: 600"
BLUE = "color: #0055AA; font-weight: 600"
GRAY = "color: #888888"
GREEN = "color: #1E7B34; font-weight: 600"
PURPLE = "color: #7B2CBF; font-weight: 600"
AMBER = "color: #B26B00; font-weight: 600"
GREEN_BG = "#D9F2E6"; AMBER_BG = "#FDF1D6"; RED_BG = "#FADBD3"   # cell backgrounds

TIME_GAP_MAX    = 2    # min — E1 vs supervisor checkpoint gap that needs review
PAIR_WINDOW_MIN = 20   # min — supervisor form paired with an E1 form if ride starts are this close
PERF_TARGET     = 0.80 # performance rates below this are shown in red

TREAT = {"S": "Sit down", "N": "Normal", "P": "Peddling"}

STATUS_RANK = {
    "Validity at risk": -1, "Wrong code selected": 0, "Not-allowed change": 1, "Duplicate E1": 2, "To confirm": 3,
    "Supervisor only": 4, "Not submitted": 5, "Done as planned": 6,
}
STATUS_STYLE = {
    "Validity at risk": RED, "Wrong code selected": RED, "Not-allowed change": RED, "Duplicate E1": RED, "To confirm": BLUE,
    "Supervisor only": BLUE, "Not submitted": GRAY, "Done as planned": GREEN,
}

C10_LABELS = {
    "0": "None", "1": "Bus/route interruption",
    "2": "Passenger disruption/confrontation",
    "3": "Marketer/pitch interruption",
    "4": "Team, equipment or materials",
    "5": "Recruitment paused/stopped", "-55": "Other",
}
# Disruptions outside the enumerator's control — count as an explanation for a flagged value
EXTERNAL_DISRUPTIONS = {"1", "2", "3", "5"}

COMPLETION_LABELS = {
    "1": "Completed — no disruption", "2": "Completed — with disruption",
    "3": "Not fully completed", "4": "Completion not confirmed",
}
from rollup import MM_REASON_LABELS
CONF_ITEM_LABELS = {"date": "date", "team": "team", "marketer": "marketer", "route": "route",
                    "ride": "ride number", "pitch": "pitch type"}
OTHER_WHEN = {"1": "before the announcement", "2": "during announcement/sign-up",
              "3": "during the pitch/pause", "4": "after the pitch/pause"}
OTHER_WHO = {"1": "preacher", "2": "seller", "3": "passenger/driver/apprentice", "-55": "other"}
OTHER_WHEN_TXT = lambda x: ", ".join(OTHER_WHEN.get(c, c) for c in str(x).split()) if x is not None and str(x) not in ("", "nan") else ""
MATCH_LABELS = {"1": "✓ Match", "2": "✗ Mismatch", "3": "Not verified"}
YESNO = {"1": "Yes", "0": "No"}
DISCREPANCY_SOURCE = {
    "1": "E1 recorded incorrectly",
    "2": "Supervisor recorded incorrectly",
    "3": "Both records have errors",
    "4": "Device clocks differed",
    "5": "Could not establish",
}
RECORDING_ISSUE = {"1": "E1 recording error", "0": "Not attributed to E1", "-1": "Unclear"}

# Supervisor behavioural observations (s-field → category)
S_CATEGORIES = {
    "Preparation":   ["s01", "s02"],
    "Form use":      ["s03"],
    "Announcement":  ["s04", "s05", "s06", "s07", "s08", "s09", "s10"],
    "Pax questions": ["s11", "s12"],
    "Pause period":  ["s15"],
    "Cards":         ["s16"],
    "Tickets":       ["s19"],
    "Payment":       ["s21", "s22"],
    "Close-out":     ["s23", "s24"],
    "Materials":     ["s25", "s26"],
}

# Flagged implementation values and the form field holding the explanation
ISSUE_DEFS = [
    ("flag_1_start_delay",  "Start delay",          "announcement_delay_explanation"),
    ("flag_2_ann_duration", "Ann./signup duration", "signup_duration_explanation"),
    ("flag_4a_pause_dur",   "Pause duration",       "pause_duration_explanation"),
    ("flag_4b_pitch_dur",   "Pitch duration",       "pause_duration_explanation"),
    ("flag_6_short_ride",   "Short ride",           "ride_duration_explanation"),
    ("flag_7_low_signup",   "Low sign-ups",         "final_comment_lowsignup"),
    # Nov form: marketer started the pitch > 2 min after sign-up ended (reference only,
    # not counted in the enumerator's timing score)
    ("flag_8_pitch_start_gap", "Pitch start gap",   "pitch_start_gap_explanation"),
]
NOT_E1_TIMING = {"flag_7_low_signup", "flag_8_pitch_start_gap"}
# Recording / data-quality problems (not protocol deviations)
RECORDING_DEFS = [
    ("flag_3_signup_discrep", "Recorded ≠ auto timing"),
    ("flag_4c_after_ride",    "Pitch/pause ends after ride end"),
    ("flag_chron_neg_delay",  "Announcement before ride start"),
    ("flag_chron_neg_ann",    "Negative auto duration"),
    ("flag_chron_neg_signup", "Negative recorded duration"),
    ("flag_chron_neg_pause",  "Negative pause duration"),
    ("flag_chron_neg_pitch",  "Negative pitch duration"),
]
CORE_FIELDS = ["announcement_start_delay", "signup_duration", "pause_duration",
               "ride_duration", "total_signups"]

# E1 vs supervisor timestamp checkpoints (label, fields — first present on both is used)
CHECKPOINTS = [
    ("Ride start",          ["b02_r"]),
    ("Ann. start",          ["b03"]),
    ("Sign-up end",         ["b04"]),          # Nov form
    ("Ann. end",            ["b05", "b06"]),
    ("Pitch/pause end",     ["b09", "b10"]),
    ("Ride end",            ["b11"]),
]

# Column names shared with app.py (styling) --------------------------------
COL_DELAY   = f"Start delay (<{DELAY_MAX_MIN} min)"
COL_REC     = f"Recorded ann./signup dur. ({ANN_DUR_MIN}–{ANN_DUR_MAX} min)"
COL_AUTO    = "Auto form-screen dur."
COL_DISCREP = f"Timing discrepancy (>{SIGNUP_DIFF_MAX} min)"
COL_PITCH   = "Pause/Pitch (min)"
COL_RIDE    = f"Ride dur. (≥{RIDE_MIN_MIN} min)"
COL_SIGNUPS = f"Signups (≥{SIGNUP_MIN})"
COL_GAP     = f"Pitch start gap (≤{DELAY_MAX_MIN} min)"

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _txt(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "nat") else s


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _is1(v) -> bool:
    return _code(v) == "1"


def _true(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and bool(v)


def fmt_code_date(s) -> str:
    """DDMMYYYY → '05 Oct 2026'."""
    s = _txt(s)
    if not s:
        return ""
    try:
        return pd.to_datetime(s, format="%d%m%Y").strftime("%d %b %Y")
    except Exception:
        return s


def fmt_day(ts) -> str:
    return ts.strftime("%d %b %Y") if pd.notna(ts) else ""


def fmt_val(v, nd=2) -> str:
    f = _num(v)
    return MISSING if np.isnan(f) else f"{f:.{nd}f}"


def fmt_rate(num, den) -> str:
    return "—" if not den else f"{num / den:.0%} ({int(num)}/{int(den)})"


def _empty_css(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame("", index=t.index, columns=t.columns)


def decode_c10(v) -> str:
    s = _txt(v)
    if not s:
        return "—"
    return "; ".join(C10_LABELS.get(c, c) for c in s.split())


# ---------------------------------------------------------------------------
# Batch code comparison for one form
# ---------------------------------------------------------------------------

def code_changes(row) -> list:
    """
    [(severity, text, part)] — severity 'design' (wrong ride for the design) or
    'confirm' (check with team); part is the batch-code component that differs
    (Treatment, Route, Date, Team, Mkt ID, Ride #, Batch code) — the name used in
    the reviews tab.
    """
    actual = _txt(row.get("actual_batch_code"))
    if not actual:
        return [("design", "No actual batch code", "Batch code")]
    if not _true(row.get("a_parse_ok")):
        return [("design", f"Actual code unreadable ({actual})", "Batch code")]
    if not _true(row.get("p_parse_ok")):
        return [("confirm", "Planned code unreadable", "Batch code")]

    g = lambda k: _txt(row.get(k))
    out = []
    if g("p_treatment") != g("a_treatment"):
        out.append(("design", f"Treatment {TREAT.get(g('p_treatment'), g('p_treatment'))}"
                              f"→{TREAT.get(g('a_treatment'), g('a_treatment'))}", "Treatment"))
    if g("p_corridor") != g("a_corridor"):
        out.append(("design", f"Corridor {g('p_route')}→{g('a_route')}", "Route"))
    elif g("p_route") != g("a_route"):
        out.append(("design", f"Direction {g('p_route')}→{g('a_route')}", "Route"))
    if g("p_date") != g("a_date"):
        out.append(("design", f"Date {fmt_code_date(g('p_date'))}→{fmt_code_date(g('a_date'))}", "Date"))
    if g("p_team") != g("a_team"):
        out.append(("allowed", f"Team {g('p_team')}→{g('a_team')} (allowed)", "Team"))
    if g("p_mkt_id") != g("a_mkt_id"):
        out.append(("allowed", f"Mkt ID {g('p_mkt_id')}→{g('a_mkt_id')} (allowed)", "Mkt ID"))
    if g("p_ride_num") != g("a_ride_num"):
        out.append(("design", f"Ride # {g('p_ride_num')}→{g('a_ride_num')}", "Ride #"))
    return out


def _changes_text(ch: list) -> str:
    return "; ".join(c[1] for c in ch) if ch else "As planned"


# ---------------------------------------------------------------------------
# Reviews — decisions you record about flagged differences
# ---------------------------------------------------------------------------
# A "reviews" tab in the Google Sheet, one row per decision:
#   batch_code | issue | decision | note | reviewed_by | date
# issue    : Route, Treatment, Date, Team, Mkt ID, Ride #, Batch code,
#            Wrong code, Duplicate E1, Missing form — or All
# decision : Approved   — a planned/accepted change: no longer a flag
#            Fix needed — a mistake to correct in the cleaning code (shown as "Correct in cleaning")
#            Fixed      — the mistake has been dealt with / data corrected: no longer a flag
# Later rows override earlier ones for the same batch_code + issue.

ISSUE_ALIASES = {
    "route": "Route", "corridor": "Route", "direction": "Route",
    "treatment": "Treatment", "date": "Date", "team": "Team",
    "mkt id": "Mkt ID", "mkt_id": "Mkt ID", "marketer": "Mkt ID", "marketer id": "Mkt ID",
    "ride #": "Ride #", "ride": "Ride #", "ride number": "Ride #", "ride_num": "Ride #",
    "batch code": "Batch code", "code": "Batch code",
    "wrong code": "Wrong code", "wrong code selected": "Wrong code",
    "duplicate": "Duplicate E1", "duplicate e1": "Duplicate E1",
    "missing form": "Missing form", "supervisor only": "Missing form", "not submitted": "Missing form",
    "all": "All", "": "All",
}
DECISION_ALIASES = {
    "approved": "Approved", "approve": "Approved", "planned": "Approved", "planned change": "Approved",
    "ok": "Approved", "accepted": "Approved",
    "fix needed": "Fix needed", "fix": "Fix needed", "communicate": "Fix needed", "to fix": "Fix needed",
    "fixed": "Fixed", "resolved": "Fixed", "corrected": "Fixed", "done": "Fixed",
    # labels shown in the dashboard (what gets written to the reviews tab)
    "ok – planned/allowed": "Approved", "ok - planned/allowed": "Approved",
    "correct in cleaning": "Fix needed",
}
# Batch-code changes that are NOT allowed (the E1 must stop and call the supervisor)
DESIGN_PARTS  = {"Treatment", "Route", "Batch code", "Date", "Ride #"}
# Allowed changes the E1 did not report to the supervisor (Nov form: mm_ok_marketer / mm_ok_team = No)
CONFIRM_PARTS = {"Unreported change"}
# Changes the field team is allowed to make on the day — shown in "What changed", never flagged
ALLOWED_PARTS = {"Mkt ID", "Team"}


def review_map(reviews: pd.DataFrame | None) -> dict:
    """{planned code: {issue: {'decision', 'note', 'by'}}} from the reviews tab."""
    out = {}
    if reviews is None or reviews.empty:
        return out
    cols = {c.strip().lower().replace(" ", "_"): c for c in reviews.columns}
    get = lambda r, k: _txt(r.get(cols[k])) if k in cols else ""
    if "batch_code" not in cols or "decision" not in cols:
        return out
    for _, r in reviews.iterrows():
        code = get(r, "batch_code")
        dec = DECISION_ALIASES.get(get(r, "decision").lower())
        if not code or not dec:
            continue
        issue = ISSUE_ALIASES.get(get(r, "issue").lower(), get(r, "issue"))
        out.setdefault(code, {})[issue] = {"decision": dec, "note": get(r, "note"),
                                           "by": get(r, "reviewed_by"),
                                           "field": get(r, "field"), "value": get(r, "correct_value")}
    return out


def _review_for(rmap: dict, code: str, part: str):
    d = rmap.get(code, {})
    return d.get(part) or d.get("All")


def _names(series) -> str:
    vals = [v for v in dict.fromkeys(_txt(x) for x in series) if v]
    return ", ".join(vals)


# ---------------------------------------------------------------------------
# TAB 1 — Ride verification
# ---------------------------------------------------------------------------

def planned_codes(batch_codes: pd.DataFrame) -> list:
    return list(dict.fromkeys(
        _txt(c) for c in batch_codes.get("batch_code", pd.Series(dtype=str)) if _txt(c)
    ))


def _ride_lookup(batch_codes: pd.DataFrame) -> dict:
    """{(date, team, ride #): planned code} — only keys that identify exactly one planned ride."""
    keys = {}
    for code in planned_codes(batch_codes):
        p = parse_batch_code(code)
        if p["parse_ok"]:
            k = (pd.to_datetime(p["date"], format="%d%m%Y", errors="coerce"), p["team"], p["ride_num"])
            keys.setdefault(k, []).append(code)
    return {k: v[0] for k, v in keys.items() if len(v) == 1}


def _only_team_or_mkt_differs(selected: str, actual: str) -> bool:
    """True when the selected planned code and the actual code differ only in team and/or marketer."""
    p, a = parse_batch_code(selected), parse_batch_code(actual)
    if not (p["parse_ok"] and a["parse_ok"]):
        return False
    same_core = all(p[k] == a[k] for k in ("date", "route", "ride_num", "treatment"))
    return same_core and (p["team"] != a["team"] or p["mkt_id"] != a["mkt_id"])


def effective_codes(subs: pd.DataFrame, batch_codes: pd.DataFrame) -> pd.DataFrame:
    """
    Which planned ride each form actually belongs to.

    The form records ride date (a01_r), team (a02a_r) and ride number (a04_r)
    separately from the planned code picked from the list. If those identify a
    different planned ride than the one picked, the form was filed under the
    wrong code: likely_code is the ride it belongs to and misfiled is True.
    eff_code = likely_code when known, else the code selected.
    """
    lookup = _ride_lookup(batch_codes)
    likely = []
    for _, r in subs.iterrows():
        team, ride = _code(r.get("a02a_r")), _code(r.get("a04_r"))
        likely.append(lookup.get((r["ride_date"], team, ride)) if team and ride and pd.notna(r["ride_date"]) else None)
    out = pd.DataFrame({"likely_code": likely}, index=subs.index)
    sel = subs["batch_code"].map(_txt)
    # Another team doing this ride is allowed: if the code selected and the actual code agree on
    # everything except team / marketer, the form belongs to the code selected
    swap = [_only_team_or_mkt_differs(s_, a_) for s_, a_ in
            zip(sel, subs.get("actual_batch_code", pd.Series(None, index=subs.index)).map(_txt))]
    out.loc[swap, "likely_code"] = sel[swap]
    out["misfiled"] = out["likely_code"].notna() & (out["likely_code"] != sel)
    out["eff_code"] = out["likely_code"].where(out["likely_code"].notna(), sel)
    return out


RIDE_FIELDS = {"batch_code_pre", "batch_code", "actual_batch_code", "a01_r", "a02a_r", "a04_r"}


def ride_identity(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                  reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    For each form: the planned ride it really belongs to and the actual code, after
    the reviewer's corrections. Order of evidence for the planned ride:
      1. a 'Correct in cleaning' / 'Corrected' value for batch_code_pre on this form
      2. date + team + ride # on the form (with any corrections to a01_r/a02a_r/a04_r)
      3. the planned code selected on the form
    Columns: planned, actual, selected, date, team, ride, note.
    """
    corr = {}
    ct = corrections_table(reviews)
    if not ct.empty:
        for _, c in ct.iterrows():
            if DECISION_ALIASES.get(_txt(c["decision"]).lower()) in ("Fix needed", "Fixed"):
                f = _txt(c["field"]).lower()
                if f in RIDE_FIELDS:
                    corr.setdefault(_txt(c["form_id"]), {})[f] = _txt(c["correct_value"])
    lookup = _ride_lookup(batch_codes) if batch_codes is not None else {}
    rows = []
    for i, r in subs.iterrows():
        cf = corr.get(_txt(r.get("form_id") or r.get("KEY")), {})
        selected = _txt(r.get("batch_code"))
        actual = cf.get("batch_code") or cf.get("actual_batch_code") or _txt(r.get("actual_batch_code"))
        date = r.get("ride_date")
        if "a01_r" in cf:
            date = pd.to_datetime(cf["a01_r"], format="mixed", errors="coerce")
        team = _code(cf.get("a02a_r", r.get("a02a_r")))
        ride = _code(cf.get("a04_r", r.get("a04_r")))
        likely = lookup.get((date, team, ride)) if team and ride and pd.notna(date) else None
        if _only_team_or_mkt_differs(selected, actual):
            likely = selected                     # another team doing this ride is allowed
        if cf.get("batch_code_pre"):
            planned, note = cf["batch_code_pre"], f"Selected {selected} → corrected (review)"
        elif likely and likely != selected:
            planned, note = likely, f"Filed under {selected} (form says ride {ride})"
        else:
            planned, note = selected, ""
        if planned == selected and note.startswith("Selected"):
            note = ""
        p = parse_batch_code(planned)
        if p.get("parse_ok"):
            p_date = pd.to_datetime(p["date"], format="%d%m%Y", errors="coerce")
            if note or not (team and ride and pd.notna(date)):
                team, ride, date = p["team"], p["ride_num"], p_date
            elif ride != p["ride_num"] or date != p_date:   # a team change alone is allowed
                # Not in the plan list, but the form and the code selected disagree:
                # show the ride the form describes and say so
                note = (f"Selected {selected} (team {p['team']}, ride {p['ride_num']}) "
                        f"but form says team {team}, ride {ride} — check")
        rows.append({"planned": planned, "actual": actual, "selected": selected,
                     "date": date, "team": team, "ride": ride, "note": note})
    return pd.DataFrame(rows, index=subs.index)


def _ride_sort_key(ident: pd.DataFrame, subs: pd.DataFrame) -> list:
    num = lambda x: int(x) if str(x).isdigit() else 999
    return sorted(ident.index, key=lambda i: (
        ident.at[i, "date"] if pd.notna(ident.at[i, "date"]) else pd.Timestamp.max,
        num(ident.at[i, "team"]), num(ident.at[i, "ride"]),
        subs.at[i, "submitted_at"] if pd.notna(subs.at[i, "submitted_at"]) else pd.Timestamp.max))


def code_changes_vs(row, planned: str) -> list:
    """code_changes() against a given planned code instead of the one selected on the form."""
    r = dict(row)
    for k, val in parse_batch_code(planned).items():
        r[f"p_{k}"] = val
    return code_changes(r)


def _role_short(role):
    return "E1" if role == E1 else "Supervisor"


def ride_ledger(subs: pd.DataFrame, batch_codes: pd.DataFrame,
                reviews: pd.DataFrame | None = None):
    """
    One row per planned ride. Forms are assigned to the ride they describe
    (date + team + ride # on the form), so a form filed under the wrong planned
    code is counted on its real ride and flagged on both rows. Status lists every
    issue that applies, worst first. Returns (table, css).
    """
    eff = effective_codes(subs, batch_codes)
    s2 = subs.assign(_eff=eff["eff_code"], _likely=eff["likely_code"], _misfiled=eff["misfiled"])
    by_eff = {k: d for k, d in s2.groupby("_eff")}
    by_sel = {k: d for k, d in s2.groupby("batch_code")}
    empty = s2.iloc[0:0]

    rmap = review_map(reviews)

    rows = []
    for code in planned_codes(batch_codes):
        p = parse_batch_code(code)
        d = by_eff.get(code, empty)                       # forms that describe this ride
        e1 = d[d["role"] == E1].sort_values("submitted_at")
        sp = d[d["role"] == SUP].sort_values("submitted_at")
        came_in = d[d["_misfiled"]]                       # belong here, filed under another code
        sel = by_sel.get(code, empty)
        went_out = sel[sel["_misfiled"]]                  # filed here, belong to another ride

        basis = e1 if len(e1) else sp
        changes = [code_changes_vs(r, code) for _, r in basis.iterrows()]

        # Every issue on this ride, by the name used in the reviews tab
        parts = {c[2] for ch in changes for c in ch if c[0] != "allowed"}
        # Nov form: an allowed change (team / marketer) the E1 says was NOT reported to the supervisor
        unreported = []
        for _, f in basis.iterrows():
            for flag, ok, chg in [("mm_team", "mm_ok_team", "team"), ("mm_marketer", "mm_ok_marketer", "marketer")]:
                if _code(f.get(flag)) == "1" and _code(f.get(ok)) == "0":
                    unreported.append(f"{chg} change not reported to supervisor ({_txt(f['person'])})")
        if unreported:
            parts.add("Unreported change")
        # Another marketer marketed on the bus (E1 or supervisor form) — the ride may not be valid
        other_mkt = [f for _, f in d.iterrows() if _code(f.get("other_marketer")) == "1"]
        if other_mkt:
            parts.add("Other marketer")
        # Sit down: the supervisor saw the team's own marketer speak or sell during the pause (s27 = No)
        mkt_spoke = [f for _, f in d.iterrows() if _code(f.get("s27")) == "0"]
        if mkt_spoke:
            parts.add("Marketer spoke in pause")
        if len(came_in):
            parts.add("Wrong code")
        if len(e1) > 1:
            parts.add("Duplicate E1")
        if e1.empty:
            parts.add("Missing form")

        # Review decisions — they never remove a flag; they mark it as reviewed
        decisions, review_txt = {}, []
        for part in sorted(parts):
            rv = _review_for(rmap, code, part)
            if rv is None:
                continue
            decisions[part] = rv["decision"]
            note = f" — {rv['note']}" if rv["note"] else ""
            by = f" ({rv['by']})" if rv["by"] else ""
            icon = "🧹" if rv["decision"] == "Fix needed" else "✓"
            corr = f" [{rv['field']} → {rv['value']}]" if rv.get("value") else ""
            review_txt.append(f"{icon} {part}: {DECISION_LABELS[rv['decision']]}{note}{corr}{by}")
        undecided = sorted(p_ for p_ in parts if p_ not in decisions and p_ != "Missing form")
        fix_needed = "Fix needed" in decisions.values()

        # Which issue parts produce which status
        tag_parts = {}
        if parts & {"Other marketer", "Marketer spoke in pause"}:
            tag_parts["Validity at risk"] = parts & {"Other marketer", "Marketer spoke in pause"}
        if "Wrong code" in parts:
            tag_parts["Wrong code selected"] = {"Wrong code"}
        if parts & DESIGN_PARTS:
            tag_parts["Not-allowed change"] = parts & DESIGN_PARTS
        if "Duplicate E1" in parts:
            tag_parts["Duplicate E1"] = {"Duplicate E1"}
        if parts & CONFIRM_PARTS and not parts & DESIGN_PARTS:
            tag_parts["To confirm"] = parts & CONFIRM_PARTS
        if "Missing form" in parts:
            tag_parts["Supervisor only" if not sp.empty else "Not submitted"] = {"Missing form"}
        if not tag_parts:
            tag_parts["Done as planned"] = set()
        tags = sorted(tag_parts, key=STATUS_RANK.get)
        # A status is "reviewed" (✓) when every issue behind it has a decision
        tag_reviewed = {t_: bool(ps) and all(x in decisions for x in ps) for t_, ps in tag_parts.items()}

        # Is each status still an open issue? OK (all approved) and corrected issues are
        # resolved: the flag stays visible with a ✓, but it no longer counts as an issue.
        def _state(ps):
            ds = [decisions.get(x) for x in ps]
            if ps and all(d_ == "Approved" for d_ in ds):
                return "ok"
            if ps and all(d_ in ("Approved", "Fixed") for d_ in ds):
                return "corrected"
            if ps and all(d_ is not None for d_ in ds):
                return "pending"      # reviewed: correction still to make in cleaning
            return "open"             # not (fully) reviewed yet
        tag_state = {t_: _state(ps) for t_, ps in tag_parts.items()}
        open_tags = [t_ for t_ in tags if t_ != "Done as planned" and tag_state[t_] in ("open", "pending")]
        undecided_tags = [t_ for t_ in tags if t_ != "Done as planned" and tag_state[t_] == "open"]
        resolved_parts = {x for x, d_ in decisions.items() if d_ in ("Approved", "Fixed")}

        relevant = {p_ for p_ in parts if p_ != "Missing form" or p_ in decisions}
        if not relevant:
            review_status = "—"
        elif fix_needed:
            review_status = "🧹 Correct in cleaning" + (" · not all reviewed" if undecided else "")
        elif undecided and decisions:
            review_status = "Partly reviewed"
        elif undecided:
            review_status = "Not reviewed"
        else:
            kinds = sorted({{"Approved": "OK", "Fixed": "corrected"}[d_] for d_ in decisions.values()})
            review_status = "✓ Reviewed (" + " / ".join(kinds) + ")"

        # Code selection notes
        notes = []
        for _, f in came_in.iterrows():
            notes.append(f"{_role_short(f['role'])} form ({_txt(f['person'])}, team {_code(f.get('a02a_r'))} "
                         f"ride {_code(f.get('a04_r'))}) was filed under {_txt(f['batch_code'])}")
        moved = [f"{_role_short(f['role'])} form by {_txt(f['person'])} chose this code but is "
                 f"{fmt_day(f['ride_date'])}, team {_code(f.get('a02a_r'))} ride {_code(f.get('a04_r'))} "
                 f"→ counted under {f['_likely']}" for _, f in went_out.iterrows()]
        if len(e1) > 1:
            people = [_txt(x) for x in e1["person"]]
            if len(set(people)) == 1:
                notes.append(f"{people[0]} submitted {len(e1)} E1 forms for this ride — keep one")
            else:
                notes.append(f"{len(e1)} E1 forms by different enumerators ({', '.join(people)}) — "
                             f"two E1s on one ride, or a wrong code")

        # One entry per issue (per form for wrong codes) — feeds the decisions editor
        part_text = {}
        for ch in changes:
            for _, txt_, part in ch:
                part_text.setdefault(part, []).append(txt_)
        if unreported:
            part_text["Unreported change"] = unreported
        if mkt_spoke:
            part_text["Marketer spoke in pause"] = [
                "Supervisor: the marketer spoke or sold during the Sit down pause"
                + (f" ({_txt(f.get('s27a'))})" if _txt(f.get('s27a')) else "") for f in mkt_spoke]
        if other_mkt:
            part_text["Other marketer"] = [
                f"Another marketer marketed on the bus ({_role_short(f['role'])} form, {_txt(f['person'])}"
                + (f": {_txt(f.get('other_marketer_what'))}" if _txt(f.get('other_marketer_what')) else "")
                + (f"; when: {OTHER_WHEN_TXT(f.get('other_marketer_when'))}" if _txt(f.get('other_marketer_when')) else "")
                + ")" for f in other_mkt]
        issues = []
        ride_lbl = (f"{fmt_code_date(p['date'])} · team {p['team']} · ride {p['ride_num']}"
                    if p["parse_ok"] else code)
        for part in sorted(parts):
            rv = _review_for(rmap, code, part)
            base = {"code": code, "ride": ride_lbl, "part": part,
                    "current": DECISION_LABELS[rv["decision"]] + (f" — {rv['note']}" if rv["note"] else "") if rv else "",
                    "form_id": "", "field": "", "value": ""}
            if part == "Wrong code":
                for _, f in came_in.iterrows():
                    issues.append({**base,
                                   "detail": f"{_role_short(f['role'])} form by {_txt(f['person'])} (team "
                                             f"{_code(f.get('a02a_r'))}, ride {_code(f.get('a04_r'))}) selected "
                                             f"{_txt(f['batch_code'])}",
                                   "form_id": _txt(f.get("form_id")), "field": "batch_code_pre", "value": code})
                continue
            if part == "Duplicate E1":
                for k_, (_, f) in enumerate(e1.iterrows(), 1):
                    t_ = f["submitted_at"].strftime("%d %b %H:%M") if pd.notna(f["submitted_at"]) else "?"
                    issues.append({**base,
                                   "detail": f"E1 form {k_}/{len(e1)} by {_txt(f['person'])}, submitted {t_}, "
                                             f"actual code {_txt(f['actual_batch_code'])}. If a supervisor filled "
                                             f"this in: Field supervisor_form, Correct value = their name. "
                                             f"To ignore it: Field drop_form, Correct value 1",
                                   "form_id": _txt(f.get("form_id")), "field": "drop_form", "value": ""})
                continue
            if part == "Missing form":
                detail = "Supervisor form only — no E1 form" if not sp.empty else "No form submitted"
            else:
                detail = "; ".join(dict.fromkeys(part_text.get(part, []))) or part
            issues.append({**base, "detail": detail})

        def _ann(ch):
            if not ch:
                return "As planned"
            return "; ".join(c[1] + {"Approved": " (OK)", "Fixed": " (corrected)"}.get(decisions.get(c[2]), "")
                             for c in ch)
        if len(basis) > 1:
            what = " | ".join(f"Form {i + 1}: {_ann(ch)}" for i, ch in enumerate(changes))
        elif changes:
            what = _ann(changes[0])
        else:
            what = ""
        open_what = "; ".join(dict.fromkeys([c[1] for ch in changes for c in ch
                                             if c[0] != "allowed" and c[2] not in resolved_parts]
                                            + (unreported if "Unreported change" not in resolved_parts else [])
                                            + (part_text.get("Other marketer", []) if "Other marketer" not in resolved_parts else [])
                                            + (part_text.get("Marketer spoke in pause", []) if "Marketer spoke in pause" not in resolved_parts else [])))

        if len(e1):
            names = _names(e1["person"])
            e1_col = f"✓ {names}" + (f" (×{len(e1)})" if len(e1) > 1 else "")
        else:
            observed = _names(sp["e1_name"])
            e1_col = "✗ none" + (f" (supervisor observed {observed})" if observed else "")
        sup_col = f"✓ {_names(sp['person'])}" if len(sp) else "✗ none"

        # Actual components (what happened), from the forms describing this ride.
        # E1 forms take precedence; supervisor forms are used when there is no E1 form.
        def actual(col, fmt=lambda x: x):
            vals = [fmt(_txt(x)) for x in basis.get(col, pd.Series(dtype=str))] if len(basis) else []
            vals = [x for x in dict.fromkeys(vals) if x]
            return " / ".join(vals) if vals else "—"

        def differs(col, planned_val):
            vals = {_txt(x) for x in basis.get(col, pd.Series(dtype=str))} - {""} if len(basis) else set()
            return bool(vals) and vals != {planned_val or ""}

        # Nov form: the E1's own explanation when the ride differs from the planned code
        e1_says = []
        for _, f in pd.concat([basis, came_in]).drop_duplicates(subset="form_id").iterrows():
            bits = []
            mm = _code(f.get("mm_reason"))          # earlier Nov version
            if mm:
                bits.append(MM_REASON_LABELS.get(mm, mm))
            if _code(f.get("mm_real")) == "1":       # current Nov version
                bits.append("confirmed this is what really happened")
            for ok, chg in [("mm_ok_team", "team change"), ("mm_ok_marketer", "marketer change"),
                            ("mm_ok_date", "date change")]:
                c_ = _code(f.get(ok))
                if c_ in ("0", "1"):
                    bits.append(f"{chg} {'reported to' if c_ == '1' else 'NOT reported to'} supervisor")
            if _code(f.get("ride_invalid_flag")) == "1":
                who = {"1": "supervisor", "2": "research team"}.get(_code(f.get("mm_contacted_who")), "")
                bits.append("NOT-ALLOWED change: " + ({"1": f"contacted {who}".strip(),
                                                         "2": "tried but could not reach supervisor"}
                                                        .get(_code(f.get("mm_contacted")), "contact not recorded")))
            wrong = _txt(f.get("conf_wrong"))
            if wrong:
                bits.append("marked wrong at confirmation: " + ", ".join(
                    CONF_ITEM_LABELS.get(w, w) for w in wrong.split()))
            if bits:
                e1_says.append(f"{_txt(f['person'])}: " + "; ".join(bits))

        # Implementation checks for Today (from the E1 form(s)): start conditions and timing targets
        checks = []
        for _, f in e1.iterrows():
            txt_, css_ = _start_check(f)
            if css_ in (RED, AMBER):
                checks.append(txt_)
            if _code(f.get("sp_when")) == "3":
                checks.append("Announcement started AFTER the strategic point")
            ann_ = _ann_signup(f)
            if not np.isnan(ann_) and not (ANN_DUR_MIN <= round(ann_, 1) <= ANN_DUR_MAX):
                checks.append(f"Announcement + sign-up {ann_:.1f} min")
            gap_ = _diff_minutes(_parse_time(f.get("b04")),
                                 _parse_time(f.get("b05")) if pd.notna(_parse_time(f.get("b05"))) else _parse_time(f.get("b06")))
            if not np.isnan(gap_) and round(gap_, 1) > DELAY_MAX_MIN:
                checks.append(f"Pitch/pause started {gap_:.1f} min after sign-up ended")
            if _code(f.get("other_speaker")) == "1":
                checks.append("Someone else spoke to the passengers ("
                              + (", ".join(OTHER_WHO.get(c, c) for c in _txt(f.get("other_speaker_who")).split()) or "?") + ")")

        last = d["submitted_at"].max() if not d.empty else pd.NaT
        rows.append({
            "Status":         " + ".join(f"{t_} ✓ OK" if tag_state[t_] == "ok"
                                         else f"{t_} ✓ corrected" if tag_state[t_] == "corrected"
                                         else t_ for t_ in tags),
            "Review status":  review_status,
            "Planned code":   code,
            "Actual code(s)": ", ".join(_txt(c) for c in basis["actual_batch_code"]) or "—",
            "Date":           actual("a_date", fmt_code_date),
            "Team":           actual("a_team"),
            "Mkt ID":         actual("a_mkt_id"),
            "Route":          actual("a_route"),
            "Ride #":         actual("a_ride_num"),
            "Treatment":      actual("a_treatment", lambda x: TREAT.get(x, x)),
            "E1 form":        e1_col,
            "Supervisor form": sup_col,
            "Code selection": "; ".join(notes + [f"({m})" for m in moved]) or "—",
            "_code_flag":     bool(notes and any("filed under" in n for n in notes)),
            "What changed":   what,
            "E1 explanation": " | ".join(e1_says) or "—",
            "Last submitted": last.strftime("%d %b %H:%M") if pd.notna(last) else "",
            "Review notes":   "; ".join(review_txt) or "—",
            "Needs decision on": ", ".join(undecided) or "—",
            "_needs_decision": bool(undecided),
            "_issues":        issues,
            "_checks":        checks,
            "_fix_needed":    fix_needed,
            "_tags":          tags,
            "_open_tags":     open_tags,
            "_undecided_tags": undecided_tags,
            "_open_what":     open_what,
            "_resolved_parts": resolved_parts,
            "_corridor":      p["corridor"],
            "_date":          pd.to_datetime(p["date"], format="%d%m%Y", errors="coerce"),
            "_team":          p["team"] or "",
            "_ride":          p["ride_num"] or "",
            "_rank":          STATUS_RANK[tags[0]],
            # which actual components differ from plan (for colouring)
            "_diff": {
                "Date":   differs("a_date", p["date"]),
                "Team":   differs("a_team", p["team"]),
                "Mkt ID": differs("a_mkt_id", p["mkt_id"]),
                "Ride #": differs("a_ride_num", p["ride_num"]),
                "Treatment": differs("a_treatment", p["treatment"]),
                "Route":  differs("a_route", p["route"]),
            },
        })

    t = pd.DataFrame(rows)
    if t.empty:
        return t, _empty_css(t)
    t = t.sort_values(["_date", "_team", "_ride", "Planned code"]).reset_index(drop=True)
    css = _empty_css(t)
    not_sub = t["_tags"].map(lambda x: x == ["Not submitted"])
    css.loc[not_sub] = GRAY
    # Status colour follows the worst OPEN issue; rides whose issues are all OK/corrected are green
    # red = not reviewed yet · purple = reviewed, correction still to make · green = resolved
    css["Status"] = [STATUS_STYLE.get(u[0], RED) if u else PURPLE if o
                     else STATUS_STYLE.get(a[0], GREEN) if a[0] in ("Not submitted", "Done as planned") else GREEN
                     for u, o, a in zip(t["_undecided_tags"], t["_open_tags"], t["_tags"])]
    # Actual components: red = wrong ride for the design (treatment, route); blue = confirm;
    # green = the difference was marked OK or corrected
    for col, style in [("Treatment", RED), ("Route", RED), ("Date", RED), ("Ride #", RED),
                       ("Team", BLUE)]:
        diff = t["_diff"].map(lambda dd: dd[col])
        ok = t["_resolved_parts"].map(lambda rp: col in rp)
        css.loc[diff & ~ok, col] = style
        css.loc[diff & ok, col] = GREEN
    css.loc[t["Code selection"] != "—", "Code selection"] = GRAY
    css.loc[t["_code_flag"], "Code selection"] = RED
    css.loc[t["_code_flag"] & ~t["_undecided_tags"].map(lambda x: "Wrong code selected" in x),
            "Code selection"] = PURPLE
    css.loc[t["E1 form"].str.startswith("✗") & ~not_sub, "E1 form"] = RED
    css.loc[t["E1 form"].str.contains("×", regex=False), "E1 form"] = RED
    css.loc[t["_tags"].map(lambda x: "Not-allowed change" in x or "To confirm" in x), "What changed"] = GREEN
    css.loc[t["_open_tags"].map(lambda x: "To confirm" in x or "Not-allowed change" in x), "What changed"] = PURPLE
    css.loc[t["_undecided_tags"].map(lambda x: "To confirm" in x), "What changed"] = BLUE
    css.loc[t["_undecided_tags"].map(lambda x: "Not-allowed change" in x), "What changed"] = RED
    css.loc[t["Review status"].str.startswith("✓"), "Review status"] = GREEN
    css.loc[t["Review status"].isin(["Not reviewed", "Partly reviewed"]), "Review status"] = RED
    css.loc[t["_fix_needed"], "Review status"] = PURPLE
    css.loc[t["_needs_decision"], "Needs decision on"] = RED
    return t, css


# Internal decision → label shown in the dashboard and written to the reviews tab
DECISION_LABELS = {"Approved": "OK – planned/allowed",
                   "Fix needed": "Correct in cleaning",
                   "Fixed": "Corrected"}
DECISION_OPTIONS = [""] + list(DECISION_LABELS.values())


def pending_decisions(ledger: pd.DataFrame, include_reviewed: bool = False) -> pd.DataFrame:
    """One editable row per open issue (optionally also reviewed ones) for the decisions editor."""
    rows = []
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    for issues in ledger["_issues"]:
        for it in issues:
            if it["part"] == "Missing form" and it["detail"] == "No form submitted":
                continue          # nothing to decide until a ride is due — chase instead
            if it["current"] and not include_reviewed:
                continue
            rows.append({
                "Planned code": it["code"], "Ride": it["ride"], "Issue": it["part"],
                "Details": it["detail"], "Current decision": it["current"] or "—",
                "Decision": "", "Note": "", "Field": it["field"], "Correct value": it["value"],
                "_form_id": it["form_id"],
            })
    return pd.DataFrame(rows)


def corrections_table(reviews: pd.DataFrame | None) -> pd.DataFrame:
    """Latest recorded correction per form + field — what the cleaning do-file applies."""
    if reviews is None or reviews.empty:
        return pd.DataFrame()
    r = reviews.copy()
    r.columns = [c.strip().lower().replace(" ", "_") for c in r.columns]
    for c in ["form_id", "field", "correct_value", "batch_code", "issue", "decision", "note", "reviewed_by", "timestamp"]:
        if c not in r.columns:
            r[c] = None
    r = r[r["correct_value"].map(_txt) != ""]
    r = r[r["form_id"].map(_txt) != ""]
    if r.empty:
        return pd.DataFrame()
    r = r.drop_duplicates(subset=["form_id", "field"], keep="last")
    return r[["form_id", "field", "correct_value", "batch_code", "issue", "decision", "note",
              "reviewed_by", "timestamp"]].reset_index(drop=True)


# Corrections about WHO filled in a form are applied straight away in the dashboard
# (so rides, supervisor pairing and enumerator performance are attributed correctly).
# Batch-code corrections are NOT applied here: they stay visible as flags.
#   supervisor_form = <supervisor name>  → the form was filled in by that supervisor
#   role_type       = 0 / 1 (or Supervisor / E1)
#   supervisor      = <supervisor name or code>
#   a02_1           = <Enumerator 1 name or roster code>
#   drop_form       = 1  → ignore this form (e.g. a test or a true duplicate)
LIVE_FIELDS = {"supervisor_form", "role_type", "supervisor", "a02_1", "drop_form"}


def apply_live_corrections(raw: pd.DataFrame, reviews: pd.DataFrame | None):
    """Return (corrected copy of the raw submissions, list of applied corrections as text)."""
    corr = corrections_table(reviews)
    if corr.empty or raw is None or raw.empty or "KEY" not in raw.columns:
        return raw, []
    out = raw.copy()
    applied = []
    for _, c in corr.iterrows():
        field = _txt(c["field"]).lower()
        val = _txt(c["correct_value"])
        dec = DECISION_ALIASES.get(_txt(c["decision"]).lower())
        if field not in LIVE_FIELDS or dec not in ("Fix needed", "Fixed") or not val:
            continue
        m = out["KEY"].map(_txt) == _txt(c["form_id"])
        if m.sum() != 1:
            continue

        def set_supervisor(name_or_code):
            # A name is stored as "Other: <name>" so it reads the same whatever form
            # version the submission came from (the supervisor codes differ by form).
            if name_or_code.lstrip("-").isdigit():
                out.loc[m, "supervisor"] = name_or_code
            else:
                out.loc[m, "supervisor"] = "-55"
                out.loc[m, "supervisor_oth"] = name_or_code.strip().title()

        if field == "drop_form":
            if val == "1":
                out = out[~m]
                applied.append(f"form {_txt(c['form_id'])[-8:]} ignored")
            continue
        if field == "supervisor_form":
            out.loc[m, "role_type"] = "0"
            set_supervisor(val)
        elif field == "role_type":
            out.loc[m, "role_type"] = {"supervisor": "0", "e1": "1", "enumerator 1": "1"}.get(val.lower(), val)
        elif field == "supervisor":
            set_supervisor(val)
        elif field == "a02_1":
            if val.lstrip("-").isdigit():
                out.loc[m, "a02_1"] = val
            else:
                same = raw.loc[raw.get("a02_1_lab", pd.Series(dtype=str)).map(_txt).str.lower() == val.lower(), "a02_1"]
                out.loc[m, "a02_1"] = _txt(same.iloc[0]) if len(same) else "-55"
                out.loc[m, "a02_1_lab"] = val
        applied.append(f"form {_txt(c['form_id'])[-8:]}: {field} → {val}")
    return out, applied


def fixes_to_communicate(subs: pd.DataFrame, batch_codes: pd.DataFrame,
                         reviews: pd.DataFrame | None) -> pd.DataFrame:
    """One row per 'Correct in cleaning' decision still open, with the E1(s) on that ride."""
    rmap = review_map(reviews)
    if not rmap:
        return pd.DataFrame()
    eff = effective_codes(subs, batch_codes)
    s2 = subs.assign(_eff=eff["eff_code"])
    rows = []
    for code, issues in rmap.items():
        for issue, rv in issues.items():
            if rv["decision"] != "Fix needed":
                continue
            d = s2[(s2["_eff"] == code) & (s2["role"] == E1)]
            if d.empty:
                d = s2[s2["_eff"] == code]
            p = parse_batch_code(code)
            rows.append({"Ride date": fmt_code_date(p["date"]), "Planned code": code,
                         "Enumerator 1": _names(d["e1_name"]) or "—",
                         "Supervisor": _names(d.loc[d["role"] == SUP, "person"]) or "—",
                         "Issue": issue, "Correction": (f"{rv['field']} → {rv['value']}" if rv.get("value") else "—"),
                         "Note": rv["note"] or "—",
                         "Decided by": rv["by"] or "—",
                         "_e1_ids": list(dict.fromkeys(d["e1_id"].dropna()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TODAY — phone-friendly board of one day's rides
# ---------------------------------------------------------------------------

def crosscheck_verdicts(xt: pd.DataFrame) -> tuple[dict, dict]:
    """From timing_crosscheck(): ({E1 form_id: (verdict, css)}, {planned code: [flip texts]})."""
    if xt is None or xt.empty:
        return {}, {}
    by_form = {f: (vd, c) for f, vd, c in zip(xt["_form_id"], xt["_verdict"], xt["_verdict_css"])}
    flips = {}
    for p, vd, fl in zip(xt["_planned"], xt["_verdict"], xt["_flip"]):
        if fl:
            flips.setdefault(p, []).append("Independent record: " + vd)
    return by_form, flips


def today_board(ledger: pd.DataFrame, day, extra_checks: dict | None = None) -> dict:
    """
    {team: (table, css)} for the planned rides on `day`, in ride order, with one
    'Action' line each. A ride with no forms is 'Waiting' until a later ride of
    the same team has a form — then its forms are overdue.
    """
    out = {}
    if ledger is None or ledger.empty:
        return out
    d = ledger[ledger["_date"] == day].copy()
    if d.empty:
        return out
    d["_ride_n"] = pd.to_numeric(d["_ride"], errors="coerce")
    for team, g in d.sort_values("_ride_n").groupby("_team", sort=True):
        submitted = g.loc[g["_tags"].map(lambda x: x != ["Not submitted"]), "_ride_n"]
        last_in = submitted.max() if not submitted.empty else -1
        rows, styles = [], []
        for _, r in g.iterrows():
            tags = r["_open_tags"] if r["_tags"] != ["Not submitted"] else ["Not submitted"]
            had_issues = r["_tags"] != ["Done as planned"]
            p = parse_batch_code(r["Planned code"])
            acts = []
            if tags == ["Not submitted"]:
                if r["_ride_n"] < last_in:
                    state, acts = "Overdue", ["Chase E1 + supervisor forms — a later ride is already in"]
                else:
                    state = "Waiting"
            else:
                if "Validity at risk" in tags:
                    acts.append(f"Validity at risk ({r['_open_what'] or 'marketer'}) — decide if the ride is valid")
                if "Wrong code selected" in tags:
                    acts += [n for n in r["Code selection"].split("; ") if "filed under" in n][:1]
                if "Not-allowed change" in tags:
                    acts.append(f"Not allowed — call supervisor: {r['_open_what']}")
                if "Duplicate E1" in tags:
                    acts.append("Two E1 forms — check which one to keep")
                if "To confirm" in tags:
                    acts.append(f"Check: {r['_open_what']}")
                if "Supervisor only" in tags:
                    acts.append("Chase the E1 form")
                if acts and r["_undecided_tags"]:
                    state = "Action"
                elif acts:
                    state = "To correct"
                else:
                    state = "Reviewed" if had_issues else "OK"
            rows.append({
                "Ride":       r["_ride"],
                "Planned":    f"{p['route'] or ''} · {TREAT.get(p['treatment'], p['treatment'] or '')}",
                "E1":         r["E1 form"].replace("✗ none", "✗"),
                "Supervisor": r["Supervisor form"].replace("✗ none", "✗"),
                "State":      state,
                "Action":     " · ".join(acts) or "—",
                "E1 explanation": r.get("E1 explanation", "—") if acts or had_issues else "—",
                "Checks":     " · ".join(list(r.get("_checks") or []) + (extra_checks or {}).get(r["Planned code"], [])) or "—",
                "Review":     r["Review status"],
            })
            styles.append(state)
        t = pd.DataFrame(rows)
        css = _empty_css(t)
        state_css = {"OK": GREEN, "Reviewed": GREEN, "Action": RED, "Overdue": RED, "Waiting": GRAY,
                     "To correct": PURPLE}
        for i, st_ in enumerate(styles):
            css.at[i, "State"] = state_css.get(st_, "")
            if st_ in ("Action", "Overdue"):
                css.at[i, "Action"] = RED
            if st_ == "To correct":
                css.at[i, "Action"] = PURPLE
            if st_ == "Waiting":
                css.loc[i] = GRAY
        css.loc[t["E1"].str.startswith("✗") & (t["State"] != "Waiting"), "E1"] = RED
        css.loc[t["Checks"] != "—", "Checks"] = AMBER
        out[team] = (t, css)
    return out


def unmatched_submissions(subs: pd.DataFrame, batch_codes: pd.DataFrame) -> pd.DataFrame:
    """Forms whose selected planned code is not in the batch_codes frame."""
    planned = set(planned_codes(batch_codes))
    m = ~subs["batch_code"].map(_txt).isin(planned)
    d = subs[m].sort_values("submitted_at")
    return pd.DataFrame({
        "Submitted":             d["submitted_at"].dt.strftime("%d %b %H:%M"),
        "Role":                  d["role"],
        "Filled by":             d["person"].map(_txt),
        "Enumerator 1":          d["e1_name"].map(_txt),
        "Planned code selected": d["batch_code"].map(_txt).replace("", "(none)"),
        "Actual code":           d["actual_batch_code"].map(_txt),
        "Issue": np.where(d["batch_code"].map(_txt) == "",
                          "No planned code selected", "Planned code not in batch_codes frame"),
    }).reset_index(drop=True)


def submission_log(subs: pd.DataFrame, batch_codes: pd.DataFrame) -> pd.DataFrame:
    """Every form, one row each — the audit trail behind Tab 1."""
    planned = set(planned_codes(batch_codes))
    eff = effective_codes(subs, batch_codes)
    d = subs.assign(_likely=eff["likely_code"], _mis=eff["misfiled"]).sort_values("submitted_at")
    return pd.DataFrame({
        "Submitted":      d["submitted_at"].dt.strftime("%d %b %H:%M"),
        "Ride date":      d["ride_date"].map(fmt_day),
        "Role":           d["role"],
        "Filled by":      d["person"].map(_txt),
        "Enumerator 1":   d["e1_name"].map(_txt),
        "Planned code":   d["batch_code"].map(_txt),
        "Actual code":    d["actual_batch_code"].map(_txt),
        "In plan":        d["batch_code"].map(lambda c: "Yes" if _txt(c) in planned else "No"),
        "Team / ride on form": [f"{_code(r.get('a02a_r')) or '?'} / {_code(r.get('a04_r')) or '?'}" for _, r in d.iterrows()],
        "Belongs to ride": [("⚠ " + l) if m else (l or "—") for l, m in zip(d["_likely"], d["_mis"])],
        "E1 forms on code": d["n_e1_forms"],
        "Code check":     [_changes_text(code_changes(r)) for _, r in d.iterrows()],
        "Form ID":        d["form_id"].map(lambda k: _txt(k)[-12:]),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Supervisor ↔ E1 pairing
# ---------------------------------------------------------------------------

def pair_supervisors(subs: pd.DataFrame) -> dict:
    """
    {E1 row index: (supervisor row index, method)}.
    A supervisor form names the E1 observed (a02_1), so candidates are supervisor
    forms for the same E1 on the same ride date. Among those, the closest ride
    start time (within PAIR_WINDOW_MIN) wins; without times, the same ride
    number, then the same planned code. Several E1 forms may share one supervisor.
    """
    e1 = subs[subs["role"] == E1]
    sp = subs[subs["role"] == SUP]
    start = subs["b02_r"].map(_parse_time) if "b02_r" in subs.columns else pd.Series(pd.NaT, index=subs.index)
    out = {}
    for i, r in e1.iterrows():
        cand = sp[sp["e1_id"] == r["e1_id"]]
        if pd.notna(r["ride_date"]):
            cand = cand[cand["ride_date"] == r["ride_date"]]
        scored = []
        for j, c in cand.iterrows():
            gap = _diff_minutes(start[i], start[j])
            if not np.isnan(gap):
                if abs(gap) <= PAIR_WINDOW_MIN:
                    scored.append((0, abs(gap), j, "ride start time"))
            elif _code(r.get("a04_r")) and _code(r.get("a04_r")) == _code(c.get("a04_r")):
                scored.append((1, 0, j, "ride number"))
            elif _txt(r["batch_code"]) and r["batch_code"] == c["batch_code"]:
                scored.append((2, 0, j, "planned code"))
        if scored:
            best = min(scored)
            out[i] = (best[2], best[3])
    return out


def time_gaps(e1_row, sup_row) -> dict:
    """{checkpoint label: |E1 − supervisor| minutes} for checkpoints recorded on both forms."""
    gaps = {}
    for label, fields in CHECKPOINTS:
        for f in fields:
            a, b = _parse_time(e1_row.get(f)), _parse_time(sup_row.get(f))
            if pd.notna(a) and pd.notna(b):
                gaps[label] = abs(_diff_minutes(a, b))
                break
    return gaps


# ---------------------------------------------------------------------------
# TAB 2 — Implementation fidelity
# ---------------------------------------------------------------------------

def assess_issues(row) -> dict:
    """Which implementation values are flagged and whether each was explained."""
    disruptions = set(_txt(row.get("c10")).split())
    external = bool(disruptions & EXTERNAL_DISRUPTIONS)
    items, n_flag, n_expl = [], 0, 0
    for flag, label, reason_col in ISSUE_DEFS:
        if _num(row.get(flag)) == 1:
            n_flag += 1
            if _txt(row.get(reason_col)):
                items.append(f"{label} (reason given)"); n_expl += 1
            elif external:
                items.append(f"{label} (disruption)"); n_expl += 1
            else:
                items.append(f"{label} (no reason)")
    recording = [label for flag, label in RECORDING_DEFS if _num(row.get(flag)) == 1]
    if n_flag == 0:
        status = "No issues"
    elif n_expl == n_flag:
        status = "Outside target, reason given"
    elif n_expl:
        status = "Outside target, some reasons missing"
    else:
        status = "Outside target, no reason"
    return {"status": status, "items": items, "n_flag": n_flag, "n_expl": n_expl,
            "recording": recording}


def _ride_cols(ident_row) -> dict:
    """Ride columns shared by the Tab 2 tables (planned ride after corrections + actual code)."""
    pt = parse_batch_code(ident_row["planned"]).get("treatment", "")
    at = parse_batch_code(ident_row["actual"]).get("treatment", "") if ident_row["actual"] else ""
    treat = TREAT.get(at or pt, at or pt)
    if at and pt and at != pt:
        treat = f"{treat} (planned {TREAT.get(pt, pt)})"
    return {
        "Ride date":     fmt_day(ident_row["date"]),
        "Team":          ident_row["team"] or "—",
        "Ride #":        ident_row["ride"] or "—",
        "Planned code":  ident_row["planned"] or "—",
        "Actual code":   ident_row["actual"] or "—",
        "Code note":     ident_row["note"] or "—",
        "Treatment":     treat or "—",
    }


def fidelity_table(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                   reviews: pd.DataFrame | None = None):
    """One row per E1 form, on its real planned ride (after corrections): values next to reasons."""
    d = subs[subs["role"] == E1]
    ident = ride_identity(d, batch_codes, reviews)
    d = d.loc[_ride_sort_key(ident, d)]
    rows, flags = [], []
    for i, r in d.iterrows():
        a = assess_issues(r)
        sig, ann = _num(r.get("signup_duration")), _num(r.get("announcement_duration"))
        rows.append({
            **_ride_cols(ident.loc[i]),
            "Enumerator 1":        _txt(r["person"]),
            "Issue status":        a["status"],
            "Flagged values":      "; ".join(a["items"]) or "—",
            "Recording problems":  "; ".join(a["recording"]) or "—",
            COL_DELAY:             fmt_val(r.get("announcement_start_delay")),
            "Delay reason":        _txt(r.get("announcement_delay_explanation")) or "—",
            COL_REC:               fmt_val(sig),
            "Ann./signup reason":  _txt(r.get("signup_duration_explanation")) or "—",
            COL_AUTO:              fmt_val(ann),
            COL_DISCREP:           fmt_val(abs(sig - ann)),
            COL_PITCH:             fmt_val(r.get("pause_duration")),
            "Pause/pitch reason":  _txt(r.get("pause_duration_explanation")) or "—",
            COL_GAP:               fmt_val(r.get("pitch_start_gap")),
            "Pitch start reason":  _txt(r.get("pitch_start_gap_explanation")) or "—",
            COL_RIDE:              fmt_val(r.get("ride_duration")),
            "Ride dur. reason":    _txt(r.get("ride_duration_explanation")) or "—",
            "Completion":          COMPLETION_LABELS.get(_code(r.get("ride_completion_outcome")), "—"),
            "Not completed — why": "; ".join(x for x in [_txt(r.get("signup_completed_why")),
                                                           _txt(r.get("pitchpause_completed_why"))] if x) or "—",
            COL_SIGNUPS:           fmt_val(r.get("total_signups"), 0),
            "Signup reason":       _txt(r.get("final_comment_lowsignup")) or "—",
            "Disruptions":         decode_c10(r.get("c10")),
            "Disruption notes":    _txt(r.get("c10a")) or "—",
            "Pax Q1":              _txt(r.get("q00")) or "—",
            "Pax Q2":              _txt(r.get("q01")) or "—",
        })
        flags.append({
            COL_DELAY: r.get("flag_1_start_delay"), COL_REC: r.get("flag_2_ann_duration"),
            COL_DISCREP: r.get("flag_3_signup_discrep"),
            COL_PITCH: 1 if (_num(r.get("flag_4a_pause_dur")) == 1 or _num(r.get("flag_4b_pitch_dur")) == 1) else 0,
            COL_RIDE: r.get("flag_6_short_ride"), COL_SIGNUPS: r.get("flag_7_low_signup"),
            COL_GAP: r.get("flag_8_pitch_start_gap"),
        })
    t = pd.DataFrame(rows)
    # Columns for questions that aren't in any form in view (e.g. the ride-completion
    # module was removed from the Nov form; pitch start gap is new) are hidden
    hide = []
    if not d.get("ride_completion_outcome", pd.Series(dtype=str)).notna().any():
        hide += ["Completion", "Not completed — why"]
    if not d.get("pitch_start_gap", pd.Series(dtype=float)).notna().any():
        hide += [COL_GAP, "Pitch start reason"]
    t = t.drop(columns=[c for c in hide if c in t.columns])
    flags = [{k: v for k, v in fl.items() if k not in hide} for fl in flags]
    css = _empty_css(t)
    if t.empty:
        return t, css
    for k, fl in enumerate(flags):
        for col, v in fl.items():
            if _num(v) == 1:
                css.at[k, col] = RED
            elif t.at[k, col] == MISSING:
                css.at[k, col] = GRAY
    css["Issue status"] = t["Issue status"].map(
        {"Outside target, no reason": RED, "Outside target, some reasons missing": RED,
         "Outside target, reason given": BLUE, "No issues": GREEN}).fillna("")
    css.loc[t["Recording problems"] != "—", "Recording problems"] = RED
    if "Completion" in t.columns:
        css.loc[t["Completion"] == "Not fully completed", "Completion"] = RED
    css.loc[t["Code note"] != "—", "Code note"] = PURPLE
    return t, css


def crosscheck_table(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                     reviews: pd.DataFrame | None = None):
    """One row per E1 form with the paired supervisor's independent record. Returns (table, css)."""
    pairs = pair_supervisors(subs)
    d = subs[subs["role"] == E1]
    ident = ride_identity(d, batch_codes, reviews)
    d = d.loc[_ride_sort_key(ident, d)]
    rows = []
    for i, r in d.iterrows():
        rc = _ride_cols(ident.loc[i])
        row = {k: rc[k] for k in ["Ride date", "Team", "Ride #", "Planned code"]}
        row["Enumerator 1"] = _txt(r["person"])
        if i not in pairs:
            row.update({"Supervisor": "— not observed —"})
            rows.append(row)
            continue
        j, method = pairs[i]
        s = subs.loc[j]
        gaps = time_gaps(r, s)
        row["Supervisor"] = _txt(s["person"])
        row["Paired by"]  = method
        row["Sup. grade (1–10)"] = fmt_val(s.get("sup_obs_grade"), 1)
        for label, _ in CHECKPOINTS:
            row[f"Gap: {label}"] = f"{gaps[label]:.1f}" if label in gaps else "—"
        if gaps:
            worst = max(gaps, key=gaps.get)
            row["Max gap (min)"] = f"{gaps[worst]:.1f}"
            row["Largest at"]    = worst
        row["Sign-ups E1 / Sup"]   = f"{fmt_val(r.get('total_signups'), 0)} / {fmt_val(s.get('total_signups'), 0)}".replace(MISSING, "?")
        row["Sup: times match"]    = YESNO.get(_code(s.get("sup_times_match")), "—")
        row["Mismatched times"]    = _txt(s.get("sup_time_mismatch_labels")) or "—"
        row["Discrepancy source"]  = DISCREPANCY_SOURCE.get(_code(s.get("sup_time_discrepancy_source")), "—")
        row["E1 recording issue"]  = RECORDING_ISSUE.get(_code(s.get("sup_recording_issue")), "—")
        row["Count match"]         = MATCH_LABELS.get(_code(s.get("s17")), "—")
        row["Ticket match"]        = MATCH_LABELS.get(_code(s.get("s20")), "—")
        row["Completion match"]    = {"1": "✓ Match", "0": "✗ Differs"}.get(_code(s.get("sup_completion_matches_e1")), "—")
        rows.append(row)

    cols = (["Ride date", "Team", "Ride #", "Planned code", "Enumerator 1", "Supervisor", "Paired by",
             "Sup. grade (1–10)"]
            + [f"Gap: {l}" for l, _ in CHECKPOINTS]
            + ["Max gap (min)", "Largest at", "Sign-ups E1 / Sup", "Sup: times match", "Mismatched times",
               "Discrepancy source", "E1 recording issue", "Count match", "Ticket match", "Completion match"])
    t = pd.DataFrame(rows).reindex(columns=cols).fillna("—").replace({MISSING: "—"})
    # Hide columns for questions that aren't in the forms in view (Oct: no grade / sign-up end;
    # Nov: no ride-completion module)
    sp_all = subs[subs["role"] == SUP]
    if not sp_all.get("sup_obs_grade", pd.Series(dtype=str)).notna().any():
        t = t.drop(columns=["Sup. grade (1–10)"])
    if not sp_all.get("sup_completion_matches_e1", pd.Series(dtype=str)).notna().any():
        t = t.drop(columns=["Completion match"])
    if not subs.get("b04", pd.Series(dtype=str)).notna().any():
        t = t.drop(columns=["Gap: Sign-up end"])
    cols = list(t.columns)
    css = _empty_css(t)
    if t.empty:
        return t, css
    for c in [c for c in cols if c.startswith("Gap:")] + ["Max gap (min)"]:
        css.loc[t[c].map(_num) > TIME_GAP_MAX, c] = RED
    css.loc[t["Sup: times match"] == "No", "Sup: times match"] = RED
    css.loc[t["E1 recording issue"] == "E1 recording error", "E1 recording issue"] = RED
    for c in [c for c in ["Count match", "Ticket match", "Completion match"] if c in t.columns]:
        css.loc[t[c].str.startswith("✗"), c] = RED
    so = t["Sign-ups E1 / Sup"].str.split(" / ")
    css.loc[so.map(lambda x: len(x) == 2 and "?" not in x and x[0] != x[1]), "Sign-ups E1 / Sup"] = RED
    css.loc[t["Supervisor"] == "— not observed —"] = GRAY
    return t, css


# ---------------------------------------------------------------------------
# TAB 3 — Enumerator performance
# ---------------------------------------------------------------------------
# Three families of signals, kept separate (never blended into one score):
#   Data quality      — from the enumerator's own E1 forms
#   Protocol          — supervisor behavioural observations (s-fields)
#   Agreement         — supervisor's independent counts/times vs E1's
# Each metric is a share of rides: numerator / denominator, pooled over the period.

E1_METRICS = [
    ("Batch code correct",    "code_ok"),
    ("Timing targets met",    "timing_ok"),
    ("Complete data",         "complete"),
    ("Recorded = auto timing","recording_ok"),
]
SUP_METRICS = [
    ("Protocol (sup. obs.)",  "obs"),        # pooled yes / assessed items
    ("Sign-up count agrees",  "count_ok"),
    ("Times agree",           "times_ok"),
    ("No E1 recording error", "no_rec_err"),
]
ALL_METRICS = E1_METRICS + SUP_METRICS


def _bin(v, yes=("1",), no=("0",)):
    c = _code(v)
    return 1.0 if c in yes else 0.0 if c in no else np.nan


def performance_records(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                        reviews: pd.DataFrame | None = None):
    """
    (e1m, supm): one row per form with 0/1 metric values (NaN = not assessed).

    "Batch code correct" (code_ok) = the E1 picked the right planned code AND the
    actual code matches the plan, except for differences you marked Approved in
    the reviews tab (planned changes). Fix needed / Fixed / unreviewed still count.
    """
    e1 = subs[subs["role"] == E1]
    rmap = review_map(reviews)
    eff = effective_codes(subs, batch_codes) if batch_codes is not None else None
    e1_rows = []
    for _, r in e1.iterrows():
        if eff is not None:
            code = eff.at[_, "eff_code"]
            parts = ({c[2] for c in code_changes_vs(r, code) if c[0] != "allowed"}
                     if _txt(code) else {"Batch code"})
            if eff.at[_, "misfiled"]:
                parts.add("Wrong code")
            parts = {p_ for p_ in parts
                     if not ((rv := _review_for(rmap, code, p_)) and rv["decision"] == "Approved")}
            code_ok = 0.0 if parts else 1.0
        else:
            f = _num(r.get("flag_batch_code_different"))
            code_ok = np.nan if np.isnan(f) else 1.0 - f
        a = assess_issues(r)
        timing_flags = [r.get(f) for f, _, _ in ISSUE_DEFS if f not in NOT_E1_TIMING]
        assessed = [f for f in timing_flags if not np.isnan(_num(f))]
        core_ok = all(not np.isnan(_num(r.get(c))) for c in CORE_FIELDS)
        # impossible timestamp sequences (the recorded-vs-auto discrepancy is its own metric)
        impossible = any(_num(r.get(f)) == 1 for f, _ in RECORDING_DEFS if f != "flag_3_signup_discrep")
        e1_rows.append({
            "_idx": _,
            "e1_id": r["e1_id"], "e1_name": r["e1_name"], "week_start": r["week_start"],
            "ride_date": r["ride_date"], "batch_code": r["batch_code"],
            "code_ok":   code_ok,
            "timing_ok": (0.0 if any(_num(f) == 1 for f in assessed) else 1.0) if assessed else np.nan,
            "explained": (1.0 if a["n_expl"] == a["n_flag"] else 0.0) if a["n_flag"] else np.nan,
            "complete":  1.0 if core_ok and not impossible else 0.0,
            "recording_ok": 1.0 - _num(r.get("flag_3_signup_discrep")) if not np.isnan(_num(r.get("flag_3_signup_discrep"))) else np.nan,
            "issue_status": a["status"], "issue_items": a["items"],
        })
    e1m = pd.DataFrame(e1_rows)

    sp = subs[subs["role"] == SUP]
    sup_rows = []
    for j, r in sp.iterrows():
        rec = {"_idx": j, "e1_id": r["e1_id"], "e1_name": r["e1_name"], "week_start": r["week_start"],
               "ride_date": r["ride_date"], "batch_code": r["batch_code"], "supervisor": r["person"]}
        yes_t = n_t = 0
        for cat, fields in S_CATEGORIES.items():
            vals = [_code(r.get(f)) for f in fields]
            y = sum(v == "1" for v in vals); n = sum(v in ("0", "1") for v in vals)
            rec[f"cat_yes:{cat}"], rec[f"cat_n:{cat}"] = y, n
            yes_t += y; n_t += n
        rec["obs_yes"], rec["obs_n"] = yes_t, n_t
        rec["count_ok"]   = _bin(r.get("s17"), yes=("1",), no=("2",))
        rec["times_ok"]   = _bin(r.get("sup_times_match"))
        rec["no_rec_err"] = _bin(r.get("sup_recording_issue"), yes=("0",), no=("1",))
        rec["strengths"]    = _txt(r.get("sup_feedback_additional_strengths"))
        rec["improvements"] = _txt(r.get("sup_feedback_additional_improvements"))
        rec["auto_improvements"] = _txt(r.get("fb_improvements"))
        rec["count_label"] = MATCH_LABELS.get(_code(r.get("s17")), "—")
        sup_rows.append(rec)
    supm = pd.DataFrame(sup_rows)
    return e1m, supm


def _rate(df, key):
    """(numerator, denominator) for a metric over a slice of records."""
    if df is None or df.empty:
        return 0, 0
    if key == "obs":
        return df["obs_yes"].sum(), df["obs_n"].sum()
    s = df[key].dropna()
    return s.sum(), len(s)


def _people(e1m, supm) -> dict:
    """{e1_id: display name} across both record sets."""
    names = {}
    for df in (e1m, supm):
        if df is not None and not df.empty:
            for i, n in zip(df["e1_id"], df["e1_name"]):
                if _txt(i) and _txt(n):
                    names.setdefault(i, n)
    return names


def performance_summary(e1m, supm, weeks=None):
    """One row per enumerator for the chosen weeks (None = all). Returns (table, css, ids)."""
    def sl(df):
        if df is None or df.empty or weeks is None:
            return df
        return df[df["week_start"].isin(weeks)]
    e1s, sups = sl(e1m), sl(supm)
    rows, ids = [], []
    for pid, name in sorted(_people(e1s, sups).items(), key=lambda x: x[1]):
        a = e1s[e1s["e1_id"] == pid] if e1s is not None and not e1s.empty else None
        b = sups[sups["e1_id"] == pid] if sups is not None and not sups.empty else None
        row = {"Enumerator": name,
               "E1 rides": 0 if a is None else len(a),
               "Observed rides": 0 if b is None else len(b)}
        for label, key in E1_METRICS:
            row[label] = fmt_rate(*_rate(a, key))
        for label, key in SUP_METRICS:
            row[label] = fmt_rate(*_rate(b, key))
        rows.append(row); ids.append(pid)
    t = pd.DataFrame(rows)
    css = _empty_css(t)
    for label, _ in ALL_METRICS:
        if label in t.columns:
            pct = t[label].map(lambda s: _num(s.split("%")[0]) / 100 if "%" in s else np.nan)
            css.loc[pct < PERF_TARGET, label] = RED
    return t, css, ids


def _mark(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else ("✓" if v == 1 else "✗")


def performance_by_ride(subs, e1m, supm, weeks=None, pids=None):
    """
    One row per ride × enumerator: the E1 form and, when observed, the paired
    supervisor form side by side, with each performance measure as ✓ / ✗ / —.
    Supervisor forms with no matching E1 form get their own row.
    Returns (table, css).
    """
    pairs = pair_supervisors(subs)
    sup_by_idx = {r["_idx"]: r for r in supm.to_dict("records")} if not supm.empty else {}
    used, rows = set(), []

    def build(e, s):
        src = e if e is not None else s
        row = {
            "Ride date":    fmt_day(src["ride_date"]),
            "Planned code": _txt(src["batch_code"]),
            "Enumerator":   _txt(src["e1_name"]),
            "Supervisor":   _txt(s["supervisor"]) if s is not None else "—",
            "Forms":        ("E1 + supervisor" if s is not None else "E1 only") if e is not None else "Supervisor only",
        }
        for label, key in E1_METRICS:
            row[label] = _mark(e[key]) if e is not None else "—"
        if s is not None:
            row["Protocol (sup. obs.)"] = fmt_rate(s["obs_yes"], s["obs_n"])
            for label, key in SUP_METRICS[1:]:
                row[label] = _mark(s[key])
        else:
            for label, _ in SUP_METRICS:
                row[label] = "—"
        row["Flagged values"]     = "; ".join(e["issue_items"]) if e is not None and e["issue_items"] else "—"
        row["Supervisor: strengths"]  = (s["strengths"] or "—") if s is not None else "—"
        row["Supervisor: to improve"] = (s["improvements"] or "—") if s is not None else "—"
        row["_e1_id"], row["_week"], row["_date"] = src["e1_id"], src["week_start"], src["ride_date"]
        row["_protocol"] = (s["obs_yes"] / s["obs_n"]) if s is not None and s["obs_n"] else np.nan
        return row

    for e in (e1m.to_dict("records") if not e1m.empty else []):
        j = pairs.get(e["_idx"], (None,))[0]
        s = sup_by_idx.get(j)
        if s is not None:
            used.add(j)
        rows.append(build(e, s))
    for j, s in sup_by_idx.items():
        if j not in used:
            rows.append(build(None, s))

    t = pd.DataFrame(rows)
    if t.empty:
        return t, _empty_css(t)
    if weeks is not None:
        t = t[t["_week"].isin(weeks)]
    if pids is not None:
        t = t[t["_e1_id"].isin(pids)]
    t = t.sort_values(["_date", "Enumerator", "Planned code"]).reset_index(drop=True)

    css = _empty_css(t)
    metric_cols = [l for l, _ in ALL_METRICS if l != "Protocol (sup. obs.)"]
    for c in metric_cols:
        css.loc[t[c] == "✗", c] = RED
        css.loc[t[c] == "✓", c] = GREEN
    css.loc[t["_protocol"] < PERF_TARGET, "Protocol (sup. obs.)"] = RED
    css.loc[t["Forms"] == "Supervisor only", "Forms"] = BLUE
    css.loc[t["Flagged values"].str.contains("no reason", na=False), "Flagged values"] = RED
    return t, css


def performance_trend(e1m, supm, pid) -> pd.DataFrame:
    """Rows = weeks, columns = metric shares (0–1) for one enumerator — for charts."""
    weeks = sorted(set(
        list(e1m.loc[e1m["e1_id"] == pid, "week_start"].dropna() if not e1m.empty else [])
        + list(supm.loc[supm["e1_id"] == pid, "week_start"].dropna() if not supm.empty else [])))
    out = []
    for w in weeks:
        a = e1m[(e1m["e1_id"] == pid) & (e1m["week_start"] == w)] if not e1m.empty else None
        b = supm[(supm["e1_id"] == pid) & (supm["week_start"] == w)] if not supm.empty else None
        row = {"Week of": w}
        for label, key in E1_METRICS:
            n, dn = _rate(a, key); row[label] = n / dn if dn else np.nan
        for label, key in SUP_METRICS:
            n, dn = _rate(b, key); row[label] = n / dn if dn else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("Week of") if out else pd.DataFrame()


def category_scores(supm, pid, weeks=None) -> pd.DataFrame:
    if supm is None or supm.empty:
        return pd.DataFrame()
    b = supm[supm["e1_id"] == pid]
    if weeks is not None:
        b = b[b["week_start"].isin(weeks)]
    rows = []
    for cat in S_CATEGORIES:
        y, n = b[f"cat_yes:{cat}"].sum(), b[f"cat_n:{cat}"].sum()
        rows.append({"Category": cat, "Score": fmt_rate(y, n), "_share": y / n if n else np.nan})
    return pd.DataFrame(rows)


def ride_history(subs, pid) -> pd.DataFrame:
    """Every ride involving this enumerator: their E1 forms and supervisor forms about them."""
    d = subs[subs["e1_id"] == pid].sort_values(["ride_date", "submitted_at"])
    rows = []
    for _, r in d.iterrows():
        if r["role"] == E1:
            a = assess_issues(r)
            rows.append({"Ride date": fmt_day(r["ride_date"]), "Planned code": _txt(r["batch_code"]),
                         "Form": "E1 form", "By": _txt(r["person"]),
                         "Issue status": a["status"], "Details": "; ".join(a["items"] + a["recording"]) or "—"})
        else:
            yes = n = 0
            for fields in S_CATEGORIES.values():
                for f in fields:
                    v = _code(r.get(f)); yes += v == "1"; n += v in ("0", "1")
            fb = " · ".join(x for x in [_txt(r.get("sup_feedback_additional_strengths")) and f"+ {_txt(r.get('sup_feedback_additional_strengths'))}",
                                         _txt(r.get("sup_feedback_additional_improvements")) and f"Δ {_txt(r.get('sup_feedback_additional_improvements'))}"] if x)
            rows.append({"Ride date": fmt_day(r["ride_date"]), "Planned code": _txt(r["batch_code"]),
                         "Form": "Supervisor observation", "By": _txt(r["person"]),
                         "Issue status": f"Protocol {fmt_rate(yes, n)}", "Details": fb or "—"})
    return pd.DataFrame(rows)


def feedback_card(e1m, supm, pid, weeks=None, fixes: pd.DataFrame | None = None) -> str:
    """Markdown feedback summary for one enumerator over the chosen weeks."""
    names = _people(e1m, supm)
    name = names.get(pid, str(pid))
    a = e1m[e1m["e1_id"] == pid] if not e1m.empty else e1m
    b = supm[supm["e1_id"] == pid] if not supm.empty else supm
    if weeks is not None:
        a = a[a["week_start"].isin(weeks)] if not a.empty else a
        b = b[b["week_start"].isin(weeks)] if not b.empty else b
    dates = pd.concat([a["ride_date"] if not a.empty else pd.Series(dtype="datetime64[ns]"),
                       b["ride_date"] if not b.empty else pd.Series(dtype="datetime64[ns]")]).dropna()
    period = f"{fmt_day(dates.min())} – {fmt_day(dates.max())}" if not dates.empty else "—"

    L = [f"# Feedback — {name}", "",
         f"**Period:** {period}  ",
         f"**Rides as Enumerator 1:** {len(a)}  ·  **Rides observed by a supervisor:** {len(b)}", "",
         "## Your forms (data quality)"]
    for label, key in E1_METRICS:
        L.append(f"- {label}: {fmt_rate(*_rate(a, key))}")
    L += ["", "## Supervisor observations"]
    for label, key in SUP_METRICS:
        L.append(f"- {label}: {fmt_rate(*_rate(b, key))}")

    cats = category_scores(supm, pid, weeks)
    if not cats.empty and cats["_share"].notna().any():
        c = cats.dropna(subset=["_share"]).sort_values("_share", ascending=False)
        strong = c[c["_share"] >= PERF_TARGET]
        weak = c[c["_share"] < PERF_TARGET].sort_values("_share")
        L += ["", "## Strongest areas"]
        L += [f"- {r.Category}: {r.Score}" for r in strong.itertuples()] or ["- —"]
        L += ["", "## Areas to work on"]
        L += [f"- {r.Category}: {r.Score}" for r in weak.itertuples()] or ["- None below target"]

    if not a.empty:
        unexplained = [(r.ride_date, r.batch_code, [i for i in r.issue_items if "(no reason)" in i])
                       for r in a.itertuples()]
        unexplained = [u for u in unexplained if u[2]]
        L += ["", "## Flagged values with no explanation recorded"]
        L += [f"- {fmt_day(d)} · {c}: {', '.join(i.replace(' (no reason)', '') for i in items)}"
              for d, c, items in unexplained] or ["- None"]

    if fixes is not None and not fixes.empty:
        mine = fixes[fixes["_e1_ids"].map(lambda ids: pid in ids)]
        if not mine.empty:
            L += ["", "## Batch code / form corrections"]
            L += [f"- {r['Ride date']} · {r['Planned code']} — {r['Issue']}"
                  + (f": {r['Note']}" if r['Note'] != "—" else "")
                  for _, r in mine.iterrows()]

    if not b.empty:
        comments = [(r.ride_date, r.supervisor, r.strengths, r.improvements) for r in b.itertuples()
                    if r.strengths or r.improvements]
        if comments:
            L += ["", "## Supervisor comments"]
            for d, s, st_, im in comments:
                L.append(f"- {fmt_day(d)} ({s}):" + (f" Strengths: {st_.rstrip('. ')}." if st_ else "")
                         + (f" To improve: {im.rstrip('. ')}." if im else ""))
    return "\n".join(L) + "\n"


def supervisor_observations(subs: pd.DataFrame):
    """One row per supervisor form with category scores. Returns (table, css)."""
    sp = subs[subs["role"] == SUP].sort_values(["ride_date", "submitted_at"])
    rows = []
    for _, r in sp.iterrows():
        row = {"Ride date": fmt_day(r["ride_date"]), "Planned code": _txt(r["batch_code"]),
               "Enumerator 1": _txt(r["e1_name"]), "Supervisor": _txt(r["person"])}
        yt = nt = 0
        for cat, fields in S_CATEGORIES.items():
            vals = [_code(r.get(f)) for f in fields]
            y = sum(v == "1" for v in vals); n = sum(v in ("0", "1") for v in vals)
            row[cat] = f"{y}/{n}" if n else "—"
            yt += y; nt += n
        row["Total"]        = f"{yt}/{nt}" if nt else "—"
        row["Count match"]  = MATCH_LABELS.get(_code(r.get("s17")), "—")
        row["Ticket match"] = MATCH_LABELS.get(_code(r.get("s20")), "—")
        row["Strengths"]    = _txt(r.get("sup_feedback_additional_strengths")) or "—"
        row["Improvements"] = _txt(r.get("sup_feedback_additional_improvements")) or "—"
        rows.append(row)
    t = pd.DataFrame(rows)
    css = _empty_css(t)
    if t.empty:
        return t, css

    def imperfect(s):
        try:
            y, n = s.split("/"); return int(y) < int(n)
        except Exception:
            return False
    for cat in list(S_CATEGORIES) + ["Total"]:
        css.loc[t[cat].map(imperfect), cat] = RED
    for c in ["Count match", "Ticket match"]:
        css.loc[t[c].str.startswith("✗"), c] = RED
    return t, css


# ---------------------------------------------------------------------------
# ENUMERATORS — day grid, appendix-style tables and daily feedback messages
# ---------------------------------------------------------------------------
# Rules (same as the training report):
#   * your review decisions apply: OK'd differences don't count; "correct in
#     cleaning" wrong codes do count;
#   * start delay, pause/pitch length and ride length depend on the routes, so
#     they are shown but NOT counted, unless the supervisor marked that E1 did
#     not respect the 25-minute period (s15 = No);
#   * "pitch/pause end after ride end" is treated as a form issue, not E1 error.

CHECK_LABELS = {
    "s01": "Sat in assigned seat", "s02": "Started the announcement at the right moment", "s03": "Followed SurveyCTO correctly and in real time",
    "s04": "Followed the script instructions", "s05": "Loud, clear voice",
    "s06": "Natural delivery and eye contact", "s07": "Persuasive, proactive attitude",
    "s08": "Active stance at the front", "s09": "Displayed the sign-up card",
    "s10": "Paused and resumed at the right points", "s11": "Addressed passenger questions appropriately",
    "s12": "Avoided guessing answers", "s15": "Stayed quiet after the announcement (pitch/pause)",
    "s16": "Checked cards for missing information", "s19": "Gave the apprentice the ticket bag",
    "s21": "Agreed the fare count with the apprentice", "s22": "Apprentice signed the payment receipt",
    "s23": "Completed recruitment questions before ride end", "s24": "RIDE form completed before ride end",
    "s25": "Labelled and stored the ride bag", "s26": "Stored all materials",
}
# How each missed checklist item is phrased as advice in a message
CHECK_ADVICE = {
    "s01": "sit in your assigned seat (B1)",
    "s02": "start the announcement as soon as the ride starts if the bus is at least half full, and never after the strategic point",
    "s03": "follow SurveyCTO step by step while you announce",
    "s04": "read the full script without skipping lines", "s05": "speak louder so everyone can hear",
    "s06": "speak naturally and keep eye contact rather than reading", "s07": "be more persuasive and proactive",
    "s08": "stand at the front and engage the passengers", "s09": "show the sign-up card to passengers",
    "s10": "pause at the right point and resume once most cards are handed out or collected",
    "s11": "answer passenger questions using the approved guidance",
    "s12": "if you don't know an answer, say you'll check with the supervisor instead of guessing",
    "s15": "stay seated and don't restart recruitment after the announcement ends",
    "s16": "check every card for missing information", "s19": "give the apprentice the ticket bag",
    "s21": "agree the number of fares with the apprentice", "s22": "make sure the apprentice signs the receipt",
    "s23": "finish the recruitment questions in SurveyCTO before the ride ends",
    "s24": "complete the RIDE form before the ride ends", "s25": "label the ride bag with the batch code and store it",
    "s26": "store all the materials properly",
}
DURATION_PARTS_NOT_COUNTED = True


def ride_summary(subs: pd.DataFrame, batch_codes: pd.DataFrame, reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per E1 ride with everything the Enumerators tab and messages need."""
    rmap = review_map(reviews)
    eff = effective_codes(subs, batch_codes)
    s2 = subs.assign(_eff=eff["eff_code"], _mis=eff["misfiled"])
    e1 = s2[s2["role"] == E1]
    sp = s2[s2["role"] == SUP]
    sup_by = {}
    for j, r in sp.iterrows():
        sup_by.setdefault(r["_eff"], r)
    PART = {"treatment": "Treatment", "route": "Route", "date": "Date", "team": "Team", "ride_num": "Ride #"}
    rows = []
    for i, r in e1.iterrows():
        code = r["_eff"]
        s = sup_by.get(code)
        dec = lambda part: (_review_for(rmap, code, part) or {}).get("decision")
        sig, ann = _num(r.get("signup_duration")), _num(r.get("announcement_duration"))
        a = assess_issues(r)
        # E1 vs supervisor actual code (approved parts and supervisor-side differences don't count)
        inconsistent = False
        if s is not None and _txt(r.get("actual_batch_code")) != _txt(code):
            pa, pb = parse_batch_code(r.get("actual_batch_code")), parse_batch_code(s.get("actual_batch_code"))
            if pa["parse_ok"] and pb["parse_ok"]:
                diff = {PART[k] for k in PART if pa[k] != pb[k]}
                inconsistent = any(dec(p_) != "Approved" for p_ in diff)
        checks, missed = {}, []
        if s is not None:
            for f in CHECK_LABELS:
                c = _code(s.get(f))
                if c in ("0", "1"):
                    checks[f] = c == "1"
                    if c == "0":
                        missed.append((f, _txt(s.get(f + "a"))))
        pause_counted = s is not None and _code(s.get("s15")) == "0"
        missing_reasons = [i_.replace(" (no reason)", "") for i_ in a["items"] if "(no reason)" in i_
                           and not (DURATION_PARTS_NOT_COUNTED and any(k in i_ for k in ("Start delay", "Pause", "Pitch", "Short ride")))]
        impossible = [x for x in a["recording"] if x not in ("Recorded ≠ auto timing", "Pitch/pause ends after ride end")]
        rows.append({
            "form_id": _txt(r.get("form_id")),
            "counts_verified": (None if s is None or _code(s.get("s17")) not in ("1", "2") else
                                _code(s.get("s17")) == "1" and _code(s.get("s20")) in ("1", None)),
            "person": r["person"], "e1_id": r["e1_id"], "date": r["ride_date"], "team": _code(r.get("a02a_r")) or "",
            "ride": _code(r.get("a04_r")) or "", "code": code, "selected": _txt(r["batch_code"]),
            "wrong_code": bool(r["_mis"]) and dec("Wrong code") != "Approved",
            "inconsistent": inconsistent,
            "same_day": pd.notna(r["submitted_at"]) and pd.notna(r["ride_date"]) and r["submitted_at"].normalize() == r["ride_date"],
            "delay": _num(r.get("announcement_start_delay")), "sig": sig, "ann": ann,
            "pause": _num(r.get("pause_duration")), "ride_dur": _num(r.get("ride_duration")),
            "signups": _num(r.get("total_signups")),
            "sig_ok": (not np.isnan(sig)) and 8 <= round(sig, 1) <= 12,
            "rec_ok": not (np.isnan(sig) or np.isnan(ann)) and abs(sig - ann) <= 2,
            "pause_flag": pause_counted,
            "sig_reason": _txt(r.get("signup_duration_explanation")),
            "missing_reasons": missing_reasons, "impossible": impossible,
            "supervisor": _txt(s["person"]) if s is not None else "",
            "supervised": s is not None and bool(checks),
            "checks": checks, "missed": missed,
            "obs_yes": sum(checks.values()), "obs_n": len(checks),
            "times": {"1": "Yes", "0": "No"}.get(_code(s.get("sup_times_match")) if s is not None else None, "—"),
            "rec_err": s is not None and _code(s.get("sup_recording_issue")) == "1",
            "grade": _num(s.get("sup_obs_grade")) if s is not None else np.nan,   # Nov form, 1–10
            "mismatch_detail": _txt(s.get("sup_time_mismatch_details")) if s is not None else "",
            "strengths": _txt(s.get("sup_feedback_additional_strengths")) if s is not None else "",
            "improve": _txt(s.get("sup_feedback_additional_improvements")) if s is not None else "",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["recording_problem"] = out["wrong_code"] | ~out["same_day"] | out["rec_err"] | out["inconsistent"] \
        | out["impossible"].map(bool) | ~out["rec_ok"]
    return out.sort_values(["person", "date", "ride"]).reset_index(drop=True)


def _day_issue_text(d: pd.DataFrame) -> str:
    bits = []
    n = int(d["wrong_code"].sum());        bits += [f"{n} wrong code" + ("s" if n > 1 else "")] if n else []
    n = int((~d["same_day"]).sum());       bits += [f"{n} late form" + ("s" if n > 1 else "")] if n else []
    n = int((d["rec_err"] | ~d["rec_ok"] | d["impossible"].map(bool)).sum()); bits += ["timing record"] if n else []
    n = int(d["inconsistent"].sum());      bits += ["code mismatch"] if n else []
    return ", ".join(bits) or "clean"


def day_grid(rs: pd.DataFrame):
    """Enumerator × day table: checklist score and recording problems, coloured. Returns (table, css)."""
    if rs is None or rs.empty:
        return pd.DataFrame(), pd.DataFrame()
    days = sorted(rs["date"].dropna().unique())
    people = list(dict.fromkeys(rs.sort_values("person")["person"]))
    rows, styles = [], []
    for p in people:
        row, sty = {"Enumerator": p}, {"Enumerator": ""}
        for d_ in days:
            col = fmt_day(pd.Timestamp(d_))[:6]
            d = rs[(rs["person"] == p) & (rs["date"] == d_)]
            if d.empty:
                row[col], sty[col] = "—", GRAY
                continue
            n_obs = d["obs_n"].sum()
            score = f"{100 * d['obs_yes'].sum() / n_obs:.0f}%" if n_obs else "no checklist"
            if d["grade"].notna().any():
                score += f" · grade {d['grade'].mean():.1f}"
            issues = _day_issue_text(d)
            row[col] = f"{score} · {issues}  ({len(d)} ride{'s' if len(d) > 1 else ''})"
            share_bad = d["recording_problem"].mean()
            pct_ = d["obs_yes"].sum() / n_obs if n_obs else 1
            sty[col] = (f"background-color: {RED_BG}" if share_bad >= 0.5 else
                        f"background-color: {GREEN_BG}" if pct_ >= 0.93 and share_bad == 0 else
                        f"background-color: {AMBER_BG}")
        rows.append(row); styles.append(sty)
    return pd.DataFrame(rows), pd.DataFrame(styles)


def ride_numbers_table(rs: pd.DataFrame):
    """Appendix A style: one row per ride. Returns (table, css)."""
    if rs is None or rs.empty:
        return pd.DataFrame(), pd.DataFrame()
    t = pd.DataFrame({
        "Enumerator 1": rs["person"], "Date": rs["date"].map(fmt_day), "Team · ride": rs["team"] + " · " + rs["ride"],
        "Supervisor": rs["supervisor"].replace("", "—"),
        "Correct code": np.where(rs["wrong_code"], "No", "Yes"),
        "Sent same day": np.where(rs["same_day"], "Yes", "No"),
        "Start delay": rs["delay"].map(lambda x: fmt_val(x, 1)),
        "Ann./ sign-up": rs["sig"].map(lambda x: fmt_val(x, 1)),
        "Recorded ≈ auto": np.where(rs["rec_ok"], "Yes", "No"),
        "Pause/ pitch": rs["pause"].map(lambda x: fmt_val(x, 1)),
        "Ride length": rs["ride_dur"].map(lambda x: fmt_val(x, 1)),
        "Sign-ups": rs["signups"].map(lambda x: fmt_val(x, 0)),
        "Times agree": rs["times"],
        "Protocol": [f"{a}/{b}" if b else "—" for a, b in zip(rs["obs_yes"], rs["obs_n"])],
        "Sup. grade": rs["grade"].map(lambda x: fmt_val(x, 1)),
    })
    if not rs["grade"].notna().any():
        t = t.drop(columns=["Sup. grade"])
    css = _empty_css(t)
    css.loc[rs["wrong_code"].values, "Correct code"] = RED
    css.loc[~rs["same_day"].values, "Sent same day"] = RED
    css.loc[~rs["sig_ok"].values, "Ann./ sign-up"] = RED
    css.loc[~rs["rec_ok"].values, "Recorded ≈ auto"] = RED
    css.loc[rs["pause_flag"].values, "Pause/ pitch"] = RED
    css.loc[(t["Times agree"] == "No").values, "Times agree"] = RED
    for c in ["Start delay", "Ride length"] + ([] if rs["pause_flag"].any() else ["Pause/ pitch"]):
        css.loc[css[c] == "", c] = GRAY
    return t, css


def qualitative_table(rs: pd.DataFrame):
    """Appendix B style: supervisor observations and feedback per ride. Returns (table, css)."""
    if rs is None or rs.empty:
        return pd.DataFrame(), pd.DataFrame()
    def missed_text(r):
        if not r["supervised"]:
            return "No supervisor checklist for this ride"
        if not r["missed"]:
            return "None: all observed behaviours done"
        return "; ".join(CHECK_LABELS[f] + (f" ({why})" if why else "") for f, why in r["missed"])
    t = pd.DataFrame({
        "Enumerator 1": rs["person"], "Date · ride": rs["date"].map(fmt_day) + " · " + rs["ride"],
        "Supervisor": rs["supervisor"].replace("", "—"),
        "Protocol": [f"{a}/{b}" if b else "—" for a, b in zip(rs["obs_yes"], rs["obs_n"])],
        "Sup. grade": rs["grade"].map(lambda x: fmt_val(x, 1)),
        "Behaviours not observed (supervisor's reason)": [missed_text(r) for _, r in rs.iterrows()],
        "Strengths": rs["strengths"].replace("", "—"), "To improve": rs["improve"].replace("", "—"),
    })
    if not rs["grade"].notna().any():
        t = t.drop(columns=["Sup. grade"])
    css = _empty_css(t)
    css.loc[t["Behaviours not observed (supervisor's reason)"].str.startswith("None").values,
            "Behaviours not observed (supervisor's reason)"] = f"background-color: {GREEN_BG}"
    return t, css


def daily_message(rs: pd.DataFrame, person: str, day, sender: str = "") -> str:
    """Draft a WhatsApp-friendly feedback message for one enumerator and day."""
    d = rs[(rs["person"] == person) & (rs["date"] == day)].sort_values("ride")
    if d.empty:
        return ""
    first = person.split()[0]
    n = len(d)
    supervised = d["supervised"].any()
    rides_txt = f"{n} ride{'s' if n > 1 else ''}"
    head = f"Hi {first}, here is your feedback for {fmt_day(pd.Timestamp(day))} ({rides_txt}"
    head += ")." if supervised else ", based on your forms, no supervisor today)."
    if d["signups"].notna().any():
        head += f" Sign-ups: {', '.join(fmt_val(x, 0) for x in d['signups'])}."
    good, work = [], []
    lbl = (lambda r: f"ride {r['ride']}") if n > 1 else (lambda r: "this ride")

    # --- from the E1 form (always available)
    if not d["wrong_code"].any():
        good.append("you picked the right batch code" + ((" on both rides" if n == 2 else f" on all {n} rides") if n > 1 else ""))
    for _, r in d[d["wrong_code"]].iterrows():
        work.append(f"on {lbl(r)} you selected {r['selected']} instead of {r['code']}. Check the date, team and "
                    f"ride number before choosing the code")
    if d["same_day"].all():
        good.append("you sent your forms the same day")
    else:
        late = d[~d["same_day"]]
        work.append("send every form before the end of the day (" + ", ".join(lbl(r) for _, r in late.iterrows())
                    + " arrived late)")
    if d["rec_ok"].all() and not d["impossible"].map(bool).any() and not d["rec_err"].any():
        good.append("your recorded times were consistent")
    else:
        for _, r in d[~d["rec_ok"]].iterrows():
            work.append(f"on {lbl(r)} your recorded announcement time ({r['sig']:.1f} min) doesn't match the form's "
                        f"own timing ({r['ann']:.1f} min). Tap each checkpoint at the moment it happens")
        for _, r in d[d["impossible"].map(bool)].iterrows():
            work.append(f"on {lbl(r)} some times are in an impossible order ({'; '.join(r['impossible']).lower()}). "
                        f"Record each step as it happens")
    # Announcement/sign-up outside 8–12 min: report the value; a reason is context, not an excuse
    # (skipped when that ride's times were not recorded reliably — the value can't be trusted)
    unreliable = ~d["rec_ok"] | d["rec_err"] | d["impossible"].map(bool)
    for _, r in d[~d["sig_ok"] & d["sig"].notna() & ~unreliable].iterrows():
        why = f" (you noted: {r['sig_reason']})" if r["sig_reason"] else " (no reason given)"
        work.append(f"on {lbl(r)} the announcement and sign-up took {r['sig']:.1f} min; aim for 8–12 min{why}")
    if d["sig_ok"].all():
        good.append("your announcement and sign-up stayed within 8–12 minutes")
    for _, r in d.iterrows():
        other = [x for x in r["missing_reasons"] if "Ann./signup" not in x]
        if other:
            work.append(f"on {lbl(r)} add a reason when a value is outside the target: {', '.join(other).lower()}")

    # --- from the supervisor (when present)
    if supervised:
        s = d[d["supervised"]]
        y, m = int(s["obs_yes"].sum()), int(s["obs_n"].sum())
        if y == m:
            good.append(f"the supervisor saw you do every checklist behaviour ({y}/{m})")
        else:
            good.append(f"the supervisor saw you do {y} of {m} checklist behaviours")
        if (s["times"] == "Yes").all() and (s["times"] != "—").any():
            good.append("your times matched the supervisor's")
        for _, r in s[s["rec_err"]].iterrows():
            det = f" ({r['mismatch_detail']})" if r["mismatch_detail"] else ""
            work.append(f"on {lbl(r)} the supervisor found a timing recording error{det}")
        advice = {}
        for _, r in s.iterrows():
            for f, why in r["missed"]:
                route_reason = bool(why) and any(k in why.lower() for k in ("route", "distance", "apprentice", "short", "traffic", "enumerator 2", "enumerators 2", "e2", "e3", "ticket", "sign up", "sign-up", "not given", "delay"))
                if f in ("s23", "s24") and route_reason:
                    continue          # the supervisor put it down to the route / rest of the team
                advice.setdefault(f, []).append(r)
        for f, rr in advice.items():
            work.append(CHECK_ADVICE[f])
        quotes_good = [x for x in s["strengths"] if x and x.lower() not in ("good", "no other", "nothing more")]
        quotes_work = [x for x in s["improve"] if x and not x.lower().startswith(("just need to continue", "should just continue", "continue"))]
        if quotes_good:
            good.append("supervisor: \"" + "\"; \"".join(q.rstrip(". ") for q in dict.fromkeys(quotes_good)) + "\"")
        if quotes_work:
            work.append("supervisor: \"" + "\"; \"".join(q.rstrip(". ") for q in dict.fromkeys(quotes_work)) + "\"")

    cap = lambda x: x[:1].upper() + x[1:]
    lines = [head, "", "Well done:"] + [f"- {cap(g)}" for g in good]
    lines += ["", "To work on:"] + ([f"- {cap(w)}" for w in work] if work else ["- Nothing to fix today. Keep it up!"])
    lines += ["", "Thank you!" + (f" {sender}" if sender else "")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TAB 2 — Ride timeline and start checks (Nov form)
# ---------------------------------------------------------------------------
BUS_FULLNESS = {"1": "< half full", "2": "≥ half full", "3": "Full (seated)", "4": "Full + standing"}
YESNO_SHORT = {"1": "Yes", "0": "No"}
MEDIA_FIELDS = [   # (column in the data, label)
    ("signup_check",      "Sign-up photos"),
    ("signup_photo_e2",   "E2 cards photo"),
    ("signup_photo_e3",   "E3 cards photo"),
    ("aa_1_ann_start",    "Audio 1: ann. start"),
    ("aa_2_ann_middle",   "Audio 2: ann. middle"),
    ("aa_3_pause_start",  "Audio 3: pitch/pause start"),
    ("aa_4_pause_middle", "Audio 4: pitch/pause middle"),
    ("aa_5_pause_end",    "Audio 5: after pitch/pause end"),
]


def _clock(t) -> str:
    return t.strftime("%H:%M") if pd.notna(t) else "—"


def _ann_signup(r) -> float:
    """Announcement + sign-up, b03 → b04 (minutes): the form's signup_duration, else from the times."""
    x = _num(r.get("signup_duration"))
    if np.isnan(x):
        x = _diff_minutes(_parse_time(r.get("b03")), _parse_time(r.get("b04")))
    return x


def _start_check(r) -> tuple[str, str]:
    """(text, style) for 'did the announcement start when it should?'."""
    delay = _num(r.get("announcement_start_delay"))
    full = _code(r.get("b02a"))
    if np.isnan(delay):
        return "—", GRAY
    delay = round(delay, 1)                   # judged on the value shown (1 decimal)
    if delay <= DELAY_MAX_MIN:
        return f"On time ({delay:.1f} min)", GREEN
    if full in ("2", "3", "4"):
        why = _txt(r.get("gap_fullbus_why"))
        return f"Waited {delay:.1f} min although bus {BUS_FULLNESS[full]}" + (f": {why}" if why else ""), RED
    reason = _code(r.get("gap_notfull_reason"))
    if full == "1" and reason == "1":
        return f"Waited {delay:.1f} min: bus < half full (expected)", GREEN
    if reason == "2":
        oth = _txt(r.get("gap_notfull_other"))
        return f"Waited {delay:.1f} min: other reason" + (f" ({oth})" if oth else ""), AMBER
    return f"Waited {delay:.1f} min, no reason recorded", RED


CLOCK_PREFIX = "🕒 "
COL_ANN   = f"Announcement + sign-up ({ANN_DUR_MIN}–{ANN_DUR_MAX} min)"
COL_LAG   = "Ann. end → sign-up end (min)"
COL_SGAP  = f"Sign-up end → pitch/pause (≤{DELAY_MAX_MIN} min)"
COL_PP    = "Pitch/pause (min)"
COL_RIDE2 = f"Ride ({RIDE_MIN_MIN}+ min)"
COL_PREV  = "Since team's previous ride (min)"


def _times(r) -> dict:
    """Parsed clock times of one form (any source) with pitch/pause start/end resolved."""
    t = {k: _parse_time(r.get(k)) for k in ["b02_r", "b03", "b04", "b05", "b06", "b09", "b10", "b11"]}
    t["pp_start"] = t["b05"] if pd.notna(t["b05"]) else t["b06"]
    t["pp_end"] = t["b09"] if pd.notna(t["b09"]) else t["b10"]
    return t


def _durations(t: dict) -> dict:
    """Durations computed the same way for every source (E1, E4, supervisor)."""
    return {"delay": _diff_minutes(t["b02_r"], t["b03"]),
            "signup": _diff_minutes(t["b03"], t["b04"]),
            "gap": _diff_minutes(t["b04"], t["pp_start"]),
            "pp": _diff_minutes(t["pp_start"], t["pp_end"]),
            "ride": _diff_minutes(t["b02_r"], t["b11"])}


def _order_problems(t: dict) -> list:
    """Steps recorded in an impossible order (more than 1 min backwards)."""
    seq = [("Ride start", t["b02_r"]), ("Ann. start", t["b03"]), ("Sign-up end", t["b04"]),
           ("Pitch/pause start", t["pp_start"]), ("Pitch/pause end", t["pp_end"]), ("Ride end", t["b11"])]
    seq = [(n, x) for n, x in seq if pd.notna(x)]
    out = []
    for (n1, x1), (n2, x2) in zip(seq, seq[1:]):
        dm = _diff_minutes(x1, x2)
        if not np.isnan(dm) and dm < -1:
            out.append(f"{n2} before {n1.lower()}")
    return out


def ride_timeline_table(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                        reviews: pd.DataFrame | None = None, verdicts: dict | None = None):
    """
    One row per E1 form. Issues first (start checks, durations and gaps against their
    targets, impossible order), then — in columns starting with 🕒 — the clock times.
    Returns (table, css).
    """
    d = subs[subs["role"] == E1]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    ident = ride_identity(d, batch_codes, reviews)
    d = d.loc[_ride_sort_key(ident, d)]
    T = {i: _times(r) for i, r in d.iterrows()}
    # time since the team's previous ride (previous ride end → this ride start), same day
    prev = {}
    keyed = sorted(d.index, key=lambda i: (ident.at[i, "date"] if pd.notna(ident.at[i, "date"]) else pd.Timestamp.max,
                                           str(ident.at[i, "team"]), T[i]["b02_r"] if pd.notna(T[i]["b02_r"]) else pd.Timestamp.max))
    for a_, b_ in zip(keyed, keyed[1:]):
        if ident.at[a_, "date"] == ident.at[b_, "date"] and str(ident.at[a_, "team"]) == str(ident.at[b_, "team"]):
            prev[b_] = _diff_minutes(T[a_]["b11"], T[b_]["b02_r"])
    rows, sty = [], []
    for i, r in d.iterrows():
        t = T[i]
        du = _durations(t)
        ann = _num(r.get("announcement_duration"))            # automatic: script screens
        ann_end = t["b03"] + pd.Timedelta(minutes=ann) if pd.notna(t["b03"]) and not np.isnan(ann) else pd.NaT
        lag = _diff_minutes(ann_end, t["b04"])
        pp = _num(r.get("pause_duration"));  pp = du["pp"] if np.isnan(pp) else pp
        ride = _num(r.get("ride_duration")); ride = du["ride"] if np.isnan(ride) else ride
        start_txt, start_css = _start_check(r)
        sp = _code(r.get("sp_when"))
        order = _order_problems(t)
        issues = []
        if start_css in (RED, AMBER):
            issues.append(start_txt)
        if sp == "3":
            issues.append("Started after strategic point")
        # judged on the value shown (1 decimal), so "2.0" never shows as over a 2-min limit
        bad = lambda x, lo=None, hi=None: not np.isnan(x) and ((lo is not None and round(x, 1) < lo) or
                                                               (hi is not None and round(x, 1) > hi))
        annsu = _ann_signup(r)                         # announcement + sign-up (b03 → b04): the 8–12 min rule
        if bad(annsu, ANN_DUR_MIN, ANN_DUR_MAX):
            issues.append(f"Announcement + sign-up {annsu:.1f} min")
        if bad(du["gap"], hi=DELAY_MAX_MIN):
            issues.append(f"Pitch/pause {du['gap']:.1f} min after sign-up")
        pp_txt = _txt(r.get("pitch_pause_check"))      # treatment-specific rule (rollup.pitch_pause_check)
        if pp_txt:
            issues.append(pp_txt)
        if bad(ride, lo=RIDE_MIN_MIN):
            issues.append(f"Ride {ride:.1f} min")
        issues += [f"Order: {o}" for o in order]
        if _code(r.get("other_marketer")) == "1":
            issues.insert(0, "ANOTHER MARKETER on the bus" + (f" ({OTHER_WHEN_TXT(r.get('other_marketer_when'))})"
                                                              if _txt(r.get("other_marketer_when")) else ""))
        if _code(r.get("other_speaker")) == "1":
            who_ = ", ".join(OTHER_WHO.get(c, c) for c in _txt(r.get("other_speaker_who")).split())
            issues.append("Someone else spoke to passengers" + (f" ({who_})" if who_ else ""))
        row = {
            **{k: v for k, v in _ride_cols(ident.loc[i]).items() if k in ("Ride date", "Team", "Ride #", "Treatment")},
            "Enumerator 1": _txt(r["person"]),
            "Timeline issues": " · ".join(issues) or "None",
            "Independent check": (verdicts or {}).get(_txt(r.get("form_id")), ("No second record", GRAY))[0],
            "Start check": start_txt,
            "Strategic point": {"1": "Before point", "2": "At point", "3": "AFTER point"}.get(sp, "—"),
            "Bus at start": BUS_FULLNESS.get(_code(r.get("b02a")), "—"),
            COL_PREV: fmt_val(prev.get(i, np.nan), 0),
            COL_ANN: fmt_val(annsu, 1),
            COL_LAG: fmt_val(lag, 1),
            COL_SGAP: fmt_val(du["gap"], 1),
            COL_PP: fmt_val(pp, 1),
            COL_RIDE2: fmt_val(ride, 1),
            "Apprentice gave seat": YESNO_SHORT.get(_code(r.get("apprentice_seat")), "—"),
            CLOCK_PREFIX + "Ride start": _clock(t["b02_r"]),
            CLOCK_PREFIX + "Ann. start": _clock(t["b03"]),
            CLOCK_PREFIX + "Sign-up end": _clock(t["b04"]),
            CLOCK_PREFIX + "Pitch/pause start": _clock(t["pp_start"]),
            CLOCK_PREFIX + "Pitch/pause end": _clock(t["pp_end"]),
            CLOCK_PREFIX + "Ride end": _clock(t["b11"]),
        }
        s = {k: "" for k in row}
        s["Timeline issues"] = RED if issues else GREEN
        s["Independent check"] = (verdicts or {}).get(_txt(r.get("form_id")), ("", GRAY))[1]
        row["_check_ok"] = s["Independent check"] in (GREEN, GRAY)
        s["Start check"] = start_css
        if sp == "3":
            s["Strategic point"] = RED
        if bad(round(ann, 1) if not np.isnan(ann) else np.nan, ANN_DUR_MIN, ANN_DUR_MAX):
            s[COL_ANN] = RED
        if bad(du["gap"], hi=DELAY_MAX_MIN):
            s[COL_SGAP] = RED
        if pp_txt:
            s[COL_PP] = RED
        if bad(ride, lo=RIDE_MIN_MIN):
            s[COL_RIDE2] = RED
        if not np.isnan(lag) and lag < -1:
            s[COL_LAG] = RED
        if order:
            for k in row:
                if k.startswith(CLOCK_PREFIX):
                    s[k] = RED
        for k, val in row.items():
            if val in ("—", MISSING) and not s[k]:
                s[k] = GRAY
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows).replace({MISSING: "—"})
    css = pd.DataFrame(sty).reindex(columns=t.columns).fillna("")
    if verdicts is None:
        t, css = t.drop(columns=["Independent check"]), css.drop(columns=["Independent check"])
    for col, src in [("Apprentice gave seat", "apprentice_seat"), ("Strategic point", "sp_when"),
                     (CLOCK_PREFIX + "Sign-up end", "b04"), (COL_LAG, "b04"), (COL_SGAP, "b04")]:
        if not d.get(src, pd.Series(dtype=str)).notna().any() and col in t.columns:
            t = t.drop(columns=[col]); css = css.drop(columns=[col])
    return t, css


# ---------------------------------------------------------------------------
# TAB 2 — Timing cross-check: E1 vs E4 (Script Checklist) vs supervisor
# ---------------------------------------------------------------------------
CHECK_POINTS = [("Ride start", "b02_r"), ("Ann. start", "b03"), ("Sign-up end", "b04"),
                ("Pitch/pause start", "pp_start"), ("Pitch/pause end", "pp_end"), ("Ride end", "b11")]
DUR_LABELS = [("delay", f"Start delay (≤{DELAY_MAX_MIN})"), ("signup", "Ann.+sign-up (b03→b04)"),
              ("gap", f"Sign-up end → pitch/pause (≤{DELAY_MAX_MIN})"),
              ("pp", "Pitch/pause"),
              ("ride", f"Ride ({RIDE_MIN_MIN}+)")]


def _dur_flag(key, x) -> bool:
    if np.isnan(x):
        return False
    x = round(x, 1)                           # judged on the value shown
    if key in ("delay", "gap"):
        return x > DELAY_MAX_MIN
    if key == "pp":
        return abs(x - PITCH_TARGET_MIN) > PITCH_TOL_MIN
    if key == "ride":
        return x < RIDE_MIN_MIN
    return False


def _match_e4(e1_planned: dict, e1_times: dict, e4: pd.DataFrame, e4_ident: pd.DataFrame) -> dict:
    """{E1 row index: E4 row index} — same planned ride; if several, closest ride start."""
    out, used = {}, set()
    for i, code in e1_planned.items():
        cand = [j for j in e4_ident.index if e4_ident.at[j, "planned"] == code and j not in used]
        if not cand:
            continue
        def gap(j):
            g = _diff_minutes(e1_times[i]["b02_r"], _parse_time(e4.at[j, "b02_r"]))
            return abs(g) if not np.isnan(g) else 999
        best = min(cand, key=gap)
        out[i] = best; used.add(best)
    return out


SRC_NAME = {"E4": "the E4", "Sup": "the supervisor"}
STEP_WORDS = {"Ride start": "ride start", "Ann. start": "announcement start", "Sign-up end": "sign-up end",
              "Pitch/pause start": "pitch/pause start", "Pitch/pause end": "pitch/pause end", "Ride end": "ride end"}
_cap = lambda s: s[:1].upper() + s[1:]


def _describe_crosscheck(step_diffs: dict, flips_red: list, flips_amber: list, row: dict,
                         confirmed: list | None = None) -> str:
    """
    Plain-language reading of one ride's comparison. step_diffs = {source: {step: other − E1 minutes}}.
    Positive = the other person recorded that step LATER than the E1.
    """
    if not step_diffs:
        return "No E4 or supervisor record to compare."
    out = []
    big = {src: {k: g for k, g in d.items() if abs(g) > TIME_GAP_MAX} for src, d in step_diffs.items()}
    for src, d in step_diffs.items():
        who, b = SRC_NAME[src], big[src]
        if not d:
            continue
        if not b:
            continue
        vals = list(d.values())
        if len(d) >= 3 and len(b) == len(d) and (max(vals) - min(vals)) <= 2:
            out.append(f"All of {who}'s times are about {np.mean(vals):+.0f} min from the E1's: probably the two "
                       f"phones' clocks differ, not a recording problem.")
            continue
        if len(b) == 1:
            step, g = next(iter(b.items()))
            others_ok = len(d) - 1
            when = "later" if g > 0 else "earlier"
            txt = (f"{_cap(who)} recorded the {STEP_WORDS.get(step, step.lower())} {abs(g):.1f} min {when} than the E1"
                   + (f"; the other {others_ok} time{'s' if others_ok > 1 else ''} agree{'' if others_ok > 1 else 's'}" if others_ok else "") + ".")
            # is the third record on one side?
            third = [s for s in step_diffs if s != src and step in step_diffs[s]]
            if third:
                g3 = step_diffs[third[0]][step]
                if abs(g3) <= TIME_GAP_MAX:
                    txt += f" {_cap(SRC_NAME[third[0]])} agrees with the E1, so {who} probably entered this time wrongly."
                elif abs(g3 - g) <= TIME_GAP_MAX:
                    txt += f" {_cap(SRC_NAME[third[0]])} agrees with {who}, so the E1 probably entered this time wrongly."
            else:
                txt += " Likely one time entered at the wrong moment (by one of the two); the durations using it differ for the same reason."
            out.append(txt)
            continue
        steps = ", ".join(f"{STEP_WORDS.get(k, k.lower())} ({g:+.1f})" for k, g in b.items())
        out.append(f"{_cap(who)} and the E1 disagree at several steps: {steps}. Check both records with them.")
    # duration differences without any single step differing by more than the limit
    dur_cols = [k for k in row if k.startswith("Δ ") and row[k] and k[2:] not in [c for c, _ in CHECK_POINTS]]
    if not out and dur_cols:
        out.append("Small differences at several steps add up to a duration difference of more than 2 min "
                   "(no single time is far off).")
    for f in flips_red:
        out.append(f"By the independent record a target is missed ({f} min) but not by the E1's times, so the E1's times may not match what happened.")
    for f in flips_amber:
        out.append(f"The E1's times miss a target but the independent record doesn't ({f} min): possibly an E1 recording slip.")
    if confirmed:
        out.append("Target missed by both records — " + "; ".join(dict.fromkeys(confirmed))
                   + ": the issue is confirmed by an independent record.")
    return " ".join(out) or "Records agree."


def timing_crosscheck(subs: pd.DataFrame, e4: pd.DataFrame | None, batch_codes: pd.DataFrame | None = None,
                      reviews: pd.DataFrame | None = None):
    """
    One row per E1 form compared with the same ride as recorded by the E4 (Script Checklist)
    and the supervisor (recruitment form, role 0).
      · cells show only DIFFERENCES (other − E1, minutes) larger than TIME_GAP_MAX;
      · target flags are not repeated (Tab 2's timeline has them) — only FLIPS are shown:
        a target missed according to E4/supervisor while the E1's own times look fine (red),
        or the reverse (amber).
    Helper columns: _compared (a second record exists), _issue (any difference or flip).
    Returns (table, css, unmatched E4 forms).
    """
    d = subs[subs["role"] == E1]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ident = ride_identity(d, batch_codes, reviews)
    d = d.loc[_ride_sort_key(ident, d)]
    T = {i: _times(r) for i, r in d.iterrows()}
    pairs = pair_supervisors(subs)
    e4 = e4 if e4 is not None else pd.DataFrame()
    e4_ident = ride_identity(e4, batch_codes, reviews) if not e4.empty else pd.DataFrame()
    m4 = _match_e4({i: ident.at[i, "planned"] for i in d.index}, T, e4, e4_ident) if not e4.empty else {}
    rows, sty = [], []
    for i, r in d.iterrows():
        others = {}
        if i in m4:
            others["E4"] = (e4.loc[m4[i]], _times(e4.loc[m4[i]]))
        if i in pairs:
            s_ = subs.loc[pairs[i][0]]
            others["Sup"] = (s_, _times(s_))
        row = {**{k: v for k, v in _ride_cols(ident.loc[i]).items() if k in ("Ride date", "Team", "Ride #", "Treatment")},
               "Enumerator 1": _txt(r["person"]),
               "E4": _txt(others["E4"][0]["observer"]) if "E4" in others else "—",
               "Supervisor": _txt(others["Sup"][0]["person"]) if "Sup" in others else "—"}
        s = {k: "" for k in row}
        if "E4" not in others: s["E4"] = GRAY
        if "Sup" not in others: s["Supervisor"] = GRAY
        issue = False
        step_diffs = {src: {} for src in others}       # every compared step, for the description
        # 1. clock times at each step: other − E1
        for lbl, key in CHECK_POINTS:
            bits = []
            for src, (_, t_) in others.items():
                g = _diff_minutes(T[i][key], t_[key])
                if not np.isnan(g):
                    step_diffs[src][lbl] = g
                if not np.isnan(g) and abs(g) > TIME_GAP_MAX:
                    bits.append(f"{src} {g:+.1f}")
            row[f"Δ {lbl}"] = " · ".join(bits)
            if bits:
                s[f"Δ {lbl}"] = RED; issue = True
        # 2. durations: other − E1, and target flips
        e1_du = _durations(T[i])
        flips_red, flips_amber, confirmed = [], [], {}
        for key, lbl in DUR_LABELS:
            name = lbl.split(" (")[0]
            bits = []
            for src, (_, t_) in others.items():
                x = _durations(t_)[key]
                if np.isnan(x) or np.isnan(e1_du[key]):
                    continue
                if abs(x - e1_du[key]) > TIME_GAP_MAX:
                    bits.append(f"{src} {x - e1_du[key]:+.1f}")
                if key == "pp":
                    trt = _txt(r.get("treatment_actual"))
                    ppc = lambda tt: pitch_pause_check(trt, tt["b05"], tt["b06"], tt["b09"], tt["b10"], tt["b11"])[0] == 1
                    f_other, f_e1 = ppc(t_), ppc(T[i])
                else:
                    f_other, f_e1 = _dur_flag(key, x), _dur_flag(key, e1_du[key])
                if f_other and f_e1:
                    confirmed.setdefault(name, [f"E1 {e1_du[key]:.1f}"]).append(f"{src} {x:.1f}")
                if f_other and not f_e1:
                    flips_red.append(f"{name}: {src} {x:.1f} vs E1 {e1_du[key]:.1f}")
                elif f_e1 and not f_other:
                    flips_amber.append(f"{name}: E1 {e1_du[key]:.1f} vs {src} {x:.1f}")
            row[f"Δ {name}"] = " · ".join(bits)
            if bits:
                s[f"Δ {name}"] = RED; issue = True
        row["Target missed per E4/Sup, E1 looks fine"] = " · ".join(flips_red)
        row["E1 outside target, E4/Sup says fine"] = " · ".join(flips_amber)
        if flips_red:
            s["Target missed per E4/Sup, E1 looks fine"] = RED; issue = True
        if flips_amber:
            s["E1 outside target, E4/Sup says fine"] = AMBER; issue = True
        # short verdict for the ride timeline ("Independent check") and Today
        srcs_txt = " + ".join(SRC_NAME[s].replace("the ", "") for s in others) if others else ""
        big_steps = [(src, k, g) for src, d_ in step_diffs.items() for k, g in d_.items() if abs(g) > TIME_GAP_MAX]
        dur_diff = [k for k in row if k.startswith("Δ ") and row[k] and k[2:] not in [c for c, _ in CHECK_POINTS]]
        if not others:
            verdict, vcss = "No second record", GRAY
        elif flips_red:
            verdict, vcss = " · ".join(f"{f.split(':')[0]}: {f.split(': ', 1)[1].split(' vs ')[0]} min, E1 looks fine"
                                       for f in flips_red), RED
        elif big_steps or flips_amber or dur_diff:
            bits = [f"{src} differs on {STEP_WORDS.get(k, k)} ({g:+.1f})" for src, k, g in big_steps[:2]]
            bits += [f"E1 outside target, {f.split(': ', 1)[1].split(' vs ')[1].split(' ')[0]} within it ({f.split(':')[0]})"
                     for f in flips_amber]
            if not bits:
                bits = ["Durations differ by > 2 min (" + ", ".join(k[2:] for k in dur_diff) + ")"]
            verdict, vcss = " · ".join(bits), AMBER
        elif confirmed:
            verdict, vcss = f"Confirmed by {srcs_txt}: " + ", ".join(k.lower() for k in confirmed) + " also outside target", GREEN
        else:
            verdict, vcss = f"Agrees ({srcs_txt})", GREEN
        row["_verdict"], row["_verdict_css"] = verdict, vcss
        row["_flip"] = bool(flips_red)
        row["_form_id"] = _txt(r.get("form_id"))
        row["_planned"] = _txt(ident.at[i, "planned"])
        row["What happened"] = _describe_crosscheck(step_diffs, flips_red, flips_amber, row,
                                                    [f"{n} ({', '.join(vals)} min)" for n, vals in confirmed.items()])
        row["_compared"] = bool(others)
        row["_issue"] = issue
        rows.append(row); sty.append(s)
    t, css = pd.DataFrame(rows), pd.DataFrame(sty).reindex(columns=pd.DataFrame(rows).columns).fillna("")
    # drop difference columns that are empty for every ride
    empty = [c for c in t.columns if c.startswith("Δ ") and (t[c] == "").all()]
    t, css = t.drop(columns=empty), css.drop(columns=empty)
    # E4 checklists that didn't match any E1 form, with the likely reason
    if not e4.empty:
        left = e4.loc[[j for j in e4.index if j not in set(m4.values())]]
        planned_e1 = set(ident["planned"])
        planned_all = set(planned_codes(batch_codes)) if batch_codes is not None else set()
        def why(j):
            pc = e4_ident.at[j, "planned"] if j in e4_ident.index else ""
            if not _txt(pc):
                return "No planned code selected"
            if planned_all and pc not in planned_all:
                return "Planned code not in batch_codes"
            if pc not in planned_e1:
                return "No E1 form for this ride (in the current filters)"
            return "E1 form already matched to another checklist (duplicate?)"
        unmatched = pd.DataFrame({
            "Ride date": left["ride_date"].map(fmt_day), "Observer": left["observer"],
            "Role": left["observer_role"], "Planned code": [e4_ident.at[j, "planned"] if j in e4_ident.index else "" for j in left.index],
            "Actual code": left["actual_batch_code"].map(_txt), "E1 named": left["e1_name"].map(_txt),
            "Why not matched": [why(j) for j in left.index],
        }).reset_index(drop=True)
    else:
        unmatched = pd.DataFrame()
    return t, css, unmatched

def media_url(filename, form_id, server: str | None, surveycto_form: str = "rec_form_nov2026") -> str | None:
    """
    Link to a photo/audio file of one submission, in the format SurveyCTO uses for
    attachments in its own exports (API v2). Opens in a browser that is logged in to
    SurveyCTO with access to the form's data.
    """
    from urllib.parse import quote
    f = _txt(filename)
    if not f or not server or not _txt(form_id):
        return None
    f = f.split("/")[-1]
    return (f"https://{server}.surveycto.com/api/v2/forms/{surveycto_form}/submissions/"
            f"{quote(_txt(form_id), safe='')}/attachments/{quote(f)}")


def media_table(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                reviews: pd.DataFrame | None = None, server: str | None = None,
                surveycto_form: str = "rec_form_nov2026") -> pd.DataFrame:
    """One row per E1 form with links to its photos and audio audits (or file names if no server is set)."""
    d = subs[subs["role"] == E1]
    present = [(c, lbl) for c, lbl in MEDIA_FIELDS if c in d.columns and d[c].notna().any()]
    if d.empty or not present:
        return pd.DataFrame()
    ident = ride_identity(d, batch_codes, reviews)
    d = d.loc[_ride_sort_key(ident, d)]
    rows = []
    for i, r in d.iterrows():
        row = {k: v for k, v in _ride_cols(ident.loc[i]).items() if k in ("Ride date", "Team", "Ride #")}
        row["Enumerator 1"] = _txt(r["person"])
        for c, lbl in present:
            row[lbl] = media_url(r.get(c), r.get("form_id"), server, surveycto_form) if server else (_txt(r.get(c)).split("/")[-1] or None)
        rows.append(row)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# TAB 3 — Enumerators: performance indicators and averages
# ---------------------------------------------------------------------------
METRICS = [   # (column in ride_facts, label, decimals, threshold for "far from the average")
    ("signups",  "Sign-ups",                         1, 3.0),
    ("delay",    "Start delay (min)",                1, 1.0),
    ("ann",      f"Announcement + sign-up (min, {ANN_DUR_MIN}–{ANN_DUR_MAX})", 1, 1.0),
    ("script",   "Script screens (min)",             1, 1.0),
    ("lag",      "Ann. end → sign-up end (min)",     1, 1.0),
    ("gap",      "Sign-up end → pitch/pause (min)",  1, 1.0),
    ("pitch",    "Pitch, Normal/Peddling (min)",     1, 3.0),
    ("pause",    "Pause, Sit down (min)",            1, 3.0),
    ("ride",     "Ride (min)",                       1, 5.0),
    ("between",  "Between rides (min)",              0, 10.0),
]
GROUP_BY = {"Enumerator 1": "person", "Route (corridor)": "corridor", "Direction": "direction",
            "Team": "team", "Treatment": "treatment", "Marketer": "marketer", "Day": "day"}


def ride_facts(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
               reviews: pd.DataFrame | None = None, verdicts: dict | None = None) -> pd.DataFrame:
    """
    One row per E1 form: the groupers (enumerator, route, direction, team, treatment, marketer, day),
    the averaged measures and the counted performance indicators (True/False/None = not assessed).
    """
    d = subs[subs["role"] == E1]
    if d.empty:
        return pd.DataFrame()
    ident = ride_identity(d, batch_codes, reviews)
    rmap = review_map(reviews)
    rs = ride_summary(subs, batch_codes, reviews)
    rs_by = {f: r for f, r in zip(rs["form_id"], rs.to_dict("records"))} if not rs.empty else {}
    T = {i: _times(r) for i, r in d.iterrows()}
    # time since the team's previous ride, same day
    order = sorted(d.index, key=lambda i: (str(ident.at[i, "date"]), str(ident.at[i, "team"]),
                                           T[i]["b02_r"] if pd.notna(T[i]["b02_r"]) else pd.Timestamp.max))
    between = {}
    for a_, b_ in zip(order, order[1:]):
        if ident.at[a_, "date"] == ident.at[b_, "date"] and str(ident.at[a_, "team"]) == str(ident.at[b_, "team"]):
            between[b_] = _diff_minutes(T[a_]["b11"], T[b_]["b02_r"])
    rows = []
    for i, r in d.iterrows():
        t, du = T[i], _durations(T[i])
        trt = _txt(r.get("treatment_actual")) or _txt(r.get("a_treatment")) or _txt(r.get("p_treatment"))
        ann = _num(r.get("announcement_duration"))
        ann_end = t["b03"] + pd.Timedelta(minutes=ann) if pd.notna(t["b03"]) and not np.isnan(ann) else pd.NaT
        route = _txt(r.get("a_route")) or _txt(r.get("p_route"))
        sr = rs_by.get(_txt(r.get("form_id")), {})
        code = _txt(ident.at[i, "planned"])
        decisions = {x["decision"] for x in rmap.get(code, {}).values()}
        start_txt, start_css = _start_check(r)
        sp = _code(r.get("sp_when"))
        vd = (verdicts or {}).get(_txt(r.get("form_id")))
        mkt = _txt(r.get("mm_act_mkt")) or _txt(r.get("a_mkt_id")) or _txt(r.get("a03_r"))
        rows.append({
            "form_id": _txt(r.get("form_id")), "person": _txt(r["person"]),
            "date": ident.at[i, "date"], "day": fmt_day(ident.at[i, "date"]),
            "team": str(ident.at[i, "team"] or ""), "ride_no": str(ident.at[i, "ride"] or ""),
            "direction": route, "corridor": "–".join(sorted([route[:2], route[2:]])) if len(route) == 4 else route,
            "treatment": TREAT.get(trt, trt), "marketer": mkt,
            # measures
            "signups": _num(r.get("total_signups")), "delay": _num(r.get("announcement_start_delay")),
            "ann": _ann_signup(r), "script": ann, "lag": _diff_minutes(ann_end, t["b04"]), "gap": du["gap"],
            "pitch": du["pp"] if trt in ("N", "P") else np.nan,
            "pause": du["pp"] if trt == "S" else np.nan,
            "ride": du["ride"], "between": between.get(i, np.nan),
            "bus_half": None if _code(r.get("b02a")) not in ("1", "2", "3", "4") else _code(r.get("b02a")) != "1",
            "seat": None if _code(r.get("apprentice_seat")) not in ("0", "1") else _code(r.get("apprentice_seat")) == "1",
            "other_mkt": None if _code(r.get("other_marketer")) not in ("0", "1") else _code(r.get("other_marketer")) == "1",
            "other_spk": None if _code(r.get("other_speaker")) not in ("0", "1") else _code(r.get("other_speaker")) == "1",
            # counted indicators (None = not assessed)
            "code_ok": not (sr.get("wrong_code", False) or decisions & {"Fix needed", "Fixed"}),
            "same_day": sr.get("same_day"),
            "order_ok": not _order_problems(t),
            "indep_ok": None if not vd or vd[1] == GRAY else vd[1] == GREEN,
            "start_ok": None if start_css == GRAY else start_css == GREEN,
            "sp_ok": None if not sp else sp != "3",
            "ann_ok": (None if np.isnan(_ann_signup(r)) else ANN_DUR_MIN <= round(_ann_signup(r), 1) <= ANN_DUR_MAX),
            "obs_yes": sr.get("obs_yes", 0), "obs_n": sr.get("obs_n", 0),
            "grade": sr.get("grade", np.nan),
            "no_rec_err": None if not sr.get("supervised") else not sr.get("rec_err", False),
            "counts_ok": sr.get("counts_verified"),
        })
    return pd.DataFrame(rows)


INDICATORS = [   # (column, label)
    ("code_ok",    "Correct batch code"),
    ("same_day",   "Sent same day"),
    ("order_ok",   "Times in logical order"),
    ("indep_ok",   "Confirmed by E4/supervisor"),
    ("start_ok",   "Start on time / justified"),
    ("sp_ok",      "At/before strategic point"),
    ("ann_ok",     f"Announcement + sign-up {ANN_DUR_MIN}–{ANN_DUR_MAX} min"),
    ("no_rec_err", "No recording error (sup.)"),
    ("counts_ok",  "Counts verified (sup.)"),
]


def enumerator_indicators(f: pd.DataFrame):
    """One row per enumerator: share of rides meeting each counted indicator. Returns (table, css)."""
    if f is None or f.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, sty = [], []
    for p, g in f.groupby("person", sort=True):
        row, s = {"Enumerator 1": p, "Rides": len(g)}, {"Enumerator 1": "", "Rides": ""}
        for col, lbl in INDICATORS:
            vals = [x for x in g[col] if x is not None and not (isinstance(x, float) and np.isnan(x))]
            n, yes = len(vals), sum(bool(x) for x in vals)
            row[lbl] = fmt_rate(yes, n)
            s[lbl] = GRAY if not n else (RED if yes / n < PERF_TARGET else GREEN if yes == n else "")
        n_obs, y_obs = g["obs_n"].sum(), g["obs_yes"].sum()
        row["Checklist (sup.)"] = fmt_rate(y_obs, n_obs)
        s["Checklist (sup.)"] = GRAY if not n_obs else (GREEN if y_obs / n_obs >= 0.93 else
                                                         AMBER if y_obs / n_obs >= 0.88 else RED)
        gr = g["grade"].dropna()
        row["Grade (sup., 1–10)"] = f"{gr.mean():.1f} ({len(gr)})" if len(gr) else "—"
        s["Grade (sup., 1–10)"] = "" if len(gr) else GRAY
        rows.append(row); sty.append(s)
    return pd.DataFrame(rows), pd.DataFrame(sty)


AVG_SD_AMBER, AVG_SD_RED = 0.5, 1.0   # distance from the overall average, in ride-level standard deviations
AVG_MIN_RIDES = 3                     # groups with fewer rides are shown but not coloured


def _sd_style(x: float, mean: float, sd: float, n: int, higher_is_better: bool = False) -> tuple[str, str]:
    """(css, arrow) for a group average x: how many ride-level SDs it is from the overall mean.
    higher_is_better (sign-ups): above the average is shaded green instead of amber/red."""
    if np.isnan(x) or np.isnan(mean) or not sd or np.isnan(sd) or n < AVG_MIN_RIDES:
        return "", ""
    z = (x - mean) / sd
    arrow = " ▲" if z > 0 else " ▼"
    if higher_is_better and z > 0:
        if z > AVG_SD_RED:
            return f"background-color: {GREEN_BG}; font-weight: 600", arrow
        if z > AVG_SD_AMBER:
            return f"background-color: {GREEN_BG}", arrow
        return "", ""
    if abs(z) > AVG_SD_RED:
        return f"background-color: {RED_BG}; font-weight: 600", arrow
    if abs(z) > AVG_SD_AMBER:
        return f"background-color: {AMBER_BG}", arrow
    return "", ""


def averages_table(f: pd.DataFrame, by: str = "person"):
    """
    Averages of each measure by group (enumerator, route, direction, team, treatment, marketer, day),
    with the overall average in the last row. Each cell is shaded by how far the group's average is
    from the overall average, measured in standard deviations of the individual rides:
    amber = 0.5–1 SD away, red = more than 1 SD away (▲ above / ▼ below). Groups with fewer than
    AVG_MIN_RIDES rides are not shaded. Returns (table, css).
    """
    if f is None or f.empty:
        return pd.DataFrame(), pd.DataFrame()
    label = {v_: k for k, v_ in GROUP_BY.items()}.get(by, by)
    overall = {m: f[m].mean() for m, *_ in METRICS}
    spread = {m: f[m].std(ddof=1) for m, *_ in METRICS}
    rows, sty = [], []
    for key, g in f.groupby(by, sort=True):
        row, s = {label: key or "—", "Rides": len(g)}, {label: "", "Rides": ""}
        for m, lbl, nd, _thr in METRICS:
            x = g[m].mean()
            css_, arrow = _sd_style(x, overall[m], spread[m], int(g[m].notna().sum()), higher_is_better=(m == "signups"))
            row[lbl] = "—" if np.isnan(x) else f"{x:.{nd}f}{arrow}"
            s[lbl] = GRAY if np.isnan(x) else css_
        rows.append(row); sty.append(s)
    tot = {label: "All rides", "Rides": len(f)}
    for m, lbl, nd, _ in METRICS:
        tot[lbl] = "—" if np.isnan(overall[m]) else f"{overall[m]:.{nd}f}"
    rows.append(tot); sty.append({k: "font-weight: 700" for k in tot})
    sd_row = {label: "1 SD (spread of rides)", "Rides": ""}
    for m, lbl, nd, _ in METRICS:
        sd_row[lbl] = "—" if np.isnan(spread[m]) else f"±{spread[m]:.{nd}f}"
    rows.append(sd_row); sty.append({k: GRAY for k in sd_row})
    t, css = pd.DataFrame(rows), pd.DataFrame(sty).reindex(columns=pd.DataFrame(rows).columns).fillna("")
    empty = [lbl for m, lbl, *_ in METRICS if (t[lbl].iloc[:-1] == "—").all()]
    return t.drop(columns=empty), css.drop(columns=empty)


def e2e3_signups(subs: pd.DataFrame, f: pd.DataFrame, by: str = "person"):
    """
    Normal sign-up cards collected by each Enumerator 2 / 3 (E1's count on the E1 form), optionally split
    by route, direction, team, treatment, marketer or day. Returns (table, css).
    """
    d = subs[subs["role"] == E1]
    if d.empty or f is None or f.empty:
        return pd.DataFrame(), pd.DataFrame()
    fx = f.set_index("form_id")
    rows = []
    for _, r in d.iterrows():
        fid = _txt(r.get("form_id"))
        if fid not in fx.index:
            continue
        for who_col, n_col in [("a02_2_lab", "c01_normal_e2"), ("a02_3_lab", "c01_normal_e3")]:
            name, n = _txt(r.get(who_col)), _num(r.get(n_col))
            if name and not np.isnan(n):
                row = {"Enumerator 2/3": name, "n": n}
                for k, col in GROUP_BY.items():
                    row[col] = fx.at[fid, col] if col in fx.columns else ""
                rows.append(row)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    x = pd.DataFrame(rows)
    keys = ["Enumerator 2/3"] + ([by] if by != "person" else [])
    g = x.groupby(keys).agg(Rides=("n", "size"), avg=("n", "mean")).reset_index()
    g = g.rename(columns={"avg": "Avg normal sign-ups", by: {v_: k for k, v_ in GROUP_BY.items()}.get(by, by)})
    overall, sd = x["n"].mean(), x["n"].std(ddof=1)
    css = pd.DataFrame("", index=g.index, columns=g.columns)
    vals = []
    for k_, (avg, n) in enumerate(zip(g["Avg normal sign-ups"], g["Rides"])):
        c_, arrow = _sd_style(avg, overall, sd, int(n), higher_is_better=True)
        css.iat[k_, list(g.columns).index("Avg normal sign-ups")] = c_
        vals.append(f"{avg:.1f}{arrow}")
    g["Avg normal sign-ups"] = vals
    return g, css



# ---------------------------------------------------------------------------
# TAB 3 — Balance between treatment arms
# ---------------------------------------------------------------------------
# Everything the E1 and the ride conditions produce before/around the pitch should look the same in
# every arm; only the pitch itself differs. Sit down is the control arm.
BALANCE_ROWS = [   # (column, label, kind)
    ("signups",  "Sign-ups per ride",                 "mean"),
    ("delay",    "Start delay (min)",                 "mean"),
    ("ann",      "Announcement + sign-up (min)",      "mean"),
    ("lag",      "Ann. end → sign-up end (min)",      "mean"),
    ("gap",      "Sign-up end → pitch/pause (min)",   "mean"),
    ("ride",     "Ride (min)",                        "mean"),
    ("between",  "Between rides (min)",               "mean"),
    ("bus_half", "Bus at least half full at start",   "share"),
    ("start_ok", "Start on time / justified",         "share"),
    ("ann_ok",   f"Announcement + sign-up {ANN_DUR_MIN}–{ANN_DUR_MAX} min", "share"),
    ("seat",     "Apprentice gave up seat",           "share"),
    ("other_mkt", "Another marketer on the bus",      "share"),
    ("other_spk", "Someone else spoke to passengers", "share"),
]
BALANCE_MIN_N = 5          # rides per arm before a difference is coloured
BALANCE_SD_AMBER, BALANCE_SD_RED = 0.25, 0.5


def _arm_values(f: pd.DataFrame, col: str, arm: str) -> np.ndarray:
    x = f.loc[f["treatment"] == arm, col]
    x = x[x.notna()]
    return np.array([float(bool(v_)) if isinstance(v_, (bool, np.bool_)) else float(v_) for v_ in x], dtype=float)


def balance_table(f: pd.DataFrame, control: str = "Sit down", within: str | None = None):
    """
    within = None → all rides together; or a grouping column ("corridor", "direction", "team",
    "person", …) → the comparison is made separately inside each group (e.g. each route), with the
    group as the first column. Returns (table, css).
    """
    if within and f is not None and not f.empty and within in f.columns:
        label = {v_: k for k, v_ in GROUP_BY.items()}.get(within, within)
        parts, cparts = [], []
        for key, g in f.groupby(within, sort=True):
            if g["treatment"].nunique() < 2 or control not in set(g["treatment"]):
                continue                       # nothing to compare inside this group
            t_, c_ = _balance_one(g, control)
            if t_.empty or t_.shape[1] <= 2:
                continue
            t_.insert(0, label, key or "—"); c_.insert(0, label, "font-weight: 700")
            parts.append(t_); cparts.append(c_)
        if not parts:
            return pd.DataFrame(), pd.DataFrame()
        t = pd.concat(parts, ignore_index=True)
        css = pd.concat(cparts, ignore_index=True).reindex(columns=t.columns).fillna("")
        t = t.fillna("—")
        return t, css
    return _balance_one(f, control)


def _balance_one(f: pd.DataFrame, control: str = "Sit down"):
    """
    Each measure by treatment arm, with the difference from the control arm and the standardised
    difference (difference ÷ pooled SD). |std. diff| > 0.25 amber, > 0.5 red, once each arm has
    BALANCE_MIN_N rides. Returns (table, css).
    """
    if f is None or f.empty or "treatment" not in f.columns:
        return pd.DataFrame(), pd.DataFrame()
    arms = [a for a in ["Sit down", "Normal", "Peddling"] if (f["treatment"] == a).any()]
    others = [a for a in arms if a != control]
    rows, sty = [], []
    for col, lbl, kind in BALANCE_ROWS:
        if col not in f.columns:
            continue
        ctrl = _arm_values(f, col, control)
        row, s = {"Measure": lbl}, {"Measure": ""}
        fmt = (lambda x: f"{x:.0%}") if kind == "share" else (lambda x: f"{x:.1f}")
        any_val = False
        for a in arms:
            x = _arm_values(f, col, a)
            row[f"{a}{' (control)' if a == control else ''}"] = f"{fmt(x.mean())} ({len(x)})" if len(x) else "—"
            s[f"{a}{' (control)' if a == control else ''}"] = "" if len(x) else GRAY
            any_val |= bool(len(x))
        for a in others:
            x = _arm_values(f, col, a)
            dkey, skey = f"{a} − control", f"Std. diff ({a})"
            if len(x) and len(ctrl):
                diff = x.mean() - ctrl.mean()
                sd = np.sqrt((x.var(ddof=1) if len(x) > 1 else 0) / 2 + (ctrl.var(ddof=1) if len(ctrl) > 1 else 0) / 2)
                std = diff / sd if sd > 0 else (0.0 if diff == 0 else float(np.sign(diff)) * 9.99)   # no spread: any gap is total
                row[dkey] = (f"{diff * 100:+.0f} pp" if kind == "share" else f"{diff:+.1f}")
                row[skey] = f"{std:+.2f}"
                enough = len(x) >= BALANCE_MIN_N and len(ctrl) >= BALANCE_MIN_N
                st_ = GRAY if not enough else RED if abs(std) > BALANCE_SD_RED else AMBER if abs(std) > BALANCE_SD_AMBER else GREEN
                s[dkey], s[skey] = (st_ if st_ != GREEN else ""), st_
            else:
                row[dkey], row[skey] = "—", "—"
                s[dkey], s[skey] = GRAY, GRAY
        if any_val:
            rows.append(row); sty.append(s)
    t = pd.DataFrame(rows)
    css = pd.DataFrame(sty).reindex(columns=t.columns).fillna("")
    return t, css



# ---------------------------------------------------------------------------
# Column descriptions (shown when hovering over a column header, and under the tables)
# ---------------------------------------------------------------------------
COLUMN_HELP = {
    # performance indicators
    "Rides": "Number of rides in the selection",
    "Correct batch code": "Right planned code selected and ride details (date, team, ride #) entered correctly. "
                          "Allowed changes (team, marketer), changes you marked OK, and not-allowed changes the E1 "
                          "reported to the supervisor don't count against them.",
    "Sent same day": "The form was submitted on the day of the ride",
    "Times in logical order": "The times follow the order of the ride: ride start → announcement start → sign-up end "
                              "→ pitch/pause start → pitch/pause end → ride end. Not met if a later step is recorded "
                              "before an earlier one (e.g. pitch end before pitch start) — a sign of times entered "
                              "wrongly.",
    "Confirmed by E4/supervisor": "The E4 (Script Checklist) or supervisor recorded the same times (within 2 min) and "
                                  "doesn't miss a target the E1's times meet. Only rides with a second record.",
    "Start on time / justified": "Announcement started within 2 min of the ride start, or later because the E1 "
                                 "waited for the bus to fill (less than half full at the start)",
    "At/before strategic point": "When the start was late: the announcement began before or at the route's "
                                 "strategic point, not after it",
    f"Announcement + sign-up {ANN_DUR_MIN}–{ANN_DUR_MAX} min": "From the start of the announcement to the end of sign-up "
                                                               f"(b03 → b04) between {ANN_DUR_MIN} and {ANN_DUR_MAX} minutes "
                                                               "(target 10)",
    "No recording error (sup.)": "The supervisor did not attribute a timing mismatch to the E1. Supervised rides only.",
    "Counts verified (sup.)": "The supervisor's count of sign-ups (and tickets) matched the E1's. Supervised rides only.",
    "Checklist (sup.)": "Share of the supervisor's checklist behaviours (s01–s26) the E1 did, pooled over rides",
    "Grade (sup., 1–10)": "Supervisor's overall grade for the E1 (average; number of rides graded in brackets)",
    # averages
    "Sign-ups": "Average normal sign-ups per ride (total on the E1 form)",
    "Start delay (min)": "Minutes from the bus leaving the park to the start of the announcement",
    f"Announcement + sign-up (min, {ANN_DUR_MIN}–{ANN_DUR_MAX})": "Start of the announcement → end of sign-up (b03 → b04); "
                                                                  "target 10 minutes, allowed 8–12",
    "Script screens (min)": "Time spent on the script screens (the announcement itself, measured by the form)",
    f"Announcement + sign-up ({ANN_DUR_MIN}–{ANN_DUR_MAX} min)": "Start of the announcement → end of sign-up (b03 → b04); "
                                                                "target 10 minutes, allowed 8–12",
    "Ann. end → sign-up end (min)": "Minutes between the end of the script screens and the end of sign-up "
                                    "(the gap: time spent collecting sign-ups after the script)",
    "Sign-up end → pitch/pause (min)": "Minutes between the end of sign-up and the start of the marketer's pitch "
                                       "(Normal/Peddling) or of the pause (Sit down); should be 2 or less",
    "Pitch, Normal/Peddling (min)": "Length of the marketer's pitch (pitch start → pitch end)",
    "Pause, Sit down (min)": "Length of the pause (pause start → ride end)",
    "Ride (min)": "Ride length (bus leaves the park → ride end)",
    "Between rides (min)": "Minutes from the end of the team's previous ride to the start of this one, same day",
    # balance
    "Measure": "What is compared across treatment arms",
    "Sit down (control)": "Average or share in the Sit down (control) arm; rides in brackets",
    "Normal": "Average or share in the Normal arm; rides in brackets",
    "Peddling": "Average or share in the Peddling arm; rides in brackets",
    "Normal − control": "Normal minus Sit down (minutes, sign-ups, or percentage points)",
    "Peddling − control": "Peddling minus Sit down (minutes, sign-ups, or percentage points)",
    "Std. diff (Normal)": "Difference ÷ pooled standard deviation: about 0 = balanced; beyond ±0.25 worth a look; "
                          "beyond ±0.5 a real imbalance",
    "Std. diff (Peddling)": "Difference ÷ pooled standard deviation: about 0 = balanced; beyond ±0.25 worth a look; "
                            "beyond ±0.5 a real imbalance",
    # E2/E3
    "Enumerator 2/3": "Enumerator 2 or 3 on the ride",
    "Avg normal sign-ups": "Average normal sign-up cards collected per ride (E1's count of their cards)",
    # ride timeline
    "Timeline issues": "Everything on this ride outside its target, in one line",
    "Independent check": "Comparison with the E4 (Script Checklist) and supervisor records of the same ride",
    "Start check": "Did the announcement start on time, or was the delay justified by a bus less than half full?",
    "Strategic point": "When the start was late: did the announcement begin before, at or after the route's "
                       "strategic point?",
    "Bus at start": "How full the bus was when it left the park",
    "Since team's previous ride (min)": "Minutes from the end of the team's previous ride to the start of this one",
    "Apprentice gave seat": "Did the apprentice give up their seat for the recruitment team?",
    "Another marketer on the bus": "Share of rides where another marketer got on and marketed — this can invalidate the ride",
    "Someone else spoke to passengers": "Share of rides where a preacher, seller or anyone else spoke to all passengers",
}



# ---------------------------------------------------------------------------
# TAB 4 — Patterns: trends over time and the full export
# ---------------------------------------------------------------------------
def trend_table(f: pd.DataFrame, measure: str, by: str = "person", period: str = "day") -> pd.DataFrame:
    """Average of one measure per day (or week) and group, wide: index = period, columns = groups."""
    if f is None or f.empty or measure not in f.columns:
        return pd.DataFrame()
    x = f[[by, "date", measure]].copy()
    x = x[x["date"].notna() & x[measure].notna()]
    if x.empty:
        return pd.DataFrame()
    x[measure] = x[measure].astype(float)
    if period == "week":
        x["period"] = x["date"] - pd.to_timedelta(x["date"].dt.weekday, unit="D")
    else:
        x["period"] = x["date"]
    t = x.pivot_table(index="period", columns=by, values=measure, aggfunc="mean").sort_index()
    t.index = pd.to_datetime(t.index)
    return t


def _plain(t: pd.DataFrame) -> pd.DataFrame:
    """Drop helper columns and arrows for export."""
    if t is None or t.empty:
        return pd.DataFrame()
    t = t[[c for c in t.columns if not str(c).startswith("_")]].copy()
    return t.apply(lambda col: col.map(lambda x: x.replace(" ▲", "").replace(" ▼", "") if isinstance(x, str) else x))


def patterns_export(f: pd.DataFrame, subs: pd.DataFrame) -> dict:
    """{sheet name: table} with every view of the Patterns tab, for Excel / Google Sheet export."""
    out = {}
    for lbl, col in GROUP_BY.items():
        t, _ = averages_table(f, col)
        out[f"Averages by {lbl}"[:31]] = _plain(t)
    for lbl, within in [("Balance - all rides", None), ("Balance - within route", "corridor"),
                        ("Balance - within direction", "direction"), ("Balance - within team", "team")]:
        t, _ = balance_table(f, within=within)
        out[lbl[:31]] = _plain(t)
    t, _ = e2e3_signups(subs, f, "person")
    out["E2-E3 sign-ups"] = _plain(t)
    # trends: long format (day × enumerator × measure)
    long = []
    for m, lbl, *_ in METRICS:
        tt = trend_table(f, m, "person", "day")
        if not tt.empty:
            s = tt.stack().reset_index()
            s.columns = ["Day", "Enumerator 1", "Average"]
            s["Measure"] = lbl
            long.append(s)
    if long:
        tr = pd.concat(long, ignore_index=True)[["Measure", "Day", "Enumerator 1", "Average"]]
        tr["Day"] = tr["Day"].dt.strftime("%Y-%m-%d"); tr["Average"] = tr["Average"].round(2)
        out["Trends by day"] = tr
    rides = f.drop(columns=[c for c in f.columns if c in ("form_id",)]).copy()
    rides["date"] = rides["date"].dt.strftime("%Y-%m-%d")
    out["Rides (one row each)"] = rides
    return {k: v_ for k, v_ in out.items() if v_ is not None and not v_.empty}


def to_excel_bytes(tables: dict) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for col_cells in ws.columns:
                width = max(len(str(c.value or "")) for c in col_cells[:200])
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, width + 2), 45)
            ws.freeze_panes = "B2"
    return buf.getvalue()



def media_index(subs: pd.DataFrame, batch_codes: pd.DataFrame | None = None,
                reviews: pd.DataFrame | None = None) -> list:
    """[{label, form_id, files: [(field label, filename)]}] for E1 forms with photos or audio, in ride order."""
    d = subs[subs["role"] == E1]
    present = [(c, lbl) for c, lbl in MEDIA_FIELDS if c in d.columns and d[c].notna().any()]
    if d.empty or not present:
        return []
    ident = ride_identity(d, batch_codes, reviews)
    d = d.loc[_ride_sort_key(ident, d)]
    out = []
    for i, r in d.iterrows():
        files = [(lbl, _txt(r.get(c)).split("/")[-1]) for c, lbl in present if _txt(r.get(c))]
        if not files:
            continue
        rc = _ride_cols(ident.loc[i])
        out.append({"label": f"{rc['Ride date']} · team {rc['Team']} · ride {rc['Ride #']} · {_txt(r['person'])}"
                             f" · {rc['Treatment']}",
                    "form_id": _txt(r.get("form_id")), "files": files})
    return out
