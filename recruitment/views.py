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
    parse_batch_code, _parse_time, _diff_minutes, _code,
    DELAY_MAX_MIN, ANN_DUR_MIN, ANN_DUR_MAX, SIGNUP_DIFF_MAX,
    PITCH_TARGET_MIN, PITCH_TOL_MIN, RIDE_MIN_MIN, SIGNUP_MIN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEWS_VERSION = "2026-10-07"   # app.py checks this to catch an out-of-date views.py

E1, SUP = "Enumerator 1", "Supervisor"
MISSING = "Missing/not assessed"

RED  = "color: #CC0000; font-weight: 600"
BLUE = "color: #0055AA; font-weight: 600"
GRAY = "color: #888888"
GREEN = "color: #1E7B34; font-weight: 600"
PURPLE = "color: #7B2CBF; font-weight: 600"

TIME_GAP_MAX    = 2    # min — E1 vs supervisor checkpoint gap that needs review
PAIR_WINDOW_MIN = 20   # min — supervisor form paired with an E1 form if ride starts are this close
PERF_TARGET     = 0.80 # performance rates below this are shown in red

TREAT = {"S": "Script", "N": "Normal", "P": "Peddling"}

STATUS_RANK = {
    "Wrong code selected": 0, "Design problem": 1, "Duplicate E1": 2, "To confirm": 3,
    "Supervisor only": 4, "Not submitted": 5, "Done as planned": 6,
}
STATUS_STYLE = {
    "Wrong code selected": RED, "Design problem": RED, "Duplicate E1": RED, "To confirm": BLUE,
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
    "Preparation":   ["s01"],
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
]
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
    ("Ann. end",            ["b05", "b06"]),
    ("Pitch/pause end",     ["b09", "b10"]),
    ("Ride end",            ["b11"]),
]

# Column names shared with app.py (styling) --------------------------------
COL_DELAY   = f"Start delay (<{DELAY_MAX_MIN} min)"
COL_REC     = f"Recorded ann./signup dur. ({ANN_DUR_MIN}–{ANN_DUR_MAX} min)"
COL_AUTO    = "Auto form-screen dur."
COL_DISCREP = f"Timing discrepancy (>{SIGNUP_DIFF_MAX} min)"
COL_PITCH   = f"Pause/Pitch ({PITCH_TARGET_MIN}±{PITCH_TOL_MIN} min)"
COL_RIDE    = f"Ride dur. (≥{RIDE_MIN_MIN} min)"
COL_SIGNUPS = f"Signups (≥{SIGNUP_MIN})"

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
        out.append(("confirm", f"Date {fmt_code_date(g('p_date'))}→{fmt_code_date(g('a_date'))}", "Date"))
    if g("p_team") != g("a_team"):
        out.append(("confirm", f"Team {g('p_team')}→{g('a_team')}", "Team"))
    if g("p_mkt_id") != g("a_mkt_id"):
        out.append(("allowed", f"Mkt ID {g('p_mkt_id')}→{g('a_mkt_id')} (allowed)", "Mkt ID"))
    if g("p_ride_num") != g("a_ride_num"):
        out.append(("confirm", f"Ride # {g('p_ride_num')}→{g('a_ride_num')}", "Ride #"))
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
DESIGN_PARTS  = {"Treatment", "Route", "Batch code"}
CONFIRM_PARTS = {"Date", "Team", "Ride #"}
# Changes the field team is allowed to make on the day — shown in "What changed", never flagged
ALLOWED_PARTS = {"Mkt ID"}


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
    out["misfiled"] = out["likely_code"].notna() & (out["likely_code"] != sel)
    out["eff_code"] = out["likely_code"].where(out["likely_code"].notna(), sel)
    return out


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
        if "Wrong code" in parts:
            tag_parts["Wrong code selected"] = {"Wrong code"}
        if parts & DESIGN_PARTS:
            tag_parts["Design problem"] = parts & DESIGN_PARTS
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
            return "open"
        tag_state = {t_: _state(ps) for t_, ps in tag_parts.items()}
        open_tags = [t_ for t_ in tags if t_ != "Done as planned" and tag_state[t_] == "open"]
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
                                             f"actual code {_txt(f['actual_batch_code'])} — put 1 in Correct "
                                             f"value to drop this form in cleaning",
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
        open_what = "; ".join(dict.fromkeys(c[1] for ch in changes for c in ch
                                            if c[0] != "allowed" and c[2] not in resolved_parts))

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
            "Last submitted": last.strftime("%d %b %H:%M") if pd.notna(last) else "",
            "Review notes":   "; ".join(review_txt) or "—",
            "Needs decision on": ", ".join(undecided) or "—",
            "_needs_decision": bool(undecided),
            "_issues":        issues,
            "_fix_needed":    fix_needed,
            "_tags":          tags,
            "_open_tags":     open_tags,
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
    css["Status"] = [STATUS_STYLE.get(o[0], "") if o else STATUS_STYLE.get(a[0], GREEN)
                     if a[0] in ("Not submitted", "Done as planned") else GREEN
                     for o, a in zip(t["_open_tags"], t["_tags"])]
    # Actual components: red = wrong ride for the design (treatment, route); blue = confirm;
    # green = the difference was marked OK or corrected
    for col, style in [("Treatment", RED), ("Route", RED),
                       ("Date", BLUE), ("Team", BLUE), ("Ride #", BLUE)]:
        diff = t["_diff"].map(lambda dd: dd[col])
        ok = t["_resolved_parts"].map(lambda rp: col in rp)
        css.loc[diff & ~ok, col] = style
        css.loc[diff & ok, col] = GREEN
    css.loc[t["Code selection"] != "—", "Code selection"] = GRAY
    css.loc[t["_code_flag"], "Code selection"] = RED
    css.loc[t["E1 form"].str.startswith("✗") & ~not_sub, "E1 form"] = RED
    css.loc[t["E1 form"].str.contains("×", regex=False), "E1 form"] = RED
    css.loc[t["_tags"].map(lambda x: "Design problem" in x or "To confirm" in x), "What changed"] = GREEN
    css.loc[t["_open_tags"].map(lambda x: "To confirm" in x), "What changed"] = BLUE
    css.loc[t["_open_tags"].map(lambda x: "Design problem" in x), "What changed"] = RED
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

def today_board(ledger: pd.DataFrame, day) -> dict:
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
                if "Wrong code selected" in tags:
                    acts += [n for n in r["Code selection"].split("; ") if "filed under" in n][:1]
                if "Design problem" in tags:
                    acts.append(f"Design: {r['_open_what']}")
                if "Duplicate E1" in tags:
                    acts.append("Two E1 forms — check which one to keep")
                if "To confirm" in tags:
                    acts.append(f"Check: {r['_open_what']}")
                if "Supervisor only" in tags:
                    acts.append("Chase the E1 form")
                if acts:
                    state = "Action"
                else:
                    state = "Reviewed" if had_issues else "OK"
            rows.append({
                "Ride":       r["_ride"],
                "Planned":    f"{p['route'] or ''} · {TREAT.get(p['treatment'], p['treatment'] or '')}",
                "E1":         r["E1 form"].replace("✗ none", "✗"),
                "Supervisor": r["Supervisor form"].replace("✗ none", "✗"),
                "State":      state,
                "Action":     " · ".join(acts) or "—",
                "Review":     r["Review status"],
            })
            styles.append(state)
        t = pd.DataFrame(rows)
        css = _empty_css(t)
        state_css = {"OK": GREEN, "Reviewed": GREEN, "Action": RED, "Overdue": RED, "Waiting": GRAY}
        for i, st_ in enumerate(styles):
            css.at[i, "State"] = state_css.get(st_, "")
            if st_ in ("Action", "Overdue"):
                css.at[i, "Action"] = RED
            if st_ == "Waiting":
                css.loc[i] = GRAY
        css.loc[t["E1"].str.startswith("✗") & (t["State"] != "Waiting"), "E1"] = RED
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
        status = "All explained"
    elif n_expl:
        status = "Partly explained"
    else:
        status = "Unexplained"
    return {"status": status, "items": items, "n_flag": n_flag, "n_expl": n_expl,
            "recording": recording}


def fidelity_table(subs: pd.DataFrame):
    """One row per E1 form: values next to reasons. Returns (table, css)."""
    d = subs[subs["role"] == E1].sort_values(["ride_date", "batch_code", "submitted_at"])
    rows, flags = [], []
    for i, r in d.iterrows():
        a = assess_issues(r)
        sig, ann = _num(r.get("signup_duration")), _num(r.get("announcement_duration"))
        rows.append({
            "Planned code":        _txt(r["batch_code"]),
            "Ride date":           fmt_day(r["ride_date"]),
            "Treatment":           TREAT.get(_txt(r.get("p_treatment")), _txt(r.get("p_treatment"))),
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
        })
    t = pd.DataFrame(rows)
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
        {"Unexplained": RED, "Partly explained": RED, "All explained": BLUE, "No issues": GREEN}).fillna("")
    css.loc[t["Recording problems"] != "—", "Recording problems"] = RED
    css.loc[t["Completion"] == "Not fully completed", "Completion"] = RED
    return t, css


def crosscheck_table(subs: pd.DataFrame):
    """One row per E1 form with the paired supervisor's independent record. Returns (table, css)."""
    pairs = pair_supervisors(subs)
    d = subs[subs["role"] == E1].sort_values(["ride_date", "batch_code", "submitted_at"])
    rows = []
    for i, r in d.iterrows():
        row = {
            "Planned code": _txt(r["batch_code"]),
            "Ride date":    fmt_day(r["ride_date"]),
            "Enumerator 1": _txt(r["person"]),
        }
        if i not in pairs:
            row.update({"Supervisor": "— not observed —"})
            rows.append(row)
            continue
        j, method = pairs[i]
        s = subs.loc[j]
        gaps = time_gaps(r, s)
        row["Supervisor"] = _txt(s["person"])
        row["Paired by"]  = method
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

    cols = (["Planned code", "Ride date", "Enumerator 1", "Supervisor", "Paired by"]
            + [f"Gap: {l}" for l, _ in CHECKPOINTS]
            + ["Max gap (min)", "Largest at", "Sign-ups E1 / Sup", "Sup: times match", "Mismatched times",
               "Discrepancy source", "E1 recording issue", "Count match", "Ticket match", "Completion match"])
    t = pd.DataFrame(rows).reindex(columns=cols).fillna("—")
    css = _empty_css(t)
    if t.empty:
        return t, css
    for c in [c for c in cols if c.startswith("Gap:")] + ["Max gap (min)"]:
        css.loc[t[c].map(_num) > TIME_GAP_MAX, c] = RED
    css.loc[t["Sup: times match"] == "No", "Sup: times match"] = RED
    css.loc[t["E1 recording issue"] == "E1 recording error", "E1 recording issue"] = RED
    for c in ["Count match", "Ticket match", "Completion match"]:
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
    ("Flags explained",       "explained"),
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
        timing_flags = [r.get(f) for f, _, _ in ISSUE_DEFS if f != "flag_7_low_signup"]
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
