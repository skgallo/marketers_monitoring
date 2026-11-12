"""
rollup.py — Recruitment HFC data rollup
========================================
Transforms raw SurveyCTO submissions + batch_codes frame into one row per
planned batch code, with all cleaning, batch code checks, and timing flags.

Usage:
    from rollup import rollup
    result = rollup(submissions_df, batch_codes_df)

Or from the command line (for testing):
    python rollup.py submissions.csv batch_codes.xlsx
"""

import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

ROUTE_CODES = [
    'LPRR', 'RRLP',   # Lumley – Regent Road
    'LPBB', 'BBLP',   # Lumley Park – Bawbaw
    'LPJU', 'JULP',   # Lumley Park – Jui
    'CTEP', 'EPCT',   # Congo Town – Eastern Police
    'MTEP', 'EPMT',   # Murray Town – Eastern Police
    'MTEE', 'EEMT',   # Murray Town – East End Police
    'BLEE', 'EEBL',   # Brima Lane – East End Police
    'RRAB', 'ABRR',   # Regent Road – Aberdeen
    'SSWE', 'WESS',   # Sacksville Street – Wellington
]

# Non-directional corridors
CORRIDORS = {
    'LPRR': 'LP-RR', 'RRLP': 'LP-RR',
    'LPBB': 'LP-BB', 'BBLP': 'LP-BB',
    'LPJU': 'LP-JU', 'JULP': 'LP-JU',
    'CTEP': 'CT-EP', 'EPCT': 'CT-EP',
    'MTEP': 'MT-EP', 'EPMT': 'MT-EP',
    'MTEE': 'MT-EE', 'EEMT': 'MT-EE',
    'BLEE': 'BL-EE', 'EEBL': 'BL-EE',
    'RRAB': 'RR-AB', 'ABRR': 'RR-AB',
    'SSWE': 'SS-WE', 'WESS': 'SS-WE',
}

# Timing thresholds
DELAY_MAX_MIN        = 2    # Flag 1: max announcement start delay (minutes)
ANN_DUR_MIN          = 8    # Flag 2: min announcement duration
ANN_DUR_MAX          = 12   # Flag 2: max announcement duration
SIGNUP_DIFF_MAX      = 2    # Flag 3: max |signup - announcement| discrepancy (matches SurveyCTO)
PITCH_TARGET_MIN     = 25   # (old rule, kept for reference) target pitch/pause duration
PITCH_TOL_MIN        = 2    # (old rule) tolerance either side
# Treatment-specific pitch/pause rules (Nov 2026)
NORMAL_PITCH_MIN     = 20   # Normal: marketer pitch at least 20 min ...
PEDDLING_PITCH_MIN   = 12   # Peddling (pitch ~15 min): at least 12 min ...
#                             ... unless the pitch ran until the end of the ride
PAUSE_END_TOL_MIN    = 2    # Sit down: pause must last until the ride ends (ends ≤ 2 min before ride end)
RIDE_MIN_MIN         = 35   # Flag 6: minimum expected ride duration (10 ann/signup + 25 pause/pitch)
SIGNUP_MIN           = 14   # Flag 7: minimum expected sign-ups

TEST_USERNAMES = {'testing', 'test', 'skgallo@uchicago.edu'}   # drop these submissions (lowercase)

# Supervisor names — choice list "supervisor" in each SurveyCTO form.
# The codes changed between forms (Daniella was added as 1 in Nov), so the
# right list is picked per submission (-55 = Other → supervisor_oth is used).
SUPERVISOR_LABELS_BY_FORM = {
    'Oct 2026': {'1': 'Fatu', '2': 'Joseph', '3': 'Kadie', '4': 'Sahr'},
    'Nov 2026': {'1': 'Daniella', '2': 'Fatu', '3': 'Joseph', '4': 'Kadie', '5': 'Sahr'},
    'Launch 0': {'1': 'Daniella', '2': 'Fatu', '3': 'Joseph', '4': 'Kadie', '5': 'Sahr'},
}
SUPERVISOR_LABELS = SUPERVISOR_LABELS_BY_FORM['Oct 2026']   # kept for older code
NOV_FORM_FROM_VERSION = 2611000000     # formdef_version of the first Nov form
LAUNCH_FORM_FROM_VERSION = 2611120000  # formdef_version of the first Launch 0 form (rec_form_launch0)

# Interruption list (c10) — the codes changed in the Launch 0 form
C10_LABELS_BY_FORM = {
    'old': {'0': 'None', '1': 'Bus/route interruption', '2': 'Passenger disruption/confrontation',
            '3': 'Marketer/pitch interruption', '4': 'Team, equipment or materials',
            '5': 'Recruitment paused/stopped', '-55': 'Other'},
    'Launch 0': {'0': 'None', '1': 'Traffic or route diversion', '2': 'Bus breakdown', '3': 'Police stop',
                 '4': 'Pharmacy Board intervention', '5': 'Marketer emergency', '6': 'Preacher or other speaker',
                 '7': 'Passenger confrontation/repeated interruptions', '8': 'Passenger emergency',
                 '9': 'Enumerator ill or emergency', '-55': 'Other'},
}
# Interruptions outside the enumerator's control (they explain a flagged value)
C10_EXTERNAL_BY_FORM = {'old': {'1', '2', '3', '5'},
                        'Launch 0': {'1', '2', '3', '4', '5', '6', '7', '8', '9'}}
RIDE_STOP_LABELS = {'0': 'Carried on to the end', '1': 'Stopped before any sign-up card was written on',
                    '2': 'Stopped after sign-up cards were written on'}
PITCH_SHORT_LABELS = {'1': 'marketer simply marketed for a short time', '2': 'something cut the pitch short'}

# Nov form: why the ride differs from the planned code (mm_reason) — only 1 or 4 can be saved
MM_REASON_LABELS = {'1': 'Plan changed (ride really happened differently)',
                    '2': 'Entered details wrong', '3': 'Selected wrong planned code',
                    '4': 'Not sure'}


# Nov form (7 Nov version): when the announcement started relative to the route's strategic point
SP_WHEN_LABELS = {'1': 'Started before the strategic point', '2': 'Started at the strategic point',
                  '3': 'Started AFTER the strategic point'}


def form_source_of(df: pd.DataFrame) -> pd.Series:
    """'Oct 2026' / 'Nov 2026' per row: from the tab it came from, else formdef_version."""
    idx = df.index
    src = df['form_source'] if 'form_source' in df.columns else pd.Series(None, index=idx, dtype=object)
    ver = pd.to_numeric(df.get('formdef_version', pd.Series(None, index=idx)), errors='coerce')
    by_ver = pd.Series(np.where(ver >= LAUNCH_FORM_FROM_VERSION, 'Launch 0',
                                np.where(ver >= NOV_FORM_FROM_VERSION, 'Nov 2026', 'Oct 2026')), index=idx)
    return src.where(src.notna(), by_ver)


def supervisor_name(code, form_source='Oct 2026'):
    labels = SUPERVISOR_LABELS_BY_FORM.get(form_source, SUPERVISOR_LABELS)
    return labels.get(code, code)

# ---------------------------------------------------------------------------
# Batch code helpers
# ---------------------------------------------------------------------------

def _find_route(code: str):
    """Return (route_code, position) or (None, None)."""
    for rc in ROUTE_CODES:
        pos = code.find(rc)
        if pos > 0:
            return rc, pos
    return None, None


def fix_batch_code(code, submission_year: int = 2026) -> str:
    """
    Fix batch codes where the year was omitted from the date portion.

    A valid code is at least 18 chars: DDMMYYYY(8) + team(1) + mkt_id(3+) + route(4) + ride(1) + treatment(1).
    If the code is shorter (e.g. 14 chars = DDMM + team + mkt_id + route + ride + treatment),
    the year is missing — insert it after the first 4 chars (DDMM).

    Examples:
      24081998LPRR1N  (14 chars, team=1 mkt=998) → 240820261998LPRR1N
      26081133RRLP1S  (14 chars, team=1 mkt=133) → 260820261133RRLP1S
      240820261133LPRR1N (18 chars, correct)      → unchanged
    """
    if pd.isna(code):
        return code
    code = str(code).strip()
    if len(code) < 16:
        # Year is missing — insert after DDMM (first 4 chars)
        code = code[:4] + str(submission_year) + code[4:]
    return code


_CODE_RE = re.compile(r"^(\d{8})(\d)(\d+)([A-Z]{4})(\d+)([A-Z])$")


def _corridor(route: str) -> str:
    """Non-directional corridor for any 4-letter route: CTEP and EPCT → 'CT-EP'."""
    a, b = route[:2], route[2:]
    return "-".join(sorted([a, b]))


def parse_batch_code(code) -> dict:
    """
    Parse a (cleaned) batch code into its components.
    Returns a dict with keys: date, team, mkt_id, route, corridor, ride_num, treatment, parse_ok
    """
    empty = dict(date=None, team=None, mkt_id=None, route=None,
                 corridor=None, ride_num=None, treatment=None, parse_ok=False)
    if pd.isna(code):
        return empty
    code = str(code).strip()

    # Structure: DDMMYYYY + team (1 digit) + marketer ID (digits) + route (4 letters)
    #            + ride # (digits) + treatment (1 letter). Any 4-letter route is accepted,
    #            so new routes don't need to be added to ROUTE_CODES first.
    m = _CODE_RE.match(code.upper())
    if m:
        date, team, mkt, route, ride, treat = m.groups()
        return dict(date=date, team=team, mkt_id=mkt, route=route,
                    corridor=CORRIDORS.get(route) or _corridor(route),
                    ride_num=ride, treatment=treat, parse_ok=True)

    # Fallback for irregular codes: locate a known route code
    route, route_pos = _find_route(code)
    if route is None or route_pos < 9:   # need at least 8-char date + 1-char team
        return empty
    after_route = code[route_pos + 4:]
    if len(after_route) < 2:
        return empty
    return dict(
        date      = code[:8],
        team      = code[8],
        mkt_id    = code[9:route_pos],
        route     = route,
        corridor  = CORRIDORS.get(route),
        ride_num  = after_route[0],
        treatment = after_route[-1],
        parse_ok  = True,
    )

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw SurveyCTO submissions:
      - Drop test submissions
      - Fix batch code typos
      - Rename: batch_code_pre → batch_code (planned); batch_code → actual_batch_code
    """
    df = df.copy()
    # Same column name twice (e.g. stacked tabs, or a field published twice): keep one, first non-empty value
    if df.columns.duplicated().any():
        merged = {}
        for c in dict.fromkeys(df.columns):
            block = df.loc[:, df.columns == c]
            merged[c] = block.iloc[:, 0] if block.shape[1] == 1 else block.astype(object).bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(merged, index=df.index)
    # The rename below must not create a duplicate 'actual_batch_code' / 'batch_code'
    df = df.drop(columns=[c for c in ('actual_batch_code',) if c in df.columns and 'batch_code' in df.columns])

    # Each SurveyCTO form once: if publishing re-sent forms, keep the latest copy of each KEY
    if 'KEY' in df.columns:
        df = df[~(df['KEY'].notna() & df.duplicated(subset='KEY', keep='last'))].copy()

    # Drop test rows
    if 'username' in df.columns:
        df = df[~df['username'].str.lower().isin(TEST_USERNAMES)].copy()

    # Which form version (supervisor codes differ between them)
    df['form_source'] = form_source_of(df)
    # The Nov form asks the E1 "What is your name?" as e1_name — keep it as e1_self
    # (e1_name is the dashboard's name for Enumerator 1, from a02_1)
    if 'e1_name' in df.columns:
        df = df.rename(columns={'e1_name': 'e1_self', 'e1_name_oth': 'e1_self_oth'})

    # Fields renamed in the 7 Nov form version: fill the names the dashboard uses
    def _first(*cols):
        out = pd.Series(None, index=df.index, dtype=object)
        for c in cols:
            if c in df.columns:
                out = out.where(out.notna() & (out.astype(str).str.strip() != ''), df[c])
        return out
    df['sup_feedback_additional_strengths'] = _first('sup_feedback_additional_strengths', 'sup_fb_add_strengths')
    df['sup_feedback_additional_improvements'] = _first('sup_feedback_additional_improvements', 'sup_fb_add_improvements')
    # Start-delay reason: the free-text question was replaced by the bus-fullness / strategic-point questions
    gap_reason = df.get('gap_notfull_reason', pd.Series(None, index=df.index)).map(_code)
    sp = df.get('sp_when', pd.Series(None, index=df.index)).map(_code)
    built = []
    for i in df.index:
        bits = []
        if gap_reason.get(i) == '1':
            bits.append('Waited: bus less than half full')
        elif gap_reason.get(i) == '2':
            o = df.at[i, 'gap_notfull_other'] if 'gap_notfull_other' in df.columns else None
            bits.append('Other: ' + str(o).strip() if pd.notna(o) and str(o).strip() else 'Other reason')
        w = df.at[i, 'gap_fullbus_why'] if 'gap_fullbus_why' in df.columns else None
        if pd.notna(w) and str(w).strip():
            bits.append('Bus at least half full: ' + str(w).strip())
        if sp.get(i) in SP_WHEN_LABELS:
            bits.append(SP_WHEN_LABELS[sp.get(i)])
        built.append('; '.join(bits) or None)
    # Launch 0 form: interruption details are a repeat group, published as dis_*_1, dis_*_2, …
    if 'c10a' not in df.columns or not df['c10a'].notna().any():
        rep = sorted([c for c in df.columns if re.match(r'^dis_what(_\d+)?$', c)],
                     key=lambda c: int(c.rsplit('_', 1)[-1]) if c.rsplit('_', 1)[-1].isdigit() else 0)
        if rep:
            df['c10a'] = [' | '.join(str(x).strip() for x in vals if pd.notna(x) and str(x).strip()) or None
                          for vals in df[rep].values.tolist()]
    # Interruption labels and "outside the E1's control", by form version
    def _c10(row_codes, src):
        key = 'Launch 0' if src == 'Launch 0' else 'old'
        codes = [c for c in str(row_codes).split() if c and str(row_codes) not in ('nan', 'None')]
        return ("; ".join(C10_LABELS_BY_FORM[key].get(c, c) for c in codes) or None,
                bool(set(codes) & C10_EXTERNAL_BY_FORM[key]))
    c10 = df.get('c10', pd.Series(None, index=df.index))
    pairs = [_c10(c, s) if pd.notna(c) else (None, False) for c, s in zip(c10, df['form_source'])]
    df['c10_labels'] = [p[0] for p in pairs]
    df['c10_external'] = [p[1] for p in pairs]
    # Launch 0 form: why a pitch was short (replaces part of the free-text explanation)
    if 'pause_short_reason' in df.columns:
        why = df['pause_short_reason'].map(_code).map(PITCH_SHORT_LABELS)
        base = _first('pause_duration_explanation')
        df['pause_duration_explanation'] = [
            '; '.join(x for x in [w if isinstance(w, str) else None, (str(b).strip() if pd.notna(b) else None)] if x) or None
            for w, b in zip(why, base)]
    df['announcement_delay_explanation'] = _first('announcement_delay_explanation').where(
        _first('announcement_delay_explanation').notna(), pd.Series(built, index=df.index))

    # Tag every submission with its role — all submissions are kept.
    # "Enumerator 1" = role_type 1 (or missing/unknown — older forms without the field).
    # "Supervisor"   = role_type 0.
    # The display layer uses this column to separate E1 and supervisor rows.
    if 'role_type' in df.columns:
        rt = df['role_type'].astype(str).str.strip()
        df['role'] = rt.map({
            '1': 'Enumerator 1', '1.0': 'Enumerator 1',
            '0': 'Supervisor',   '0.0': 'Supervisor',
        }).fillna('Enumerator 1')
    else:
        df['role'] = 'Enumerator 1'

    # Extract submission year for typo fixes
    if 'SubmissionDate' in df.columns:
        years = pd.to_datetime(df['SubmissionDate'], format='mixed', errors='coerce').dt.year.fillna(2026).astype(int)
    else:
        years = pd.Series([2026] * len(df), index=df.index)

    df['batch_code']     = [fix_batch_code(c, y) for c, y in zip(df['batch_code'],     years)]
    df['batch_code_pre'] = [fix_batch_code(c, y) for c, y in zip(df['batch_code_pre'], years)]

    # Rename columns to make the join logic explicit
    df = df.rename(columns={
        'batch_code':     'actual_batch_code',   # what was actually submitted
        'batch_code_pre': 'batch_code',          # what was planned (= frame key)
    })

    return df

# ---------------------------------------------------------------------------
# Batch code comparison flags
# ---------------------------------------------------------------------------

def _parsed_frame(codes: pd.Series, prefix: str) -> pd.DataFrame:
    """parse_batch_code() for every code, as columns with a prefix. Works for 0 rows too (when every form
    is a test form, Series.apply(pd.Series) would return a Series and duplicate the code column)."""
    keys = list(parse_batch_code(None).keys())
    rows = [parse_batch_code(c) for c in codes]
    return pd.DataFrame(rows, index=codes.index, columns=keys).add_prefix(prefix)


def add_batch_code_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse planned and actual batch codes
    p = _parsed_frame(df['batch_code'], 'p_')
    a = _parsed_frame(df['actual_batch_code'], 'a_')
    df = pd.concat([df, p, a], axis=1)

    has_actual = df['actual_batch_code'].notna()
    codes_differ = has_actual & (df['actual_batch_code'] != df['batch_code'])

    def flag(condition):
        """Return 1/0 only where both codes parsed OK; else NA."""
        both_ok = df['p_parse_ok'] & df['a_parse_ok']
        return np.where(both_ok & codes_differ, condition.astype(int), np.where(has_actual, 0, pd.NA))

    df['flag_batch_code_different'] = np.where(has_actual, codes_differ.astype(int), pd.NA)

    df['flag_batch_treatment'] = flag(df['p_treatment'] != df['a_treatment'])
    df['flag_batch_corridor']  = flag(df['p_corridor']  != df['a_corridor'])
    df['flag_batch_direction']  = flag(
        (df['p_corridor'] == df['a_corridor']) & (df['p_route'] != df['a_route'])
    )
    df['flag_batch_date']      = flag(df['p_date']      != df['a_date'])
    df['flag_batch_marketer']  = flag(df['p_mkt_id']    != df['a_mkt_id'])
    df['flag_batch_team']      = flag(df['p_team']      != df['a_team'])

    df['flag_design_problem']     = flag(
        (df['p_treatment'] != df['a_treatment']) |
        (df['p_corridor']  != df['a_corridor'])  |
        ((df['p_corridor'] == df['a_corridor']) & (df['p_route'] != df['a_route']))
    )
    df['flag_needs_confirmation'] = flag(
        (df['p_date']   != df['a_date']) |
        (df['p_mkt_id'] != df['a_mkt_id'])
    )

    return df

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_time(val) -> pd.Timestamp:
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    for fmt in ('%I:%M:%S %p', '%H:%M:%S', '%I:%M %p'):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def _diff_minutes(t1, t2) -> float:
    """t2 − t1 in minutes; handles midnight crossover."""
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    diff = (t2 - t1).total_seconds() / 60.0
    if diff < -720:   # crossed midnight
        diff += 1440
    return diff

# ---------------------------------------------------------------------------
# Timing flags
# ---------------------------------------------------------------------------

def pitch_pause_check(treatment, b05, b06, b09, b10, b11):
    """
    (flag 1/0/None, text) for the pitch / pause of one ride, by treatment.
      S (sit down): pause should run until the ride ends.
      N (normal):   pitch ≥ NORMAL_PITCH_MIN, unless it ran until the ride ended.
      P (peddling): pitch ≥ PEDDLING_PITCH_MIN, unless it ran until the ride ended.
    """
    t = str(treatment) if treatment is not None and not (isinstance(treatment, float) and np.isnan(treatment)) else ''
    start, end = (b06, b10) if t == 'S' else (b05, b09)
    dur = _diff_minutes(start, end)
    to_end = _diff_minutes(end, b11)          # minutes between pitch/pause end and ride end
    dur = round(dur, 1) if not np.isnan(dur) else dur          # judged on the value shown (1 decimal)
    to_end = round(to_end, 1) if not np.isnan(to_end) else to_end
    if t == 'S':
        if np.isnan(to_end):
            return None, ''
        if to_end > PAUSE_END_TOL_MIN:
            return 1, f"Pause ended {to_end:.1f} min before the ride ended"
        return 0, ''
    if t in ('N', 'P'):
        if np.isnan(dur):
            return None, ''
        floor = NORMAL_PITCH_MIN if t == 'N' else PEDDLING_PITCH_MIN
        ran_to_end = not np.isnan(to_end) and to_end <= PAUSE_END_TOL_MIN
        if dur < floor and not ran_to_end:
            return 1, f"Pitch {dur:.1f} min (< {floor})" + (
                f", ended {to_end:.1f} min before the ride" if not np.isnan(to_end) else "")
        return 0, ''
    return None, ''


def add_timing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Parse raw SurveyCTO time columns ────────────────────────────────────
    for col in ['b02_r', 'b03', 'b05', 'b06', 'b09', 'b10', 'b11']:
        df[f'_{col}'] = df[col].apply(_parse_time) if col in df.columns else pd.NaT

    # Derive raw durations from timestamps (for chronological checks)
    df['_delay_min']     = df.apply(lambda r: _diff_minutes(r['_b02_r'], r['_b03']), axis=1)
    df['_pitch_dur_min'] = df.apply(lambda r: _diff_minutes(r['_b05'],   r['_b09']), axis=1)  # N
    df['_pause_dur_min'] = df.apply(lambda r: _diff_minutes(r['_b06'],   r['_b10']), axis=1)  # S
    df['_b09_to_b11']    = df.apply(lambda r: _diff_minutes(r['_b09'],   r['_b11']), axis=1)
    df['_b10_to_b11']    = df.apply(lambda r: _diff_minutes(r['_b10'],   r['_b11']), axis=1)

    # ── SurveyCTO computed fields ────────────────────────────────────────────
    # signup_duration      = enumerator-RECORDED announcement/signup activity duration
    # announcement_duration = AUTOMATIC form-screen timing (b03→b05/b06 area)
    # Round to 6 dp for comparison; display layer rounds to 2 dp.
    sig_dur = pd.to_numeric(df.get('signup_duration',       pd.Series(np.nan, index=df.index)), errors='coerce').round(6)
    ann_dur = pd.to_numeric(df.get('announcement_duration', pd.Series(np.nan, index=df.index)), errors='coerce').round(6)

    # Overwrite with rounded versions so display layer gets consistent values
    df['signup_duration']       = sig_dur
    df['announcement_duration'] = ann_dur

    # ── Treatment and submission presence ────────────────────────────────────
    # What actually happened: the script entered on the form (a06_r: 0 sit down, 1 normal,
    # 2 peddling), else the actual code's treatment letter, else the planned one
    idx = df.index
    form_t = df.get('a06_r', pd.Series(None, index=idx)).map(_code).map({'0': 'S', '1': 'N', '2': 'P'})
    treatment = (form_t.fillna(df.get('a_treatment', pd.Series(None, index=idx)))
                       .fillna(df.get('p_treatment', pd.Series(None, index=idx))))
    df['treatment_actual'] = treatment
    is_N    = treatment.isin(['N', 'P'])   # Normal and Peddling both have a marketer pitch (b05→b09)
    is_S    = treatment == 'S'
    has_sub = df['actual_batch_code'].notna()

    # ── NA-safe flag helper ──────────────────────────────────────────────────
    def flag_where(condition, *required):
        """
        Return 1/0/NA.
        NA when: no submission OR any required value is missing.
        Missing values NEVER receive a green pass (0).
        """
        valid = has_sub.copy()
        for s in required:
            valid = valid & pd.Series(s, index=df.index).notna()
        return np.where(valid, condition.fillna(False).astype(int), pd.NA)

    # ── Flag 1: announcement start delay > 2 min ─────────────────────────────
    delay = pd.to_numeric(df.get('announcement_start_delay', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['flag_1_start_delay'] = flag_where(delay > DELAY_MAX_MIN, delay)

    # ── Flag 2: RECORDED ann./signup duration outside 8–12 min ───────────────
    # Uses signup_duration (enumerator-recorded).
    # announcement_duration (auto form timing) is NOT independently flagged here.
    df['flag_2_ann_duration'] = flag_where(
        (sig_dur < ANN_DUR_MIN) | (sig_dur > ANN_DUR_MAX), sig_dur
    )

    # ── Flag 3: discrepancy between recorded and auto timing ──────────────────
    # Both rounded to 6 dp before comparison.
    # Label: "Recorded activity timing differs from automatic form timing—review required."
    # Does NOT imply either measure is wrong.
    discrep = (sig_dur - ann_dur).abs()
    df['flag_3_signup_discrep'] = flag_where(discrep > SIGNUP_DIFF_MAX, sig_dur, ann_dur)

    # ── Flags 4a/4b: treatment-specific pitch / pause rules ──────────────────
    # Sit down (4a): the pause should last from the end of sign-up until the ride ends,
    #   so it is flagged when it ends more than PAUSE_END_TOL_MIN before the ride end.
    # Normal / Peddling (4b): the pitch should last at least NORMAL_PITCH_MIN /
    #   PEDDLING_PITCH_MIN minutes, unless it ran until the end of the ride.
    flags_ab = [pitch_pause_check(t, b05, b06, b09, b10, b11)
                for t, b05, b06, b09, b10, b11 in zip(treatment, df['_b05'], df['_b06'], df['_b09'],
                                                      df['_b10'], df['_b11'])]
    f = pd.Series([x[0] for x in flags_ab], index=df.index)
    df['pitch_pause_check'] = [x[1] for x in flags_ab]
    df['flag_4a_pause_dur'] = np.where(has_sub & is_S & f.notna(), f.fillna(0).astype(int), pd.NA)
    df['flag_4b_pitch_dur'] = np.where(has_sub & is_N & f.notna(), f.fillna(0).astype(int), pd.NA)
    pitch_raw = pd.Series(df['_pitch_dur_min'], index=df.index)   # used by the chronological checks below

    # ── Flag 4c: pitch/pause ends AFTER ride end (chronological impossible) ───
    b09_to_b11 = pd.Series(df['_b09_to_b11'], index=df.index)
    b10_to_b11 = pd.Series(df['_b10_to_b11'], index=df.index)
    df['flag_4c_after_ride'] = np.where(
        has_sub & is_S & b10_to_b11.notna(), (b10_to_b11 < 0).astype(int),
        np.where(
            has_sub & is_N & b09_to_b11.notna(), (b09_to_b11 < 0).astype(int),
            pd.NA
        )
    )

    # ── Chronological impossibility flags ────────────────────────────────────
    # These catch impossible timestamp sequences regardless of computed field values.
    delay_raw = pd.Series(df['_delay_min'], index=df.index)
    pause_raw = pd.Series(df['_pause_dur_min'], index=df.index)

    # Announcement started before ride (negative delay from raw timestamps)
    df['flag_chron_neg_delay'] = np.where(
        has_sub & delay_raw.notna(), (delay_raw < 0).astype(int), pd.NA
    )
    # Negative auto form-screen duration
    df['flag_chron_neg_ann'] = flag_where(ann_dur < 0, ann_dur)
    # Negative enumerator-recorded duration
    df['flag_chron_neg_signup'] = flag_where(sig_dur < 0, sig_dur)
    # Negative pause/pitch duration from raw timestamps
    df['flag_chron_neg_pause'] = np.where(
        has_sub & is_S & pause_raw.notna(), (pause_raw < 0).astype(int), pd.NA
    )
    df['flag_chron_neg_pitch'] = np.where(
        has_sub & is_N & pitch_raw.notna(), (pitch_raw < 0).astype(int), pd.NA
    )

    # ── Flag 6: short ride ────────────────────────────────────────────────────
    ride_dur = pd.to_numeric(df.get('ride_duration', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['flag_6_short_ride'] = flag_where(ride_dur < RIDE_MIN_MIN, ride_dur)

    # ── Flag 7: low signups ───────────────────────────────────────────────────
    total_signups = pd.to_numeric(df.get('total_signups', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['flag_7_low_signup'] = flag_where(total_signups < SIGNUP_MIN, total_signups)

    # ── Flag 8 (Nov form): marketer pitch started > 2 min after sign-up ended ─
    # pitch_start_gap = b04 (sign-up end) → b05 (pitch start); Normal/Peddling rides only.
    gap = pd.to_numeric(df.get('pitch_start_gap', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['pitch_start_gap'] = gap.round(6)
    df['flag_8_pitch_start_gap'] = np.where(has_sub & gap.notna(), (gap.round(6) > DELAY_MAX_MIN).astype(int), pd.NA)

    # ── Flag 5: any timing/chronological issue ───────────────────────────────
    timing_flags = [
        'flag_1_start_delay', 'flag_2_ann_duration', 'flag_3_signup_discrep',
        'flag_4a_pause_dur', 'flag_4b_pitch_dur', 'flag_4c_after_ride',
        'flag_6_short_ride',
        'flag_chron_neg_delay', 'flag_chron_neg_ann', 'flag_chron_neg_signup',
        'flag_chron_neg_pause', 'flag_chron_neg_pitch',
    ]
    df['flag_5_any_timing'] = np.where(
        has_sub,
        df[timing_flags].apply(lambda row: int(pd.to_numeric(row, errors='coerce').eq(1).any()), axis=1),
        pd.NA
    )

    return df

# ---------------------------------------------------------------------------
# People and dates
# ---------------------------------------------------------------------------

def _code(v):
    """Normalise a select_one value: 1.0 → '1', NaN → None."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    s = str(v).strip()
    if s in ('', 'nan', 'None'):
        return None
    return s[:-2] if s.endswith('.0') else s


def add_people(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the people on each form.

    e1_id / e1_name : Enumerator 1 (a02_1 / a02_1_lab). Recorded on BOTH E1 and
                      supervisor forms, so a supervisor form says which E1 was observed.
                      e1_id is the roster code (stable even if name spelling varies);
                      'Other' (-55) uses the typed name as the id.
    sup_name        : supervisor (supervisor code → SUPERVISOR_LABELS; -55 → supervisor_oth).
    person          : who filled THIS form (E1 name on E1 forms, supervisor on supervisor forms).
    """
    df = df.copy()
    idx = df.index
    username = df.get('username', pd.Series(None, index=idx))

    e1_code = df.get('a02_1', pd.Series(None, index=idx)).map(_code)
    e1_lab  = df.get('a02_1_lab', pd.Series(None, index=idx)).map(
        lambda v: str(v).strip() if pd.notna(v) and str(v).strip() not in ('', 'nan') else None)
    # The roster codes changed between the Oct and Nov forms (e.g. Daniel 3 → 14),
    # so enumerators are identified by name (spaces/case/dots ignored) where available.
    norm = lambda s: re.sub(r'[^a-z]', '', s.lower())
    df['e1_id'] = [
        (f"name:{norm(lab)}" if lab and norm(lab) else c) if (c or lab) else (f"user:{u}" if pd.notna(u) else None)
        for c, lab, u in zip(e1_code, e1_lab, username)
    ]
    # Canonical display name per e1_id = most frequent spelling
    canon = (pd.DataFrame({'id': df['e1_id'], 'lab': e1_lab}).dropna()
             .groupby('id')['lab'].agg(lambda s: s.value_counts().index[0]).to_dict())
    df['e1_name'] = [canon.get(i) or (str(u) if pd.notna(u) else None)
                     for i, u in zip(df['e1_id'], username)]

    sup_code = df.get('supervisor', pd.Series(None, index=idx)).map(_code)
    sup_oth  = df.get('supervisor_oth', pd.Series(None, index=idx))
    src      = df['form_source'] if 'form_source' in df.columns else pd.Series('Oct 2026', index=idx)
    df['sup_name'] = [
        (str(o).strip() if c == '-55' and pd.notna(o) else supervisor_name(c, s)) if c else None
        for c, o, s in zip(sup_code, sup_oth, src)
    ]
    is_sup = df['role'] == 'Supervisor'
    df['person'] = np.where(is_sup, df['sup_name'], df['e1_name'])
    df['person'] = df['person'].where(pd.notna(df['person']), username)

    # Ride date: form date (a01_r) → actual code date → submission date
    form_date = pd.to_datetime(df.get('a01_r', pd.Series(None, index=idx)), format='mixed', errors='coerce')
    code_date = pd.to_datetime(df.get('a_date', pd.Series(None, index=idx)), format='%d%m%Y', errors='coerce')
    sub_date  = pd.to_datetime(df.get('SubmissionDate', pd.Series(None, index=idx)),
                               format='mixed', errors='coerce', utc=True).dt.tz_localize(None)
    ride_date = form_date.fillna(code_date).fillna(sub_date.dt.normalize())
    df['ride_date']  = ride_date.dt.normalize()
    df['week_start'] = (df['ride_date'] - pd.to_timedelta(df['ride_date'].dt.weekday, unit='D'))
    df['submitted_at'] = sub_date
    df['form_id'] = df.get('KEY', pd.Series(None, index=idx)).fillna(pd.Series(idx.astype(str), index=idx))
    return df


def prepare_submissions(submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    All cleaned submissions (E1 and supervisor), one row per form, with batch
    code flags, timing flags, people, ride date and duplicate-E1 flag.
    """
    subs = clean_submissions(submissions_df)
    subs = add_batch_code_flags(subs)
    subs = add_timing_flags(subs)
    subs = add_people(subs)

    # More than one E1 form for the same planned code (mistake, or two E1s on one ride)
    e1_mask = subs['role'] == 'Enumerator 1'
    e1_counts = subs[e1_mask].groupby('batch_code').size().to_dict()
    subs['n_e1_forms'] = subs['batch_code'].map(lambda bc: e1_counts.get(bc, 0))
    subs['flag_duplicate_submission'] = np.where(e1_mask & (subs['n_e1_forms'] > 1), 1, 0)
    return subs.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Main rollup
# ---------------------------------------------------------------------------

def rollup(submissions_df: pd.DataFrame, batch_codes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Core rollup: one row per submission, grouped under planned batch codes.

    A planned batch code may appear multiple times if:
      - Both an Enumerator 1 and a Supervisor submitted (expected).
      - More than one E1 submitted for the same ride (flagged as duplicate).
    Planned codes with no submission appear once with NaN submission fields.

    Parameters
    ----------
    submissions_df : raw SurveyCTO wide-format DataFrame
    batch_codes_df : batch codes frame (one row per scheduled ride)

    Returns
    -------
    DataFrame with one row per submission (or one row per unmatched planned
    code), all flags attached.
    """
    subs = prepare_submissions(submissions_df)

    # Left-join frame onto submissions (frame is the source of truth).
    # Multiple submissions for the same batch_code produce multiple rows.
    result = batch_codes_df.merge(subs, on='batch_code', how='left')

    # Status column
    result['submission_status'] = np.where(
        result['actual_batch_code'].notna(), 'Submitted', 'No submission'
    )

    # Drop internal working columns
    drop_cols = [c for c in result.columns if c.startswith('_')]
    result = result.drop(columns=drop_cols)

    return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    xlsx_path = sys.argv[2] if len(sys.argv) > 2 else 'batch_codes.xlsx'

    if csv_path is None:
        print("Usage: python rollup.py submissions.csv batch_codes.xlsx")
        sys.exit(1)

    subs = pd.read_csv(csv_path)
    bc   = pd.read_excel(xlsx_path, sheet_name='archived')

    result = rollup(subs, bc)

    cols = [
        'batch_code', 'actual_batch_code', 'submission_status',
        'flag_batch_code_different', 'flag_design_problem', 'flag_needs_confirmation',
        'flag_1_start_delay', 'flag_2_ann_duration', 'flag_3_signup_discrep',
        'flag_4a_pause_dur', 'flag_4b_pitch_dur', 'flag_4c_after_ride', 'flag_5_any_timing',
    ]
    print(result[cols].to_string())


# ---------------------------------------------------------------------------
# Script Checklist (Enumerator 4) — independent timing record of the same ride
# ---------------------------------------------------------------------------
# Form "Script Checklist November 2026" (script_checklist_nov2026). Filled by the
# Enumerator 4 (role_type 1) or a supervisor (role_type 0). Only field activities
# (activity 3 = Field Training, 4 = Field Observation) describe a ride.
E4_OBSERVER_LABELS = {'41': 'Aminata Kamara', '42': 'Davida A L Yambasu', '43': 'Mamusu Bangura',
                      '44': 'Fatorma Sheriff', '14': 'Daniel T G Sesay'}
E4_SUPERVISOR_LABELS = {'1': 'Daniella Mansaray', '2': 'Kadiatu Nabay', '3': 'Sahr Fasuluku',
                        '4': 'Joseph Samuels', '5': 'Fatu Conteh'}
E1_LABELS_NOV = {'11': 'Misheal Sellu', '12': 'Cecil Massaquoi', '13': 'Kadija Esther Kamara',
                 '14': 'Daniel T G Sesay', '44': 'Fatorma Sheriff'}


def prepare_e4(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per Script Checklist form that describes a field ride, with the same
    planned/actual code columns as the recruitment forms (batch_code = planned)."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if 'KEY' in df.columns:
        df = df[~(df['KEY'].notna() & df.duplicated(subset='KEY', keep='last'))].copy()
    if 'username' in df.columns:
        df = df[~df['username'].astype(str).str.lower().isin(TEST_USERNAMES)].copy()
    if 'activity' in df.columns:
        act = pd.to_numeric(df['activity'], errors='coerce')
        df = df[act.isna() | (act > 2)].copy()
    if df.empty:
        return df
    years = pd.to_datetime(df.get('SubmissionDate'), format='mixed', errors='coerce').dt.year.fillna(2026).astype(int)
    for c in ['batch_code', 'batch_code_pre']:
        if c not in df.columns:
            df[c] = None
        df[c] = [fix_batch_code(x, y) for x, y in zip(df[c], years)]
    df = df.rename(columns={'batch_code': 'actual_batch_code', 'batch_code_pre': 'batch_code'})
    p = _parsed_frame(df['batch_code'], 'p_')
    a = _parsed_frame(df['actual_batch_code'], 'a_')
    df = pd.concat([df, p, a], axis=1)

    rt = df.get('role_type', pd.Series(None, index=df.index)).map(_code)
    obs = [E4_SUPERVISOR_LABELS.get(_code(s), _code(s)) if r == '0' else E4_OBSERVER_LABELS.get(_code(e), _code(e))
           for r, s, e in zip(rt, df.get('supervisor', pd.Series(None, index=df.index)),
                              df.get('e4_name', pd.Series(None, index=df.index)))]
    for oth_col, mask in [('supervisor_oth', rt == '0'), ('e4_name_oth', rt != '0')]:
        if oth_col in df.columns:
            obs = [str(o).strip() if (m and str(x) == '-55' and pd.notna(o)) else x
                   for x, o, m in zip(obs, df[oth_col], mask)]
    df['observer'] = obs
    df['observer_role'] = np.where(rt == '0', 'Supervisor (checklist)', 'E4')
    lab = df.get('a02_1_lab', pd.Series(None, index=df.index))
    df['e1_name'] = [str(l).strip() if pd.notna(l) and str(l).strip() else E1_LABELS_NOV.get(_code(c), _code(c))
                     for l, c in zip(lab, df.get('a02_1', pd.Series(None, index=df.index)))]
    form_date = pd.to_datetime(df.get('a01_r', pd.Series(None, index=df.index)), format='mixed', errors='coerce')
    code_date = pd.to_datetime(df.get('a_date', pd.Series(None, index=df.index)), format='%d%m%Y', errors='coerce')
    df['ride_date'] = form_date.fillna(code_date).dt.normalize()
    df['submitted_at'] = pd.to_datetime(df.get('SubmissionDate'), format='mixed', errors='coerce',
                                        utc=True).dt.tz_localize(None)
    df['form_id'] = df.get('KEY', pd.Series(None, index=df.index))
    df['person'] = df['observer']
    df['role'] = 'E4'
    return df.reset_index(drop=True)
