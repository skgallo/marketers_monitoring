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
PITCH_TARGET_MIN     = 25   # Flag 4a/4b: target pitch/pause duration
PITCH_TOL_MIN        = 2    # Flag 4a/4b: tolerance either side (±2 → 23–27 min)
RIDE_MIN_MIN         = 35   # Flag 6: minimum expected ride duration (10 ann/signup + 25 pause/pitch)

TEST_USERNAMES = {'testing', 'test'}   # drop these submissions

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

    # Drop test / enumerator rows
    if 'username' in df.columns:
        df = df[~df['username'].str.lower().isin(TEST_USERNAMES)].copy()

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

def add_batch_code_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse planned and actual batch codes
    p = df['batch_code'].apply(parse_batch_code).apply(pd.Series).add_prefix('p_')
    a = df['actual_batch_code'].apply(parse_batch_code).apply(pd.Series).add_prefix('a_')
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

def add_timing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse raw time columns
    for col in ['b02_r', 'b03', 'b05', 'b06', 'b09', 'b10', 'b11']:
        if col in df.columns:
            df[f'_{col}'] = df[col].apply(_parse_time)
        else:
            df[f'_{col}'] = pd.NaT

    # Derive durations (recompute from raw regardless of SurveyCTO's calculated fields)
    df['_delay_min']     = df.apply(lambda r: _diff_minutes(r['_b02_r'], r['_b03']),  axis=1)
    df['_pitch_dur_min'] = df.apply(lambda r: _diff_minutes(r['_b05'],   r['_b09']),  axis=1)  # N
    df['_pause_dur_min'] = df.apply(lambda r: _diff_minutes(r['_b06'],   r['_b10']),  axis=1)  # S
    df['_b09_to_b11']    = df.apply(lambda r: _diff_minutes(r['_b09'],   r['_b11']),  axis=1)
    df['_b10_to_b11']    = df.apply(lambda r: _diff_minutes(r['_b10'],   r['_b11']),  axis=1)

    # Use SurveyCTO's computed durations where the raw calculation fails
    for src, dst in [('announcement_duration', '_ann_dur'), ('signup_duration', '_signup_dur')]:
        df[dst] = pd.to_numeric(df.get(src, pd.Series(dtype=float)), errors='coerce')

    # Determine pitch type from planned batch code treatment (fall back to actual)
    treatment = df.get('p_treatment', pd.Series(dtype=str)).fillna(
                df.get('a_treatment', pd.Series(dtype=str)))

    is_N = treatment == 'N'
    is_S = treatment == 'S'
    has_sub = df['actual_batch_code'].notna()

    def mk_flag(cond_series):
        """0/1 where submission exists, NA where no submission."""
        return np.where(has_sub, cond_series.fillna(0).astype(int), pd.NA)

    # Flag 1: announcement start delay > 2 min
    df['flag_1_start_delay'] = mk_flag(df['_delay_min'] > DELAY_MAX_MIN)

    # Flag 2: announcement duration outside 8–12 min
    df['flag_2_ann_duration'] = mk_flag(
        (df['_ann_dur'] < ANN_DUR_MIN) | (df['_ann_dur'] > ANN_DUR_MAX)
    )

    # Flag 3: |signup_duration − announcement_duration| > 1 min
    df['flag_3_signup_discrep'] = mk_flag(
        abs(df['_signup_dur'] - df['_ann_dur']) > SIGNUP_DIFF_MAX
    )

    # Flag 4a: S-ride pause duration outside 23–27 min (no short-ride exception)
    # Use SurveyCTO's pause_duration where available (computed from pause_start field,
    # not raw b06 which can differ slightly). Fall back to b06→b10 for older submissions.
    s_pause_dur = pd.to_numeric(df.get('pause_duration', pd.Series(dtype=float)), errors='coerce')
    s_pause_dur = s_pause_dur.where(s_pause_dur.notna(), df['_pause_dur_min'])
    df['flag_4a_pause_dur'] = np.where(
        has_sub & is_S,
        (abs(s_pause_dur - PITCH_TARGET_MIN) > PITCH_TOL_MIN).astype(int),
        pd.NA
    )

    # Flag 4b: N-ride pitch duration outside 23–27 min (no short-ride exception)
    # No SurveyCTO field for pitch_duration — always computed from b05→b09
    df['flag_4b_pitch_dur'] = np.where(
        has_sub & is_N,
        (abs(df['_pitch_dur_min'] - PITCH_TARGET_MIN) > PITCH_TOL_MIN).astype(int),
        pd.NA
    )

    # Flag 4c: pitch/pause ends after ride end
    df['flag_4c_after_ride'] = np.where(
        has_sub & is_S, (df['_b10_to_b11'] < 0).astype(int),
        np.where(
            has_sub & is_N, (df['_b09_to_b11'] < 0).astype(int),
            pd.NA
        )
    )

    # Flag 6: short ride — ride_duration < 35 min (10 ann/signup + 25 pause/pitch minimum)
    ride_dur = pd.to_numeric(df.get('ride_duration', pd.Series(dtype=float)), errors='coerce')
    df['flag_6_short_ride'] = mk_flag(ride_dur < RIDE_MIN_MIN)

    # Flag 7: low signups — total_signups < 14
    total_signups = pd.to_numeric(df.get('total_signups', pd.Series(dtype=float)), errors='coerce')
    df['flag_7_low_signup'] = mk_flag(total_signups < 14)

    # Flag 5: any timing issue (includes short ride)
    timing = ['flag_1_start_delay', 'flag_2_ann_duration', 'flag_3_signup_discrep',
              'flag_4a_pause_dur', 'flag_4b_pitch_dur', 'flag_4c_after_ride',
              'flag_6_short_ride']
    df['flag_5_any_timing'] = np.where(
        has_sub,
        df[timing].apply(lambda row: int(row.fillna(0).eq(1).any()), axis=1),
        pd.NA
    )

    return df

# ---------------------------------------------------------------------------
# Main rollup
# ---------------------------------------------------------------------------

def rollup(submissions_df: pd.DataFrame, batch_codes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Core rollup: one row per planned batch code.

    Parameters
    ----------
    submissions_df : raw SurveyCTO wide-format DataFrame
    batch_codes_df : batch codes frame (one row per scheduled ride)

    Returns
    -------
    DataFrame with one row per planned batch_code, all flags attached.
    """
    subs = clean_submissions(submissions_df)
    subs = add_batch_code_flags(subs)
    subs = add_timing_flags(subs)

    # Left-join frame onto submissions (frame is the source of truth)
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
