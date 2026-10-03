"""
app.py — Recruitment HFC Dashboard
====================================
Passenger recruitment monitoring. One row per planned batch code.
Values shown in red when outside acceptable range.

Run locally:
    streamlit run app.py

Requires .streamlit/secrets.toml with:
    app_password     = "..."
    submissions_url  = "https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=..."
    batch_codes_url  = "https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=..."
"""

import pandas as pd
import numpy as np
import streamlit as st

from auth import require_password
from data_io import load_data, diagnostics
from rollup import CORRIDORS, DELAY_MAX_MIN, ANN_DUR_MIN, ANN_DUR_MAX, \
                   SIGNUP_DIFF_MAX, PITCH_TARGET_MIN, PITCH_TOL_MIN, \
                   RIDE_MIN_MIN

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recruitment HFC Dashboard",
    page_icon="🚌",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

require_password()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading data…"):
    try:
        submissions_raw, batch_codes, result = load_data()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Build display table
# ---------------------------------------------------------------------------

def build_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ride date
    for col in ["ride_day", "ride_month", "ride_year"]:
        out[col] = pd.to_numeric(out.get(col, pd.Series(dtype=float)), errors="coerce")
    try:
        out["ride_date"] = pd.to_datetime(
            out[["ride_year", "ride_month", "ride_day"]].rename(
                columns={"ride_year": "year", "ride_month": "month", "ride_day": "day"}
            ), errors="coerce",
        ).dt.date
    except Exception:
        out["ride_date"] = pd.NaT

    # Corridor and treatment from parsed planned batch code
    out["corridor"] = out.get("p_corridor", pd.Series(dtype=str))
    treat_map = {"S": "Script", "N": "Normal", "P": "Peddling"}
    out["treatment"] = out.get("p_treatment", pd.Series(dtype=str)).map(treat_map)

    # Numeric measurement columns
    for col in ["announcement_start_delay", "announcement_duration",
                "signup_duration", "pause_duration", "ride_duration", "total_signups"]:
        out[col] = pd.to_numeric(out.get(col, pd.Series(dtype=float)), errors="coerce")

    # Batch code issues — short text label
    def batch_issue(row):
        if pd.isna(row.get("actual_batch_code")):
            return ""
        parts = []
        if row.get("flag_design_problem") == 1:
            parts.append("Design")
        if row.get("flag_needs_confirmation") == 1:
            parts.append("Confirm")
        return ", ".join(parts)

    out["batch_issue"] = out.apply(batch_issue, axis=1)

    return out


display = build_display(result)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

# Date
all_dates = sorted(display["ride_date"].dropna().unique())
if all_dates:
    date_opts = ["All dates"] + [str(d) for d in all_dates]
    sel_dates = st.sidebar.multiselect("Ride date", date_opts, default=["All dates"])
    if "All dates" not in sel_dates and sel_dates:
        display = display[display["ride_date"].astype(str).isin(sel_dates)]

# Corridor
all_corridors = sorted(set(CORRIDORS.values()))
sel_corridors = st.sidebar.multiselect("Corridor", all_corridors, default=all_corridors)
if sel_corridors:
    display = display[display["corridor"].isin(sel_corridors)]

# Status
status_filter = st.sidebar.radio(
    "Submission status", ["All", "Submitted only", "No submission only"], index=0
)
if status_filter == "Submitted only":
    display = display[display["submission_status"] == "Submitted"]
elif status_filter == "No submission only":
    display = display[display["submission_status"] == "No submission"]

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh data"):
    load_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚌 Recruitment HFC Dashboard")
st.caption(
    "One row per planned batch code. "
    "Values shown in **red** are outside the acceptable range."
)

# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------

total     = len(display)
submitted = (display["submission_status"] == "Submitted").sum()
no_sub    = (display["submission_status"] == "No submission").sum()

flag_cols = [c for c in display.columns if c.startswith("flag_")]
any_flag  = display[flag_cols].apply(
    lambda row: row.fillna(0).eq(1).any(), axis=1
).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Planned rides", total)
c2.metric("Submitted",     submitted)
c3.metric("No submission", no_sub)
c4.metric("Any issue",     int(any_flag))

st.markdown("---")

# ---------------------------------------------------------------------------
# Build styled ride-by-ride table
# ---------------------------------------------------------------------------

def make_table(df: pd.DataFrame) -> pd.DataFrame:
    """Select and label columns for display."""
    has_sub = df["submission_status"] == "Submitted"

    t = pd.DataFrame()
    t["Date"]         = df["ride_date"]
    t["Planned code"] = df["batch_code"]
    t["Actual code"]  = df["actual_batch_code"].where(has_sub, "—")
    t["Status"]       = df["submission_status"]
    t["Corridor"]     = df["corridor"]
    t["Treatment"]    = df["treatment"]
    t["Batch issue"]  = df["batch_issue"].where(has_sub, "")

    def fmt(series, fmt_str="{:.1f}"):
        return series.apply(
            lambda v: fmt_str.format(v) if pd.notna(v) else "—"
        ).where(has_sub, "—")

    t["Start delay"]   = fmt(df["announcement_start_delay"])
    t["Ann. dur."]     = fmt(df["announcement_duration"])
    t["Signup dur."]   = fmt(df["signup_duration"])
    t["Pause/Pitch"]   = fmt(df["pause_duration"])
    t["Ride dur."]     = fmt(df["ride_duration"])
    t["Signups"]       = fmt(df["total_signups"], "{:.0f}")

    return t


def style_table(t: pd.DataFrame, df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply red text to cells with out-of-range values."""
    RED  = "color: #CC0000; font-weight: 600"
    GRAY = "color: #888888"

    styles = pd.DataFrame("", index=t.index, columns=t.columns)
    has_sub = df["submission_status"] == "Submitted"

    # Actual code — red if different from planned
    diff = df.get("flag_batch_code_different", pd.Series(0, index=df.index))
    styles.loc[diff == 1, "Actual code"] = RED

    # Batch issue — red if non-empty
    styles.loc[t["Batch issue"].str.len() > 0, "Batch issue"] = RED

    # Start delay > DELAY_MAX_MIN
    delay = df["announcement_start_delay"]
    styles.loc[has_sub & (delay > DELAY_MAX_MIN), "Start delay"] = RED

    # Ann. duration outside [ANN_DUR_MIN, ANN_DUR_MAX]
    ann = df["announcement_duration"]
    styles.loc[has_sub & ((ann < ANN_DUR_MIN) | (ann > ANN_DUR_MAX)), "Ann. dur."] = RED

    # Signup duration: |signup - ann| > SIGNUP_DIFF_MAX
    sig = df["signup_duration"]
    styles.loc[has_sub & (abs(sig - ann) > SIGNUP_DIFF_MAX), "Signup dur."] = RED

    # Pause/Pitch outside [PITCH_TARGET_MIN ± PITCH_TOL_MIN]
    pause = df["pause_duration"]
    styles.loc[
        has_sub & (abs(pause - PITCH_TARGET_MIN) > PITCH_TOL_MIN), "Pause/Pitch"
    ] = RED

    # Ride duration < RIDE_MIN_MIN
    ride = df["ride_duration"]
    styles.loc[has_sub & (ride < RIDE_MIN_MIN), "Ride dur."] = RED

    # Signups < 14
    signups = df["total_signups"]
    styles.loc[has_sub & (signups < 14), "Signups"] = RED

    # Gray out no-submission rows
    no_sub_idx = df[~has_sub].index
    for col in ["Actual code", "Batch issue", "Start delay", "Ann. dur.",
                "Signup dur.", "Pause/Pitch", "Ride dur.", "Signups"]:
        styles.loc[no_sub_idx, col] = GRAY

    return t.style.apply(lambda _: styles, axis=None)


st.subheader("Ride-by-ride monitoring")
st.markdown(
    f"Thresholds: start delay > {DELAY_MAX_MIN} min · "
    f"announcement {ANN_DUR_MIN}–{ANN_DUR_MAX} min · "
    f"signup discrepancy > {SIGNUP_DIFF_MAX} min · "
    f"pause/pitch {PITCH_TARGET_MIN}±{PITCH_TOL_MIN} min · "
    f"ride < {RIDE_MIN_MIN} min · signups < 14"
)

t = make_table(display)
styled = style_table(t, display)

st.dataframe(styled, use_container_width=True, hide_index=True)

# CSV download (raw values, not styled)
csv = t.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download as CSV",
    data=csv,
    file_name="recruitment_hfc.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Column guide
# ---------------------------------------------------------------------------

with st.expander("Column guide"):
    st.markdown(f"""
| Column | What it shows | Flagged red when |
|--------|--------------|-----------------|
| Actual code | Batch code submitted by enumerator | Different from planned code |
| Batch issue | Design or confirmation problem | "Design" = wrong treatment/corridor; "Confirm" = different date or marketer |
| Start delay | Minutes between ride start and announcement start | > {DELAY_MAX_MIN} min |
| Ann. dur. | Announcement duration (min) | < {ANN_DUR_MIN} or > {ANN_DUR_MAX} min |
| Signup dur. | Signup duration (min) | Differs from announcement by > {SIGNUP_DIFF_MAX} min |
| Pause/Pitch | Pause (S) or pitch (N) duration (min) | Outside {PITCH_TARGET_MIN}±{PITCH_TOL_MIN} min (i.e. < {PITCH_TARGET_MIN - PITCH_TOL_MIN} or > {PITCH_TARGET_MIN + PITCH_TOL_MIN}) |
| Ride dur. | Total ride duration (min) | < {RIDE_MIN_MIN} min |
| Signups | Total sign-ups on this ride | < 14 |
""")

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

with st.expander("Diagnostics"):
    diag = diagnostics(submissions_raw, batch_codes, result)
    st.markdown(f"""
| | |
|---|---|
| Submissions tab rows read | {diag['submissions_tab_rows']} |
| Batch codes tab rows read | {diag['batch_codes_tab_rows']} |
| Rollup rows (planned rides) | {diag['rollup_rows']} |
| Submitted | {diag['submitted']} |
| No submission | {diag['no_submission']} |
""")
    st.caption(
        "If rows = 0, check the Sheet is shared as 'Anyone with link → Viewer' "
        "and the URLs in secrets.toml point to the correct tabs."
    )
