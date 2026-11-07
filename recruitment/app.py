"""
app.py — Recruitment HFC Dashboard
====================================
Three views, each with one unit of analysis:

  1 · Ride verification       one row per planned batch code
  2 · Implementation fidelity one row per E1 form (+ supervisor cross-check)
  3 · Enumerator performance  one row per enumerator, with weekly trends

All table logic lives in views.py; this file only lays out the page.

Run locally:
    streamlit run app.py

Requires .streamlit/secrets.toml (never committed) with:
    app_password     = "..."
    submissions_url  = "https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=..."
    batch_codes_url  = "https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=..."
    reviews_url      = "..."   # optional — CSV export URL of the "reviews" tab (see data_io.load_reviews)
"""

import pandas as pd
import streamlit as st

from auth import require_password
from data_io import load_data, load_reviews, append_reviews, can_write_reviews, diagnostics, load_e4, \
    write_patterns, patterns_sheet_id, can_fetch_media, fetch_attachment
import datetime as dt
from rollup import CORRIDORS, DELAY_MAX_MIN, ANN_DUR_MIN, ANN_DUR_MAX, SIGNUP_DIFF_MAX, \
                   PITCH_TARGET_MIN, PITCH_TOL_MIN, RIDE_MIN_MIN, SIGNUP_MIN
import importlib
import rollup as _rollup
import views as v
# Always use the newest rollup.py / views.py on disk — Streamlit can otherwise keep
# an older copy in memory after a push, so app.py and views.py stop matching.
importlib.reload(_rollup)
importlib.reload(v)
EXPECTED_VIEWS_VERSION = "2026-11-07n"

# ---------------------------------------------------------------------------
# Page setup, auth, data
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Recruitment HFC Dashboard", page_icon="🚌", layout="wide")
require_password()

if getattr(v, "VIEWS_VERSION", None) != EXPECTED_VIEWS_VERSION:
    st.error(f"views.py on the server is out of date (version {getattr(v, 'VIEWS_VERSION', 'unknown')}, "
             f"expected {EXPECTED_VIEWS_VERSION}). Copy the latest views.py into recruitment/, push it, "
             f"and reboot the app. Loaded from: {v.__file__}")
    st.stop()

with st.spinner("Loading data…"):
    try:
        submissions_raw, batch_codes, subs = load_data()
        reviews = load_reviews()
        # Corrections about who filled in a form (e.g. a supervisor form saved as E1)
        # are applied in the dashboard straight away; the Sheet itself is not changed.
        raw_fixed, live_applied = v.apply_live_corrections(submissions_raw, reviews)
        if live_applied:
            subs = _rollup.prepare_submissions(raw_fixed)
        # Only monitor rides from the start date onward (training and earlier days are ignored).
        # Change it in Streamlit -> Settings -> Secrets with:  monitor_from = "2026-10-22"
        try:
            MONITOR_FROM = pd.Timestamp(st.secrets.get("monitor_from", "2026-10-22"))
        except Exception:
            MONITOR_FROM = pd.Timestamp("2026-10-22")
        subs = subs[subs["ride_date"].isna() | (subs["ride_date"] >= MONITOR_FROM)].reset_index(drop=True)
        if "batch_code" not in batch_codes.columns:
            st.error("The batch_codes tab was read, but it has no 'batch_code' column. Check that its first row "
                     "is the header (batch_code, ride_day, …) and that batch_codes_url points to that tab. "
                     f"Columns found: {', '.join(map(str, batch_codes.columns[:8]))}")
            st.stop()
        _bc_dates = pd.to_datetime(batch_codes["batch_code"].astype(str).str[:8], format="%d%m%Y", errors="coerce")
        batch_codes = batch_codes[_bc_dates.isna() | (_bc_dates >= MONITOR_FROM)].reset_index(drop=True)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


def show(t: pd.DataFrame, css: pd.DataFrame | None = None, height: int | None = None, legend: bool = False):
    """Render a table with per-cell CSS, hiding helper columns (prefixed '_'). Column descriptions from
    views.COLUMN_HELP appear when hovering over a header; legend=True also lists them under the table."""
    if t is None or t.empty:
        st.info("Nothing to show.")
        return
    keep = [c for c in t.columns if not str(c).startswith("_")]
    t = t[keep].reset_index(drop=True)
    helps = {c: v.COLUMN_HELP[c] for c in keep if c in getattr(v, "COLUMN_HELP", {})}
    kwargs = dict(width="stretch", hide_index=True,
                  column_config={c: st.column_config.Column(help=h) for c, h in helps.items()})
    if height:
        kwargs["height"] = height
    if css is not None:
        css = css[keep].reset_index(drop=True)
        st.dataframe(t.style.apply(lambda _: css, axis=None), **kwargs)
    else:
        st.dataframe(t, **kwargs)
    if legend and helps:
        with st.expander("What the columns mean", expanded=False):
            st.markdown("\n".join(f"- **{c}** — {h}" for c, h in helps.items()))


def download(t: pd.DataFrame, name: str, key: str, label="⬇ Download CSV"):
    keep = [c for c in t.columns if not str(c).startswith("_")]
    st.download_button(label, t[keep].to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv", key=key)


# ---------------------------------------------------------------------------
# Header + sidebar filters (apply to tabs 1 and 2)
# ---------------------------------------------------------------------------

col_title, col_refresh = st.columns([6, 1])
with col_title:
    st.title("🚌 Recruitment HFC Dashboard")
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh"):
        load_data.clear()
        load_reviews.clear()
        st.rerun()

ledger_all, ledger_css_all = v.ride_ledger(subs, batch_codes, reviews)

# E4 (Script Checklist) forms — the independent record used in Today and Tab 2
e4_all = load_e4()
if not e4_all.empty:
    e4_all = e4_all[e4_all["ride_date"].isna() | (e4_all["ride_date"] >= MONITOR_FROM)]
_xt_all, _, _ = v.timing_crosscheck(subs, e4_all, batch_codes, reviews)
verdicts_all, flips_by_code = v.crosscheck_verdicts(_xt_all)

st.sidebar.caption(f"Showing rides from {MONITOR_FROM.strftime('%d %b %Y')} onward.")
st.sidebar.header("Filters (tabs 1 & 2)")
dates = sorted(set(ledger_all["_date"].dropna()) | set(subs["ride_date"].dropna())) if not ledger_all.empty else \
        sorted(set(subs["ride_date"].dropna()))
date_labels = {v.fmt_day(d): d for d in dates}
sel_dates = st.sidebar.multiselect("Ride date", list(date_labels), default=[],
                                   placeholder="All dates")
all_corridors = sorted(set(CORRIDORS.values())
                       | (set(ledger_all["_corridor"].dropna()) if not ledger_all.empty else set()))
sel_corr = st.sidebar.multiselect("Corridor", all_corridors, default=[], placeholder="All corridors")

def filter_ledger(t, css):
    m = pd.Series(True, index=t.index)
    if sel_dates:
        m &= t["_date"].isin([date_labels[d] for d in sel_dates])
    if sel_corr:
        m &= t["_corridor"].isin(sel_corr)
    return t[m], css[m]

def filter_subs(df):
    m = pd.Series(True, index=df.index)
    if sel_dates:
        m &= df["ride_date"].isin([date_labels[d] for d in sel_dates])
    if sel_corr:
        corr = df["p_corridor"].fillna(df["a_corridor"])
        m &= corr.isin(sel_corr)
    return df[m]

subs_f = filter_subs(subs)
st.sidebar.caption("Tab 3 uses every day from the start date, regardless of these filters.")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Today", "1 · Ride verification", "2 · Implementation fidelity",
                                        "3 · Enumerators", "4 · Patterns"])

# ---------------------------------------------------------------------------
# TODAY — what needs action now (phone-friendly)
# ---------------------------------------------------------------------------

with tab0:
    day_opts = sorted(ledger_all["_date"].dropna().unique()) if not ledger_all.empty else []
    if not day_opts:
        st.info("No planned rides in the batch_codes tab.")
    else:
        today = pd.Timestamp.now(tz="Africa/Freetown").normalize().tz_localize(None)
        with_forms = sorted(set(subs["ride_date"].dropna()) & set(day_opts))
        default = today if today in day_opts else (with_forms[-1] if with_forms else day_opts[-1])
        day = st.selectbox("Day", day_opts, index=day_opts.index(default),
                           format_func=lambda d: v.fmt_day(pd.Timestamp(d)) + (" (today)" if d == today else ""),
                           key="today_day")
        board = v.today_board(ledger_all, day, flips_by_code)
        unreadable = ledger_all.loc[ledger_all["_date"].isna(), "Planned code"].tolist() if not ledger_all.empty else []
        if unreadable:
            st.warning(f"{len(unreadable)} planned code(s) in the batch_codes tab can't be read, so they don't "
                       f"appear on any day: {', '.join(unreadable[:10])}" + (" …" if len(unreadable) > 10 else "")
                       + ". Expected format: DDMMYYYY + team + marketer ID + 4-letter route + ride # + N/S/P.")
        forms_day = subs[subs["ride_date"] == day]
        n_rides = int((ledger_all["_date"] == day).sum())
        states = pd.concat([t["State"] for t, _ in board.values()]) if board else pd.Series(dtype=str)
        c = st.columns(4)
        c[0].metric("Rides planned", n_rides)
        n_e1 = int((forms_day["role"] == "Enumerator 1").sum())
        c[1].metric("E1 forms in", n_e1, help="Enumerator 1 forms received for this day")
        c[1].caption(f"+ {len(forms_day) - n_e1} supervisor form(s)")
        c[2].metric("Need action", int(states.isin(["Action", "Overdue"]).sum()))
        last = forms_day["submitted_at"].max() if not forms_day.empty else pd.NaT
        c[3].metric("Last form", (last.tz_localize("UTC").tz_convert("Africa/Freetown").strftime("%H:%M")
                                  if pd.notna(last) else "—"))
        st.caption("🔴 Action / Overdue — not reviewed yet · 🟣 To correct — reviewed, correction still to make · "
                   "🟢 OK / Reviewed · ⚪ Waiting — ride not done yet. "
                   "Press 🔄 Refresh for the newest forms. Record decisions in tab 1.")
        for team, (t, css) in board.items():
            st.markdown(f"##### Team {team}")
            show(t, css)

# ---------------------------------------------------------------------------
# TAB 1 — Ride verification
# ---------------------------------------------------------------------------

with tab1:
    st.caption("Did each planned ride happen as planned, and if not, what still needs resolving? "
               "One row per planned batch code.")
    ledger, ledger_css = filter_ledger(ledger_all, ledger_css_all)

    # Counts use OPEN issues: differences marked OK or corrected no longer count as issues
    counts = ledger["_open_tags"].explode().value_counts() if not ledger.empty else pd.Series(dtype=int)
    n_not_sub = int(ledger["_tags"].map(lambda x: x == ["Not submitted"]).sum()) if not ledger.empty else 0
    n_done = int(ledger.apply(lambda r: not r["_open_tags"] and r["_tags"] != ["Not submitted"], axis=1).sum()) \
        if not ledger.empty else 0
    cols = st.columns(6)
    cols[0].metric("Planned rides", len(ledger))
    cols[1].metric("Done / OK", n_done,
                   help="Done as planned, or every difference marked OK – planned/allowed or corrected")
    cols[2].metric("Not reviewed yet", int(ledger["_needs_decision"].sum()) if not ledger.empty else 0,
                   help="Rides with a flagged difference that has no decision in the reviews tab")
    cols[3].metric("Reviewed", int(ledger["Review status"].str.startswith("✓").sum()) if not ledger.empty else 0,
                   help="Every flagged difference on the ride has a decision (OK / corrected)")
    cols[4].metric("To correct in cleaning", int(ledger["_fix_needed"].sum()) if not ledger.empty else 0,
                   help="Mistakes you marked 'Correct in cleaning' that aren't marked 'Corrected' yet")
    cols[5].metric("Not submitted", n_not_sub)
    st.caption(" · ".join(f"{k}: {int(counts.get(k, 0))}" for k in
                          ["Validity at risk", "Wrong code selected", "Not-allowed change", "Duplicate E1", "To confirm", "Supervisor only"]
                          if counts.get(k, 0)) or "No open flags.")

    f1, f2 = st.columns([3, 2])
    with f1:
        sel_status = st.multiselect("Show status", list(v.STATUS_RANK), default=[],
                                    placeholder="All statuses", key="status_f")
    with f2:
        sel_review = st.selectbox("Review", ["All rides", "Not reviewed yet", "Reviewed", "Correct in cleaning"],
                                  key="review_f")
    m = pd.Series(True, index=ledger.index)
    if sel_status:
        m &= ledger["_tags"].map(lambda tags: any(x in tags for x in sel_status))
    if sel_review == "Not reviewed yet":
        m &= ledger["_needs_decision"]
    elif sel_review == "Reviewed":
        m &= ledger["Review status"].str.startswith("✓")
    elif sel_review == "Correct in cleaning":
        m &= ledger["_fix_needed"]
    if reviews.attrs.get("error"):
        st.warning(reviews.attrs["error"])
    show(ledger[m], ledger_css[m])
    download(ledger[m], "ride_verification.csv", "dl_ledger")

    # ---- Record decisions --------------------------------------------------
    st.markdown("#### Record decisions")
    writable = can_write_reviews()
    st.caption("One row per open issue. Choose a decision, add a note (what happened / what to tell the "
               "enumerator) and, if the data should be corrected, the correct value. Saved rows go to the "
               "**reviews** tab of the Sheet — the flag stays visible and is marked as reviewed. "
               "Decisions are never deleted: to change one, save a new decision for the same issue.")
    if not writable:
        st.info("Saving from the dashboard isn't set up yet — it needs a Google service account in the "
                "Streamlit secrets (see *How to record decisions* below). Until then this table is read-only.")
    d1, d2 = st.columns([2, 3])
    with d1:
        reviewer = st.text_input("Your name", key="reviewer", placeholder="e.g. Sofia")
    with d2:
        st.markdown("<br>", unsafe_allow_html=True)
        include_reviewed = st.toggle("Also show issues already reviewed (to change a decision)", value=False)
    pend = v.pending_decisions(ledger, include_reviewed)
    if pend.empty:
        st.success("No open issues to decide for the selected rides. 🎉")
    else:
        ver = st.session_state.get("dec_ver", 0)
        edited = st.data_editor(
            pend, key=f"decisions_{ver}", hide_index=True, width="stretch",
            disabled=["Planned code", "Ride", "Issue", "Details", "Current decision"],
            column_config={
                "_form_id": None,
                "Decision": st.column_config.SelectboxColumn(
                    "Decision", options=v.DECISION_OPTIONS, width="small",
                    help="OK – planned/allowed = not a mistake · Correct in cleaning = mistake, fill in the correct value · "
                         "Corrected = the correction is now in the cleaning code"),
                "Note": st.column_config.TextColumn("Note", width="large"),
                "Field": st.column_config.TextColumn("Field", width="small",
                                                     help="Variable to correct in cleaning (e.g. batch_code_pre)"),
                "Correct value": st.column_config.TextColumn("Correct value", width="medium",
                                                             help="Leave blank if no data correction is needed"),
                "Details": st.column_config.TextColumn("Details", width="large"),
            },
        )
        chosen = edited[edited["Decision"].fillna("") != ""]
        b1, b2 = st.columns([1, 4])
        with b1:
            save = st.button(f"💾 Save {len(chosen)} decision(s)", type="primary",
                             disabled=not writable or chosen.empty)
        with b2:
            if not chosen.empty and not reviewer.strip():
                st.caption("Add your name before saving.")
        if save:
            if not reviewer.strip():
                st.error("Please add your name first.")
            else:
                now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                rows = [{
                    "timestamp": now, "batch_code": r["Planned code"], "issue": r["Issue"],
                    "decision": r["Decision"], "note": r["Note"] or "", "reviewed_by": reviewer.strip(),
                    "form_id": r["_form_id"] or "",
                    "field": (r["Field"] or "") if (r["Correct value"] or "") else "",
                    "correct_value": r["Correct value"] or "",
                } for _, r in chosen.iterrows()]
                try:
                    n = append_reviews(rows)
                    st.session_state["dec_ver"] = ver + 1
                    st.toast(f"Saved {n} decision(s) to the reviews tab.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save to the Sheet: {e}")

    fixes = v.fixes_to_communicate(subs, batch_codes, reviews)
    if not fixes.empty:
        with st.expander(f"🧹 Corrections to make in cleaning ({len(fixes)})", expanded=True):
            st.caption("Mistakes marked 'Correct in cleaning'. Once the correction is in your do-file, "
                       "save 'Corrected' for the same issue and it drops off this list.")
            show(fixes)
            download(fixes, "fixes_to_communicate.csv", "dl_fixes")

    corr = v.corrections_table(reviews)
    if not corr.empty:
        with st.expander(f"🧹 Corrections for the cleaning do-file ({len(corr)})"):
            st.caption("Latest correction per form and variable. The do-file `apply_reviews.do` reads the "
                       "reviews tab and applies these — `form_id` is the SurveyCTO KEY.")
            show(corr)
            download(corr, "corrections.csv", "dl_corr")

    st.markdown("#### Forms that don't match any planned ride")
    unmatched = v.unmatched_submissions(filter_subs(subs), batch_codes)
    if unmatched.empty:
        st.success("Every form points to a planned batch code.")
    else:
        show(unmatched)

    with st.expander("Full submission log — every form"):
        log = v.submission_log(subs_f, batch_codes)
        show(log)
        download(log, "submission_log.csv", "dl_log")

    with st.expander("How statuses are assigned"):
        st.markdown("""
Each form is assigned to the ride it describes, using the **date, team and ride number entered on the
form** — not only the planned code picked from the list. A ride can carry several statuses; the worst is listed first.

| Status | Meaning | What to do |
|---|---|---|
| 🔴 **Validity at risk** | The E1 or supervisor reported that **another marketer** got on the bus and marketed during the ride, or the supervisor saw **the team's marketer speak or sell during a Sit down pause** (s27) — this can invalidate the ride (e.g. a Sit down ride is no longer a control) | Decide: *OK – planned/allowed* = the ride still counts; *Correct in cleaning* = mark it invalid in cleaning |
| 🔴 **Wrong code selected** | A form describing this ride (by its date, team and ride #) was filed under another planned code. The form is counted here only; the row it was filed under shows a grey note in *Code selection* but is not flagged | Decide: Correct in cleaning, or Corrected |
| 🔴 **Not-allowed change** | The actual code has a different date, route/direction, ride number or treatment, or no readable code | Act now: the E1 should have stopped and called the supervisor; the ride may not count |
| 🔴 **Duplicate E1** | More than one E1 form describes this ride | Same person → keep one; different people → two E1s on one ride, or a wrong code |
| 🔵 **To confirm** | A team or marketer change (allowed) that the E1 says was NOT reported to the supervisor | Check with the supervisor |
| 🔵 **Supervisor only** | Supervisor form(s) but no E1 form for this ride | Chase the E1 form |
| ⚪ **Not submitted** | No form describes this ride | Check whether the ride happened |
| 🟢 **Done as planned** | One E1 form, filed under the right code, actual code = planned code | — |

**Date … Treatment** show what actually happened (from the actual batch code on the E1 form, or the
supervisor form if there is no E1 form). 🔴 Red = date, route/direction, ride # or treatment differs from plan (not allowed);
🔵 blue = team differs (allowed). Marketer ID changes are allowed and not coloured. If forms disagree, both values are shown ("A / B"). "—" = no form yet.

**E1 form / Supervisor form** show who submitted for the ride (✗ = missing).
**What changed** lists planned → actual for each part that differs (one entry per form when there are several).
**Forms that don't match** lists forms whose selected planned code is not in the batch_codes tab.
A **✓** after a status means you've recorded a decision for every issue behind it — the flag stays, so the
history is kept, but you know it's been dealt with. **Review status** sums this up per ride: *Not reviewed*,
*Partly reviewed*, *✓ Reviewed (OK / corrected)* or 🟣 *🧹 Correct in cleaning* (correction still to add to the cleaning code).
**Review notes** shows each decision and its note.

**Needs decision on** lists the issues on that ride you haven't reviewed yet — use these exact names in the reviews tab.
""")

    with st.expander("How to record decisions"):
        st.markdown("""
**Daily routine:** set the *Review* filter to *Not reviewed yet*, then go through **Record decisions**:

| Decision | Use when | What happens |
|---|---|---|
| **OK – planned/allowed** | The difference was planned or is allowed (e.g. the team was moved to another route) | Flag gets a ✓; doesn't count against the enumerator |
| **Correct in cleaning** | It's a mistake and the data must be corrected (e.g. wrong planned code selected) — fill in **Correct value** | Appears in *Corrections to make in cleaning*; shown in the enumerator's feedback card |
| **Corrected** | The correction is now in your cleaning code | Flag gets a ✓; drops off the corrections list; still counts in the enumerator's batch-code score |

**Note** — what happened, or what to tell the enumerator. **Field / Correct value** — fill these when the
data itself must change in cleaning (for a wrong planned code they're pre-filled, e.g.
`batch_code_pre → 051020261000CTEP3N`). Every saved decision is a new row in the Sheet's **reviews**
tab with your name and the time, so the full history is kept; the newest decision for an issue counts.

**Setup (once):** create a Google service account, share the Sheet with its e-mail as *Editor*, and paste
its key into Streamlit → Settings → Secrets under `[gcp_service_account]`. The reviews tab is created
automatically on the first save.
""")

# ---------------------------------------------------------------------------
# TAB 2 — Implementation fidelity
# ---------------------------------------------------------------------------

with tab2:
    st.caption("Were timings, sign-ups and procedures followed, and why not when they weren't? "
               "One row per Enumerator 1 form.")
    fid, fid_css = v.fidelity_table(subs_f, batch_codes, reviews)
    xc, xc_css = v.crosscheck_table(subs_f, batch_codes, reviews)

    if fid.empty:
        st.info("No E1 forms for the selected filters.")
    else:
        st_counts = fid["Issue status"].value_counts()
        c = st.columns(5)
        c[0].metric("E1 forms", len(fid))
        c[1].metric("No issues", int(st_counts.get("No issues", 0)))
        c[2].metric("Outside target, reason given", int(st_counts.get("Outside target, reason given", 0)))
        c[3].metric("Outside target, reason missing", int(st_counts.get("Outside target, no reason", 0) + st_counts.get("Outside target, some reasons missing", 0)))
        c[4].metric("Observed by supervisor", int((xc["Supervisor"] != "— not observed —").sum()))

        only_issues = st.toggle("Only rides with issues", value=False)
        m = fid["Issue status"] != "No issues" if only_issues else pd.Series(True, index=fid.index)
        if only_issues:
            m |= fid["Recording problems"] != "—"

        # Same ride as recorded by the E4 (Script Checklist) and the supervisor
        if e4_all.attrs.get("error"):
            st.warning(e4_all.attrs["error"])
        e4 = e4_all
        if not e4.empty and not subs_f.empty and subs_f["ride_date"].notna().any():
            e4 = e4[e4["ride_date"].isna() | e4["ride_date"].isin(subs_f["ride_date"].unique())]
        xt, xt_css, e4_unmatched = v.timing_crosscheck(subs_f, e4, batch_codes, reviews)
        verdicts, _ = v.crosscheck_verdicts(xt)

        st.markdown("#### Ride timeline and start checks")
        st.caption("Issues first: whether the announcement started when it should (bus fullness, strategic point) "
                   "and the time between steps, against their targets. **Independent check** compares the E1's times "
                   "with the E4 (Script Checklist) and the supervisor: 🟢 agrees or confirms; 🔴 the independent record "
                   "misses a target the E1's times don't; 🟠 a time or duration differs by more than "
                   f"{v.TIME_GAP_MAX} min. Details are in the expander below the table.")
        tl, tl_css = v.ride_timeline_table(subs_f, batch_codes, reviews, verdicts)
        show_clock = st.toggle("Show clock times", value=False, key="tl_clock")
        if not tl.empty:
            cols = [c for c in tl.columns if show_clock or not c.startswith(v.CLOCK_PREFIX)]
            sel = ((tl["Timeline issues"] != "None") | ~tl["_check_ok"]) if only_issues \
                else pd.Series(True, index=tl.index)
            show(tl.loc[sel, cols], tl_css.loc[sel, [c for c in cols if c in tl_css.columns]], legend=True)
            download(tl, "ride_timeline.csv", "dl_tl")

        if not xt.empty:
            n_cmp = int(xt["_compared"].sum()); n_iss = int(xt["_issue"].sum())
            with st.expander(f"Details: E1 vs E4 vs supervisor — {n_iss} ride(s) with differences, "
                             f"{n_cmp}/{len(xt)} with a second record"):
                st.caption(f"**Δ** = E4 or supervisor minus E1, in minutes (+ = recorded later than the E1), listed when "
                           f"more than {v.TIME_GAP_MAX} min apart; blank = they agree. **What happened** reads the "
                           "comparison in plain language.")
                only_diff = st.toggle("Only rides with differences", value=True, key="xt_only")
                sel = xt["_issue"] if only_diff else xt["_compared"]
                if sel.any():
                    show(xt[sel], xt_css[sel])
                else:
                    st.success("No differences between the E1 and the independent records." if n_cmp else
                               "No E4 or supervisor records to compare yet.")
                download(xt, "timing_crosscheck.csv", "dl_xt")
                if not e4_unmatched.empty:
                    st.markdown(f"**E4 checklists not matched to an E1 form ({len(e4_unmatched)})**")
                    st.dataframe(e4_unmatched, hide_index=True, width="stretch")
        try:
            _has_e4 = bool(st.secrets.get("e4_url"))
        except Exception:
            _has_e4 = False
        if not _has_e4:
            st.caption("E4 data isn't connected yet: publish the Script Checklist form to a Google Sheet and add "
                       "e4_url (its CSV export link) to Secrets.")

        try:
            _server = st.secrets.get("surveycto_server")
        except Exception:
            _server = None
        try:
            _form = st.secrets.get("recruitment_form_id", "rec_form_nov2026")
        except Exception:
            _form = "rec_form_nov2026"
        media_rides = v.media_index(subs_f, batch_codes, reviews)
        if media_rides:
            with st.expander(f"📷 🎧 Photos and audio audits ({len(media_rides)} rides)"):
                if can_fetch_media():
                    pick = st.selectbox("Ride", range(len(media_rides)), key="media_ride",
                                        format_func=lambda k: media_rides[k]["label"])
                    ride = media_rides[pick]
                    for lbl, fn in ride["files"]:
                        st.markdown(f"**{lbl}** · `{fn}`")
                        data, msg = fetch_attachment(_form, ride["form_id"], fn)
                        if data is None:
                            st.warning(msg)
                        elif fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            st.image(data, width=420)
                        else:
                            st.audio(data)
                    st.caption("Files are fetched from SurveyCTO by the dashboard (with the login in Secrets) and "
                               "shown only here. Audio includes passengers' voices — don't download or share it.")
                else:
                    st.caption("To view photos and audio here, add surveycto_username and surveycto_password "
                               "(a SurveyCTO user allowed to download this form's data) to Secrets, next to "
                               "surveycto_server. Until then, file names only:")
                    st.dataframe(v.media_table(subs_f, batch_codes, reviews, None, _form),
                                 hide_index=True, width="stretch")

        st.markdown("#### Implementation — values and the reasons given")
        show(fid[m], fid_css[m])
        download(fid[m], "implementation_fidelity.csv", "dl_fid")

        st.markdown("#### Supervisor cross-check — counts, protocol and timing agreement")
        st.caption(f"Gaps are |E1 − supervisor| minutes at each checkpoint; red above {v.TIME_GAP_MAX} min. "
                   "Supervisor forms are paired with E1 forms by the E1 named on the supervisor form, "
                   f"ride date and ride start time (within {v.PAIR_WINDOW_MIN} min).")
        show(xc[m.values] if only_issues else xc, xc_css[m.values] if only_issues else xc_css)
        download(xc, "supervisor_crosscheck.csv", "dl_xc")

    with st.expander("Column guide"):
        st.markdown(f"""
| Column | Flagged (🔴) when |
|---|---|
| Start delay | > {DELAY_MAX_MIN} min after ride start |
| Recorded ann./signup dur. | Outside {ANN_DUR_MIN}–{ANN_DUR_MAX} min (enumerator-recorded) |
| Auto form-screen dur. | Not flagged — SurveyCTO screen timing, for reference |
| Timing discrepancy | Recorded and auto differ by > {SIGNUP_DIFF_MAX} min — review; doesn't mean either is wrong |
| Pause/Pitch | **Sit down:** the pause lasts until the ride ends (in the current forms the pause end *is* the ride end, so only older forms can be flagged here). **Normal:** pitch under 20 min, unless it ran until the ride ended. **Peddling:** pitch under 12 min, unless it ran until the ride ended |
| **Ride timeline** — Start check | Announcement started > {DELAY_MAX_MIN} min after the ride start although the bus was at least half full (🔴), or with no reason; 🟢 if the E1 waited because the bus was less than half full; 🟠 another reason |
| **Ride timeline** — Strategic point | The announcement started AFTER the route's strategic point (asked only when the start was late) |
| **Ride timeline** — Announcement | Automatic length of the script screens outside {ANN_DUR_MIN}–{ANN_DUR_MAX} min |
| **Ride timeline** — Ann. end → sign-up end | Lag between the end of the announcement and the end of sign-up (no target; red if sign-up "ended" > 1 min before the announcement) |
| **Ride timeline** — Sign-up end → pitch/pause | Pitch/pause started > {DELAY_MAX_MIN} min after sign-up ended |
| Pitch start gap *(Nov form)* | Marketer started the pitch > {DELAY_MAX_MIN} min after sign-up ended (b04 → b05; Normal/Peddling rides). For reference — not counted against the enumerator |
| Ride dur. | < {RIDE_MIN_MIN} min |
| Signups | < {SIGNUP_MIN} |

From the Nov form the recorded sign-up duration runs from announcement start to **sign-up end** (b03 → b04);
before it ran to the pitch/pause start. The ride-completion questions were removed, so those columns only
appear for Oct forms. **Sup. grade** in the cross-check is the supervisor's 1–10 grade for the E1.

**Issue status** — whether any value is outside its target, and whether the enumerator wrote a reason
(*reason given* / *no reason*). A reason is context only: it does not make the value acceptable.

**Recording problems** — impossible or inconsistent timestamps (data quality, not protocol).

**Missing/not assessed** (grey) — the value is absent from the form; it neither passes nor fails.
""")

# ---------------------------------------------------------------------------
# TAB 3 — Enumerator performance
# ---------------------------------------------------------------------------

with tab3:
    st.caption("How is each enumerator doing? **Performance indicators** = what they are responsible for "
               "(route-driven durations are not counted). **Averages** = patterns: every enumerator should get "
               "roughly the same sign-ups, announcement length, ride length… — cells far from the overall "
               "average are coloured. Your review decisions apply.")
    rs = v.ride_summary(subs, batch_codes, reviews)
    facts = v.ride_facts(subs, batch_codes, reviews, verdicts_all)
    if rs.empty or facts.empty:
        st.info("No E1 forms yet.")
    else:
        f1, f2 = st.columns(2)
        people = sorted(facts["person"].unique())
        days_all = sorted(facts["date"].dropna().unique())
        with f1:
            who = st.selectbox("Enumerator", ["All enumerators"] + people, key="enum_who")
        with f2:
            day_sel = st.selectbox("Day", ["All days"] + days_all, key="enum_day",
                                   format_func=lambda d: d if isinstance(d, str) else v.fmt_day(pd.Timestamp(d)))
        fm = pd.Series(True, index=facts.index)
        m = pd.Series(True, index=rs.index)
        if who != "All enumerators":
            fm &= facts["person"] == who; m &= rs["person"] == who
        if day_sel != "All days":
            fm &= facts["date"] == day_sel; m &= rs["date"] == day_sel
        fsel = facts[fm].reset_index(drop=True)
        sel = rs[m].reset_index(drop=True)

        st.markdown("#### Performance indicators")
        ind, ind_css = v.enumerator_indicators(fsel)
        show(ind, ind_css, legend=True)
        st.caption(f"Share of rides meeting each indicator (rides assessed in brackets). Red below "
                   f"{v.PERF_TARGET:.0%}; green = all met; grey = not assessed. Checklist: green 93%+, amber 88–92%, "
                   "red below 88%. Not counted here (see Averages): pitch/pause and ride length, sign-up end → pitch/pause, "
                   "sign-ups and time between rides — they depend on the marketer, route and passengers.")

        st.markdown("#### Ride by ride")
        t_num, c_num = v.ride_numbers_table(sel)
        show(t_num, c_num)
        st.caption("Red = wrong code, form sent late, announcement/sign-up outside 8–12 min, recorded time not "
                   "matching the form's own timing, or times not matching the supervisor's. Grey = shown for "
                   "reference, not counted.")

        st.markdown("#### Supervisor observations and feedback")
        t_q, c_q = v.qualitative_table(sel)
        show(t_q, c_q)

        st.divider()
        if st.toggle("Show more: weekly measures, trends and feedback cards", value=False, key="more_meas"):
            st.caption("How is each enumerator doing, and are they improving? Three separate signal families — "
                       "never blended into one score: **data quality** (their own forms), **protocol** "
                       "(supervisor observations) and **agreement** (supervisor's independent record vs theirs). "
                       f"Shares below {v.PERF_TARGET:.0%} are red.")

            e1m, supm = v.performance_records(subs, batch_codes, reviews)
            weeks = sorted(set(e1m["week_start"].dropna() if not e1m.empty else [])
                           | set(supm["week_start"].dropna() if not supm.empty else []))
            if not weeks:
                st.info("No E1 or supervisor forms yet.")
            else:
                week_opts = ["All weeks"] + [f"Week of {v.fmt_day(w)}" for w in reversed(weeks)]
                week_map = {f"Week of {v.fmt_day(w)}": w for w in weeks}
                sel_week = st.selectbox("Period", week_opts, index=0)
                sel_weeks = None if sel_week == "All weeks" else [week_map[sel_week]]

                summ, summ_css, ids = v.performance_summary(e1m, supm, sel_weeks)
                show(summ, summ_css)
                download(summ, "enumerator_performance.csv", "dl_perf")

                with st.expander("What each measure means"):
                    st.markdown(f"""
        | Measure | Share of … | Source |
        |---|---|---|
        | Batch code correct | E1 forms filed under the right planned code with an actual code matching the plan (differences you approved as planned changes don't count against) | E1 form |
        | Timing targets met | E1 forms with no timing flag (delay, ann./signup, pause/pitch, ride length) | E1 form |
        | Complete data | E1 forms with all core values present and no impossible timestamps | E1 form |
        | Recorded = auto timing | E1 forms where recorded and auto durations agree within {SIGNUP_DIFF_MAX} min | E1 form |
        | Protocol (sup. obs.) | Observed behaviours (s01–s26) marked Yes, pooled across rides | Supervisor form |
        | Sign-up count agrees | Supervisor-verified rides where the counts matched | Supervisor form |
        | Times agree | Observed rides where the supervisor said all times matched | Supervisor form |
        | No E1 recording error | Rides where a timing mismatch was not attributed to E1 | Supervisor form |

        Sign-up totals are deliberately left out: they depend heavily on route and passengers.
        """)

                st.markdown("#### Ride by ride — each enumerator's performance on each ride")
                st.caption("One row per ride × enumerator: their E1 form and, when observed, the paired supervisor "
                           "form. ✓ met · ✗ not met · — not assessed. The table above pools these rows.")
                sel_people = st.multiselect("Enumerators", list(summ["Enumerator"]), default=[],
                                            placeholder="All enumerators", key="ride_people")
                pid_map = dict(zip(summ["Enumerator"], ids))
                by_ride, by_ride_css = v.performance_by_ride(
                    subs, e1m, supm, sel_weeks, [pid_map[p] for p in sel_people] if sel_people else None)
                show(by_ride, by_ride_css)
                if not by_ride.empty:
                    download(by_ride, "performance_by_ride.csv", "dl_by_ride")

                st.divider()
                if ids:
                    names = dict(zip(summ["Enumerator"], ids))
                    who = st.selectbox("Enumerator detail", list(names))
                    pid = names[who]

                    left, right = st.columns([3, 2])
                    with left:
                        st.markdown(f"#### {who} — weekly trend")
                        trend = v.performance_trend(e1m, supm, pid)
                        if len(trend) >= 2:
                            st.line_chart(trend * 100, height=260)
                        elif not trend.empty:
                            st.caption("Only one week so far — the chart appears from the second week.")
                        if not trend.empty:
                            tr = trend.copy()
                            tr.index = [v.fmt_day(w) for w in tr.index]
                            st.dataframe(tr.apply(lambda col: col.map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")),
                                         width="stretch")
                    with right:
                        st.markdown("#### Supervisor observation by category")
                        cats = v.category_scores(supm, pid, sel_weeks)
                        if cats.empty or cats["_share"].isna().all():
                            st.caption("No supervisor observations in this period.")
                        else:
                            ccss = pd.DataFrame("", index=cats.index, columns=cats.columns)
                            ccss.loc[cats["_share"] < v.PERF_TARGET, "Score"] = v.RED
                            show(cats, ccss)

                    st.markdown(f"#### {who} — rides")
                    pr, pr_css = v.performance_by_ride(subs, e1m, supm, sel_weeks, [pid])
                    show(pr, pr_css)

                    card = v.feedback_card(e1m, supm, pid, sel_weeks, fixes)
                    with st.expander("📝 Feedback card (to share with the enumerator)", expanded=False):
                        st.markdown(card)
                    safe = "".join(ch if ch.isalnum() else "_" for ch in who)
                    st.download_button("⬇ Download feedback card", card.encode("utf-8"),
                                       file_name=f"feedback_{safe}.md", mime="text/markdown", key="dl_card")

                st.divider()
                with st.expander("All supervisor observations (one row per supervisor form)"):
                    obs, obs_css = v.supervisor_observations(subs)
                    show(obs, obs_css)
                    if not obs.empty:
                        download(obs, "supervisor_observations.csv", "dl_obs")
                    st.markdown("""
        | Category | Indicators |
        |---|---|
        | Preparation | s01 Sat in assigned position |
        | Form use | s03 Followed SurveyCTO in real time |
        | Announcement | s04 Script · s05 Voice · s06 Eye contact · s07 Persuasive · s08 Active stance · s09 Displayed card · s10 Correct pause/resume |
        | Pax questions | s11 Answered correctly · s12 Avoided guessing |
        | Pause period | s15 Respected the pause |
        | Cards | s16 Checked cards for missing info |
        | Tickets | s19 Gave apprentice ticket bag |
        | Payment | s21 Agreed fare count · s22 Got signed receipt |
        | Close-out | s23 Completed recruitment form · s24 Completed RIDE form |
        | Materials | s25 Labelled/stored ride bag · s26 Stored all materials |
        """)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

with st.expander("Diagnostics"):
    diag = diagnostics(submissions_raw, batch_codes, subs)
    if live_applied:
        st.caption("Form corrections applied from the reviews tab: " + "; ".join(live_applied))
    st.markdown("\n".join(["| | |", "|---|---|"] + [f"| {k.replace('_', ' ').capitalize()} | {val} |"
                                                     for k, val in diag.items()]))
    st.caption("If rows = 0, check the Sheet is shared as 'Anyone with link → Viewer' "
               "and the URLs in secrets point to the correct tabs.")


# ---------------------------------------------------------------------------
# TAB 4 — Patterns (weekly look-back): averages, balance, E2/E3, trends, export
# ---------------------------------------------------------------------------

with tab4:
    st.caption("Look back over several days to spot patterns: does an enumerator, route, team or marketer stand "
               "out, are the treatment arms balanced, and is anything getting better or worse over time?")
    facts_p = v.ride_facts(subs, batch_codes, reviews, verdicts_all)
    if facts_p.empty:
        st.info("No E1 forms yet.")
    else:
        d_min, d_max = facts_p["date"].min().date(), facts_p["date"].max().date()
        pc1, pc2 = st.columns(2)
        with pc1:
            rng = st.date_input("Period", value=(d_min, d_max), min_value=d_min, max_value=d_max, key="pat_rng")
        with pc2:
            who_p = st.multiselect("Enumerators", sorted(facts_p["person"].unique()), default=[],
                                   placeholder="All enumerators", key="pat_who")
        lo, hi = (rng if isinstance(rng, (list, tuple)) and len(rng) == 2 else (d_min, d_max))
        fp = facts_p[(facts_p["date"].dt.date >= lo) & (facts_p["date"].dt.date <= hi)]
        if who_p:
            fp = fp[fp["person"].isin(who_p)]
        fp = fp.reset_index(drop=True)
        st.caption(f"{len(fp)} rides selected.")

        st.markdown("#### Averages — patterns by enumerator, route, team, treatment…")
        gb_label = st.selectbox("Group by", list(v.GROUP_BY), key="avg_by")
        av, av_css = v.averages_table(fp, v.GROUP_BY[gb_label])
        show(av, av_css, legend=True)
        st.caption("Average per ride. Shading = how far a group's average is from the overall average, measured "
                   "in standard deviations (SD) of the individual rides (the grey bottom row shows 1 SD for each "
                   f"measure): 🟠 {v.AVG_SD_AMBER}–{v.AVG_SD_RED} SD away, 🔴 more than {v.AVG_SD_RED} SD away; "
                   "▲ above / ▼ below the average. **Sign-ups:** more is better, so above the average is 🟢 green "
                   "(below is still 🟠/🔴). Groups with fewer than "
                   f"{v.AVG_MIN_RIDES} rides are not shaded — a couple of rides can be far off by chance.")

        st.markdown("#### Balance between treatment arms")
        within_lbl = st.radio("Compare", ["All rides together", "Within each route", "Within each direction",
                                          "Within each team"], horizontal=True, key="bal_within")
        within = {"All rides together": None, "Within each route": "corridor",
                  "Within each direction": "direction", "Within each team": "team"}[within_lbl]
        bal, bal_css = v.balance_table(fp, within=within)
        if bal.empty or bal.shape[1] <= 2:
            st.info("Need rides from at least two treatment arms to compare.")
        else:
            show(bal, bal_css, legend=True)
            st.caption("Everything before and around the pitch should look the same in every arm — only the pitch "
                       "differs. Sit down is the control. Each cell: average or share (rides). **Std. diff** = "
                       "difference ÷ pooled standard deviation: 🟢 within ±0.25, 🟠 0.25–0.5, 🔴 above 0.5 — coloured "
                       f"only once each arm has {v.BALANCE_MIN_N}+ rides (grey = too few to judge). Pick one enumerator "
                       "in the filter above to see whether *they* behave differently by treatment.")

        e23, e23_css = v.e2e3_signups(subs[subs["form_id"].isin(fp["form_id"])], fp, v.GROUP_BY[gb_label])
        if not e23.empty:
            st.markdown("#### Sign-ups collected by Enumerators 2 and 3")
            show(e23, e23_css)
            st.caption("Normal sign-up cards counted by the E1 for each E2/E3 (c01_normal_e2/e3), split by the same "
                       "'Group by'. Shaded the same way as the averages: 🟢 above the average, 🟠/🔴 below (0.5–1 / more than 1 SD). To check that the right "
                       "sign-up cards were used, open the sign-up photos in Tab 2 → Photos and audio audits.")


        st.markdown("#### Trends over time")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            m_lbl = st.selectbox("Measure", [lbl for _, lbl, *_ in v.METRICS], key="tr_m")
        with tc2:
            by_lbl = st.selectbox("One line per", ["Enumerator 1", "Route (corridor)", "Direction", "Team",
                                                   "Treatment", "Marketer"], key="tr_by")
        with tc3:
            per = st.radio("Period", ["day", "week"], horizontal=True, key="tr_per")
        m_col = next(c for c, lbl, *_ in v.METRICS if lbl == m_lbl)
        tr = v.trend_table(fp, m_col, v.GROUP_BY[by_lbl], per)
        if tr.empty:
            st.info("No data for this measure yet.")
        elif len(tr) < 2:
            st.caption("Only one " + per + " so far — the chart appears from the second.")
            st.dataframe(tr.round(1), width="stretch")
        else:
            st.line_chart(tr, height=320)
            st.caption("Average per " + per + ". Gaps = no rides for that line on that " + per + ".")

        st.divider()
        st.markdown("#### Export")
        tables = v.patterns_export(fp, subs[subs["form_id"].isin(fp["form_id"])])
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button("⬇ Download all as Excel", v.to_excel_bytes(tables),
                               file_name=f"recruitment_patterns_{lo:%Y%m%d}_{hi:%Y%m%d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_patterns")
            st.caption("One sheet per table: averages by every grouping, balance (overall and within route, "
                       "direction, team), E2/E3 sign-ups, trends by day, and one row per ride.")
        with ec2:
            if can_write_reviews():
                if st.button("💾 Save to Google Drive (Sheet)", key="save_patterns"):
                    try:
                        url = write_patterns(tables, dt.datetime.now().strftime("%d %b %Y %H:%M"))
                        st.success(f"Saved: the 'P · …' tabs in [this Google Sheet]({url}) now hold these tables. "
                                   "File → Download → Microsoft Excel gives the .xlsx.")
                    except Exception as e:
                        st.error(f"Could not save: {e}")
                st.caption("Writes every table as a 'P · …' tab of a Google Sheet (the HFC Sheet, or the one in "
                           "patterns_sheet_id in Secrets), overwriting the previous save.")
            else:
                st.caption("Saving to Google Drive needs the service-account key in Secrets.")
