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
from data_io import load_data, load_reviews, append_reviews, can_write_reviews, diagnostics
import datetime as dt
from rollup import CORRIDORS, DELAY_MAX_MIN, ANN_DUR_MIN, ANN_DUR_MAX, SIGNUP_DIFF_MAX, \
                   PITCH_TARGET_MIN, PITCH_TOL_MIN, RIDE_MIN_MIN, SIGNUP_MIN
import views as v

# ---------------------------------------------------------------------------
# Page setup, auth, data
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Recruitment HFC Dashboard", page_icon="🚌", layout="wide")
require_password()

with st.spinner("Loading data…"):
    try:
        submissions_raw, batch_codes, subs = load_data()
        reviews = load_reviews()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


def show(t: pd.DataFrame, css: pd.DataFrame | None = None, height: int | None = None):
    """Render a table with per-cell CSS, hiding helper columns (prefixed '_')."""
    if t is None or t.empty:
        st.info("Nothing to show.")
        return
    keep = [c for c in t.columns if not str(c).startswith("_")]
    t = t[keep].reset_index(drop=True)
    kwargs = dict(width="stretch", hide_index=True)
    if height:
        kwargs["height"] = height
    if css is not None:
        css = css[keep].reset_index(drop=True)
        st.dataframe(t.style.apply(lambda _: css, axis=None), **kwargs)
    else:
        st.dataframe(t, **kwargs)


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

st.sidebar.header("Filters (tabs 1 & 2)")
dates = sorted(set(ledger_all["_date"].dropna()) | set(subs["ride_date"].dropna())) if not ledger_all.empty else \
        sorted(set(subs["ride_date"].dropna()))
date_labels = {v.fmt_day(d): d for d in dates}
sel_dates = st.sidebar.multiselect("Ride date", list(date_labels), default=[],
                                   placeholder="All dates")
all_corridors = sorted(set(CORRIDORS.values()))
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
st.sidebar.caption("Tab 3 uses all dates — pick weeks inside the tab.")

tab0, tab1, tab2, tab3 = st.tabs(["Today", "1 · Ride verification", "2 · Implementation fidelity",
                                  "3 · Enumerator performance"])

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
        board = v.today_board(ledger_all, day)
        forms_day = subs[subs["ride_date"] == day]
        n_rides = int((ledger_all["_date"] == day).sum())
        states = pd.concat([t["State"] for t, _ in board.values()]) if board else pd.Series(dtype=str)
        c = st.columns(4)
        c[0].metric("Rides planned", n_rides)
        c[1].metric("Forms in", len(forms_day))
        c[2].metric("Need action", int(states.isin(["Action", "Overdue"]).sum()))
        last = forms_day["submitted_at"].max() if not forms_day.empty else pd.NaT
        c[3].metric("Last form", (last.tz_localize("UTC").tz_convert("Africa/Freetown").strftime("%H:%M")
                                  if pd.notna(last) else "—"))
        st.caption("🔴 Action / Overdue — call the team · 🟢 OK / Reviewed · ⚪ Waiting — ride not done yet. "
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

    counts = ledger["_tags"].explode().value_counts() if not ledger.empty else pd.Series(dtype=int)
    cols = st.columns(6)
    cols[0].metric("Planned rides", len(ledger))
    cols[1].metric("Done as planned", int(counts.get("Done as planned", 0)))
    cols[2].metric("Not reviewed yet", int(ledger["_needs_decision"].sum()) if not ledger.empty else 0,
                   help="Rides with a flagged difference that has no decision in the reviews tab")
    cols[3].metric("Reviewed", int(ledger["Review status"].str.startswith("✓").sum()) if not ledger.empty else 0,
                   help="Every flagged difference on the ride has a decision (approved / fixed)")
    cols[4].metric("Fix needed", int(ledger["_fix_needed"].sum()) if not ledger.empty else 0,
                   help="You decided this needs communicating to the enumerator(s)")
    cols[5].metric("Not submitted", int(counts.get("Not submitted", 0)))
    st.caption(" · ".join(f"{k}: {int(counts.get(k, 0))}" for k in
                          ["Wrong code selected", "Design problem", "Duplicate E1", "To confirm", "Supervisor only"]
                          if counts.get(k, 0)) or "No open flags.")

    f1, f2 = st.columns([3, 2])
    with f1:
        sel_status = st.multiselect("Show status", list(v.STATUS_RANK), default=[],
                                    placeholder="All statuses", key="status_f")
    with f2:
        sel_review = st.selectbox("Review", ["All rides", "Not reviewed yet", "Reviewed", "Fix needed"],
                                  key="review_f")
    m = pd.Series(True, index=ledger.index)
    if sel_status:
        m &= ledger["_tags"].map(lambda tags: any(x in tags for x in sel_status))
    if sel_review == "Not reviewed yet":
        m &= ledger["_needs_decision"]
    elif sel_review == "Reviewed":
        m &= ledger["Review status"].str.startswith("✓")
    elif sel_review == "Fix needed":
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
                    help="Approved = planned/accepted change · Fix needed = tell the enumerator / correct it · "
                         "Fixed = dealt with"),
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
        with st.expander(f"🔧 Fixes to communicate to enumerators ({len(fixes)})", expanded=True):
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
| 🔴 **Wrong code selected** | A form describing this ride (by its date, team and ride #) was filed under another planned code. The form is counted here only; the row it was filed under shows a grey note in *Code selection* but is not flagged | Decide: Fix needed or Fixed |
| 🔴 **Design problem** | The actual code has a different treatment, corridor or direction, or no readable code | Act now: the ride may not count for the design |
| 🔴 **Duplicate E1** | More than one E1 form describes this ride | Same person → keep one; different people → two E1s on one ride, or a wrong code |
| 🔵 **To confirm** | Date, team, marketer ID or ride # differ from plan | Confirm with the team |
| 🔵 **Supervisor only** | Supervisor form(s) but no E1 form for this ride | Chase the E1 form |
| ⚪ **Not submitted** | No form describes this ride | Check whether the ride happened |
| 🟢 **Done as planned** | One E1 form, filed under the right code, actual code = planned code | — |

**Date … Treatment** show what actually happened (from the actual batch code on the E1 form, or the
supervisor form if there is no E1 form). 🔴 Red = treatment or route differs from plan; 🔵 blue = date, team,
marketer ID or ride # differs. If forms disagree, both values are shown ("A / B"). "—" = no form yet.

**E1 form / Supervisor form** show who submitted for the ride (✗ = missing).
**What changed** lists planned → actual for each part that differs (one entry per form when there are several).
**Forms that don't match** lists forms whose selected planned code is not in the batch_codes tab.
A **✓** after a status means you've recorded a decision for every issue behind it — the flag stays, so the
history is kept, but you know it's been dealt with. **Review status** sums this up per ride: *Not reviewed*,
*Partly reviewed*, *✓ Reviewed (approved / fixed)* or 🟣 *Fix needed* (still to communicate or correct).
**Review notes** shows each decision and its note.

**Needs decision on** lists the issues on that ride you haven't reviewed yet — use these exact names in the reviews tab.
""")

    with st.expander("How to record decisions"):
        st.markdown("""
**Daily routine:** set the *Review* filter to *Not reviewed yet*, then go through **Record decisions**:

| Decision | Use when | What happens |
|---|---|---|
| **Approved** | The difference was planned or is acceptable (e.g. the team was moved to another route) | Flag gets a ✓; doesn't count against the enumerator |
| **Fix needed** | A mistake to tell the enumerator about and/or correct | Appears in *Fixes to communicate* and the enumerator's feedback card |
| **Fixed** | The mistake has been communicated / corrected | Flag gets a ✓; still counts in the enumerator's batch-code score |

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
    fid, fid_css = v.fidelity_table(subs_f)
    xc, xc_css = v.crosscheck_table(subs_f)

    if fid.empty:
        st.info("No E1 forms for the selected filters.")
    else:
        st_counts = fid["Issue status"].value_counts()
        c = st.columns(5)
        c[0].metric("E1 forms", len(fid))
        c[1].metric("No issues", int(st_counts.get("No issues", 0)))
        c[2].metric("All explained", int(st_counts.get("All explained", 0)))
        c[3].metric("Unexplained / partly", int(st_counts.get("Unexplained", 0) + st_counts.get("Partly explained", 0)))
        c[4].metric("Observed by supervisor", int((xc["Supervisor"] != "— not observed —").sum()))

        only_issues = st.toggle("Only rides with issues", value=False)
        m = fid["Issue status"] != "No issues" if only_issues else pd.Series(True, index=fid.index)
        if only_issues:
            m |= fid["Recording problems"] != "—"

        st.markdown("#### Implementation — values and the reasons given")
        show(fid[m], fid_css[m])
        download(fid[m], "implementation_fidelity.csv", "dl_fid")

        st.markdown("#### Supervisor cross-check — independent record of the same ride")
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
| Pause/Pitch | Outside {PITCH_TARGET_MIN}±{PITCH_TOL_MIN} min |
| Ride dur. | < {RIDE_MIN_MIN} min |
| Signups | < {SIGNUP_MIN} |

**Issue status** — *All explained*: every flagged value has a reason, or an external disruption was
reported (bus/route, passenger, marketer/pitch interruption, recruitment paused). *Unexplained*: flagged
with no reason and no external disruption. A reason being given is not a judgement that it is valid.

**Recording problems** — impossible or inconsistent timestamps (data quality, not protocol).

**Missing/not assessed** (grey) — the value is absent from the form; it neither passes nor fails.
""")

# ---------------------------------------------------------------------------
# TAB 3 — Enumerator performance
# ---------------------------------------------------------------------------

with tab3:
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
| Flags explained | E1 forms with flags where every flag has a reason or external disruption | E1 form |
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
    st.markdown("\n".join(["| | |", "|---|---|"] + [f"| {k.replace('_', ' ').capitalize()} | {val} |"
                                                     for k, val in diag.items()]))
    st.caption("If rows = 0, check the Sheet is shared as 'Anyone with link → Viewer' "
               "and the URLs in secrets point to the correct tabs.")
