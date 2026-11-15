"""
Forms and Attendance — second page of the Recruitment HFC app
==============================================================
Live view of who worked each day and whether all their SurveyCTO forms came in:
Recruitment (E1), Script Checklist (E4), BnB Drug Info (E4, 2 per marketer), BnB Data Entry (E2 + E3).
Nothing to download: every form is published by SurveyCTO to a Google Sheet and read here.

Uses the Recruitment app's Secrets (submissions_url, batch_codes_url, e4_url) plus:
    roster_url        = "https://docs.google.com/spreadsheets/d/ID/export?format=csv&gid=ROSTER_GID"
    batch_codes_url   = "…gid=BATCH_CODES_GID"
    recruitment_url   = "…"      # Recruitment form data tab
    script_url        = "…"      # Script Checklist data tab
    bnb_druginfo_url  = "…"      # BnB Drug Information Form data tab
    bnb_dataentry_url = "…"      # BnB Data Entry data tab
    monitor_from      = "2026-11-12"   # optional
  Optional column overrides if a form uses different field names, e.g.
    [forms.bnb_dataentry]
    role_col = "enum_role"
    person_cols = { E2 = ["e2_name"], E3 = ["e3_name"] }
"""

import copy
import datetime as dt

import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attendance_forms as F
from auth import require_password

st.set_page_config(page_title="Forms and Attendance", page_icon="🗂️", layout="wide")


require_password()          # same password as the Recruitment HFC page (shared session)


# --------------------------------------------------------------------------- data
def _read(url: str, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.warning(f"Could not read **{name}**: {e}. Check the link in Secrets and that the Sheet is shared "
                   "as 'Anyone with the link → Viewer'.")
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df.replace({"": None})


@st.cache_data(ttl=300)
def load_all():
    alias = {"recruitment_url": "submissions_url", "script_url": "e4_url"}
    get = lambda k: st.secrets.get(k) or (st.secrets.get(alias[k]) if k in alias else None)
    roster = F.clean_roster(_read(get("roster_url"), "roster") if get("roster_url") else pd.DataFrame())
    plan = F.plan_table(_read(get("batch_codes_url"), "batch codes") if get("batch_codes_url") else pd.DataFrame())
    cfgs = copy.deepcopy(F.FORM_DEFAULTS)
    overrides = st.secrets.get("forms", {})
    parts, status = [], {}
    for f in F.FORM_ORDER:
        for k, val in dict(overrides.get(f, {})).items():
            cfgs[f][k] = dict(val) if hasattr(val, "keys") else val
        url = get(f"{f}_url")
        if not url:
            status[f] = "not connected"
            continue
        raw = _read(url, cfgs[f]["label"])
        s = F.submissions(raw, f, cfgs[f], roster)
        status[f] = f"{len(s)} forms"
        parts.append(s)
    subs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["form", "date", "code", "team", "role", "person", "key", "submitted_at", "marketer"])
    return roster, plan, subs, status


def show(t, css):
    if t is None or t.empty:
        st.info("Nothing to show.")
        return
    st.dataframe(t.style.apply(lambda _: css, axis=None), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- page
c1, c2 = st.columns([6, 1])
c1.title("🗂️ Forms and Attendance")
if c2.button("🔄 Refresh"):
    load_all.clear()
    st.rerun()

roster, plan, subs, status = load_all()
try:
    start = pd.Timestamp(st.secrets.get("monitor_from", "2026-11-12"))
except Exception:
    start = pd.Timestamp("2026-11-12")
subs = subs.copy()
subs["date"] = pd.to_datetime(subs["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
roster = roster.copy()
roster["date"] = pd.to_datetime(roster["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
plan = plan.copy()
plan["date"] = pd.to_datetime(plan["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
subs = subs[subs["date"].isna() | (subs["date"] >= start)]

st.caption("Data sources: " + " · ".join(f"{F.FORM_DEFAULTS[f]['label']}: {status.get(f, '—')}" for f in F.FORM_ORDER)
           + f" · Roster: {len(roster)} rows · Planned rides: {len(plan)}")

days = sorted(set(roster["date"].dropna()) | set(plan["date"].dropna()) | set(subs["date"].dropna()))
days = [d for d in days if d >= start]
if not days:
    st.info("No roster, planned rides or forms yet from the start date.")
    st.stop()
today = pd.Timestamp(dt.date.today())
past = [i for i, d in enumerate(days) if d <= today]
default = days.index(today) if today in days else (past[-1] if past else 0)

tab_day, tab_att = st.tabs(["Day", "Attendance"])

with tab_day:
    day = st.selectbox("Day", days, index=default, format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y")
                       + (" (today)" if pd.Timestamp(d) == today else ""))
    t, css = F.day_people(day, roster, plan, subs)
    if not t.empty:
        rostered = t[t["Attendance"] != "Not on roster"]
        k = st.columns(4)
        k[0].metric("Rostered", len(rostered))
        k[1].metric("Present", int((rostered["Attendance"] == "Present").sum()))
        k[2].metric("Absent / no forms", int((rostered["Attendance"] != "Present").sum()))
        k[3].metric("Still missing forms", int(rostered["All forms in"].str.contains("missing").sum()))
    st.markdown("#### People")
    st.caption("Each form: submitted / expected. Expected per day — Recruitment: 1 per ride (E1) · Script Checklist: "
               "1 per ride (E4) · BnB Drug Info: 2 per marketer (E4) · BnB Data Entry: 1 per ride each (E2, E3). "
               "🟢 complete · 🟠 some missing · 🔴 none yet · purple = submitted but not on the roster.")
    show(t, css)

    st.markdown("#### Rides")
    st.caption("Which per-ride forms came in for each planned ride (matched by batch code).")
    tr, trc = F.day_rides(day, roster, plan, subs)
    show(tr, trc)

with tab_att:
    st.caption("✓ = submitted at least one form that day · ✗ absent = on the roster but no form · "
               "purple = worked but not on the roster.")
    sel = st.slider("Days", 1, len(days), min(14, len(days)), help="How many of the most recent days to show")
    tg, tgc = F.attendance_grid(roster, subs, days[-sel:])
    show(tg, tgc)
    st.download_button("⬇ Download attendance (CSV)", tg.to_csv(index=False).encode("utf-8"),
                       file_name="attendance.csv", mime="text/csv")

with st.expander("Set-up and how matching works"):
    st.markdown("""
- **Roster tab** (one row per person per day): `date`, `team`, `role` (E1, E2, E3, E4, Supervisor), `name`,
  and optionally `code` (the person's SurveyCTO choice code). Names are matched ignoring spaces, dots and capitals;
  a short name ('Fatu') matches a full one ('Fatu Conteh') when there is only one match.
- **Forms** are matched to the day by the date on the form (or the batch code / submission date), and to rides by
  the planned batch code. Test usernames are ignored.
- **Attendance**: present = at least one form submitted that day.
""")
