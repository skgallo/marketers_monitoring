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
    dataentry_url     = "…"      # Data Entry (sign-up cards) data tab — 27 per ride
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


# --------------------------------------------------------------------------- name fixes (fixes tab in the roster Sheet)
FIX_COLS = ["KEY", "name", "filed_under", "form", "date", "saved_at"]


def _roster_sheet_id():
    import re
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", str(st.secrets.get("roster_url", "")))
    return m.group(1) if m else None


def _fixes_ws(create=False):
    """The 'fixes' worksheet in the roster Sheet via the service account, or None if not possible."""
    try:
        import gspread
        from data_io import _service_account_info
        info = _service_account_info()
        sid = _roster_sheet_id()
        if not info or not sid:
            return None
        sh = gspread.service_account_from_dict(info).open_by_key(sid)
        try:
            ws = sh.worksheet("fixes")
        except gspread.WorksheetNotFound:
            if not create:
                return None
            ws = sh.add_worksheet("fixes", rows=500, cols=len(FIX_COLS))
        if not ws.row_values(1):
            ws.update(values=[FIX_COLS], range_name="A1")
        return ws
    except Exception as e:
        st.session_state["fixes_error"] = str(e)
        return None


def _load_fixes() -> pd.DataFrame:
    ws = _fixes_ws()
    if ws is not None:
        try:
            return pd.DataFrame(ws.get_all_records(), dtype=str)
        except Exception as e:
            st.session_state["fixes_error"] = str(e)
    url = st.secrets.get("fixes_url")
    return _read(url, "fixes") if url else pd.DataFrame()


def _save_fixes(rows: list) -> str | None:
    ws = _fixes_ws(create=True)
    if ws is None:
        return st.session_state.get("fixes_error") or "No service account or roster_url in Secrets."
    try:
        ws.append_rows(rows, value_input_option="RAW")
        return None
    except Exception as e:
        return str(e)


def _rewrite_fixes(df: pd.DataFrame) -> str | None:
    """Replace the whole fixes tab with `df` (used to edit or undo saved fixes)."""
    ws = _fixes_ws(create=True)
    if ws is None:
        return st.session_state.get("fixes_error") or "No service account or roster_url in Secrets."
    try:
        cols = list(dict.fromkeys(FIX_COLS + [c for c in df.columns if c in FIX_COLS]))
        body = df.reindex(columns=cols).fillna("").astype(str).values.tolist()
        ws.clear()
        ws.update(values=[cols] + body, range_name="A1")
        return None
    except Exception as e:
        return str(e)


# --------------------------------------------------------------------------- data
def _read(url: str, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()          # empty tab (e.g. no name fixes yet): nothing to read, not an error
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
        columns=["form", "date", "code", "team", "role", "person", "key", "submitted_at", "marketer", "form_key"])
    subs = F.apply_fixes(subs, _load_fixes(), roster)
    status["fixes"] = f"{int(subs['fixed_from'].notna().sum())} forms"
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
if status.get("dataentry") != "not connected":
    F.TRACKED_ROLES.add("Data entry")      # data-entry people now have a form: ✓ / ✗ like everyone else
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
roster = F.fix_roster_dates(roster, plan)
subs = subs[subs["date"].isna() | (subs["date"] >= start)]

st.caption("Data sources: " + " · ".join(f"{F.FORM_DEFAULTS[f]['label']}: {status.get(f, '—')}" for f in F.FORM_ORDER)
           + f" · Name fixes: {status.get('fixes', '—')} · Roster: {len(roster)} rows · Planned rides: {len(plan)}")

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
    t, css = F.day_people(day, roster, plan, subs, today)
    if not t.empty:
        rostered = t[~t["Attendance"].isin(["Not on roster", "Not tracked (no form)"])]
        k = st.columns(4)
        k[0].metric("Rostered", len(rostered))
        k[1].metric("Present", int((rostered["Attendance"] == "Present").sum()))
        k[2].metric("Absent / no forms", int((rostered["Attendance"] == "Absent / no forms").sum()))
        k[3].metric("Still missing forms", int(rostered["All forms in"].str.contains("missing").sum()))
    st.markdown("#### People")
    st.caption("Each form: submitted / expected. Expected per day — Recruitment: 1 per ride (E1) · Script Checklist: "
               "1 per ride (E4) · BnB Drug Info: 2 per marketer (E4) · BnB Data Entry: 1 per ride each (E2, E3). "
               "🟢 complete · 🟠 some missing · 🔴 none yet · purple = submitted but not on the roster.")
    show(t, css)

    nc = F.name_checks(day, roster, subs)
    st.markdown("#### Check names")
    if nc.empty:
        st.caption("✓ Every form is filed under someone rostered on that team today.")
    else:
        st.warning(f"{len(nc)} form(s) filed under a name that doesn't match today's roster — usually the wrong "
                   "name was picked on the phone. 'Probably' = who is rostered with that role on that team.")
        names = sorted(set(roster.loc[roster["date"] == pd.Timestamp(day), "name"])) or sorted(set(roster["name"]))
        ed = nc.copy()
        ed.insert(0, "Fix?", False)
        ed.insert(1, "Correct name", ed["Probably"].where(ed["Probably"].isin(names), None))
        out = st.data_editor(
            ed, hide_index=True, width="stretch", key=f"fix_{pd.Timestamp(day).date()}",
            disabled=[c for c in ed.columns if c not in ("Fix?", "Correct name")],
            column_config={
                "Fix?": st.column_config.CheckboxColumn(help="Tick to save this correction"),
                "Correct name": st.column_config.SelectboxColumn(options=names, help="Who really filled in this form"),
                "KEY": st.column_config.TextColumn(width="small")})
        st.caption("Tick **Fix?**, check **Correct name** (pre-filled with 'Probably'), then **Save**. "
                   "Saved to the *fixes* tab of the roster Sheet — only this dashboard uses it; "
                   "the SurveyCTO data is not changed.")
        if st.button("💾 Save name fixes", type="primary"):
            pick = out[out["Fix?"] & out["Correct name"].notna()]
            if pick.empty:
                st.info("Tick at least one row and choose a name.")
            else:
                now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                rows = [[r["KEY"], r["Correct name"], r["Filed under"], r["Form"],
                         str(pd.Timestamp(day).date()), now] for _, r in pick.iterrows()]
                err = _save_fixes(rows)
                if err:
                    st.error(f"Could not save: {err}. Share the roster Sheet with the service account "
                             "(hfc-dashboard@recruitment-hfc-dashboard.iam.gserviceaccount.com) as **Editor**.")
                else:
                    st.success(f"Saved {len(rows)} fix(es).")
                    load_all.clear()
                    st.rerun()

    with st.expander("✏️ Saved name fixes — change or undo"):
        saved = _load_fixes()
        if saved.empty or "KEY" not in saved.columns:
            st.caption("No fixes saved yet.")
        else:
            saved = saved.reindex(columns=FIX_COLS).fillna("")
            all_names = sorted(set(roster["name"]) | set(saved["name"]) - {""})
            sv = saved.copy()
            sv.insert(0, "Undo?", False)
            new_sv = st.data_editor(
                sv, hide_index=True, width="stretch", key="saved_fixes",
                disabled=[c for c in sv.columns if c not in ("Undo?", "name")],
                column_config={"Undo?": st.column_config.CheckboxColumn(help="Tick to delete this fix"),
                               "name": st.column_config.SelectboxColumn("Correct name", options=all_names)})
            st.caption("Change **Correct name**, or tick **Undo?** to remove a fix (the form goes back to the name "
                       "picked on the phone). Then save.")
            if st.button("💾 Save changes to fixes"):
                keep = new_sv[~new_sv["Undo?"]].drop(columns="Undo?")
                err = _rewrite_fixes(keep)
                if err:
                    st.error(f"Could not save: {err}")
                else:
                    st.success(f"Saved — {len(new_sv) - len(keep)} removed, {len(keep)} kept.")
                    load_all.clear()
                    st.rerun()

    st.markdown("#### Rides")
    st.caption("Which per-ride forms came in for each planned ride (matched by batch code).")
    tr, trc = F.day_rides(day, roster, plan, subs)
    show(tr, trc)

    st.markdown("#### Sign-up card data entry")
    if status.get("dataentry") == "not connected":
        st.info("Data Entry (cards) form not connected yet — add `dataentry_url` to Secrets once it is published.")
    else:
        st.caption(f"Each planned ride should have {F.CARDS_PER_RIDE} Data Entry submissions (one per sign-up card), "
                   "matched by batch code — whatever day they were entered. **Entry date check**: cards entered by someone "
                   "who wasn't on the roster for data entry that day.")
        tc, tcc = F.card_entry(plan, subs, day=day, roster=roster)
        show(tc, tcc)
        back, backc = F.card_entry(plan, subs, upto=today, roster=roster)
        if not back.empty:
            open_ = back["Status"] != "✓ complete"
            with st.expander(f"All rides so far: {int(open_.sum())} of {len(back)} not complete"):
                show(back[open_.values].reset_index(drop=True), backc[open_.values].reset_index(drop=True))

with tab_att:
    st.caption("✓ = submitted at least one form that day · ✗ absent = on the roster but no form · "
               "purple = worked but not on the roster. **Field days** = days on the roster as E1–E4 / supervisor "
               "with at least one form. **Data entry days** = days they entered sign-up cards (until the Data Entry "
               "form is connected, past days on the roster for data entry). **Total** = days worked, either kind, "
               "each day counted once.")
    sel = st.slider("Days", 1, len(days), min(14, len(days)), help="How many of the most recent days to show")
    tg, tgc = F.attendance_grid(roster, subs, days[-sel:], today)
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
