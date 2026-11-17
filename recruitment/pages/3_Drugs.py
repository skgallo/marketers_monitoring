"""
Drugs — third page of the Recruitment HFC app
=============================================
1. What drugs the E4 recorded in the morning (BnB Drug Info, check-in 1)
2. Whether the bag has at least one of each core drug: Pain, Deworming, Cold, Infection
3. With the Script Checklist: the 4 core drugs were pitched on each ride, and the order of the drugs
   (flag if it changes from the marketer's first ride of the day)

Uses the app's Secrets: bnb_druginfo_url, e4_url (or script_url), batch_codes_url.
"""

import datetime as dt
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attendance_forms as AF
import drugs as DR
from auth import require_password

st.set_page_config(page_title="Drugs", page_icon="💊", layout="wide")
require_password()


def _read(url, name):
    try:
        df = pd.read_csv(url, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not read **{name}**: {e}.")
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df.replace({"": None})


@st.cache_data(ttl=300)
def load():
    g = lambda *ks: next((st.secrets.get(k) for k in ks if st.secrets.get(k)), None)
    di_url, sc_url, bc_url = g("bnb_druginfo_url"), g("script_url", "e4_url"), g("batch_codes_url")
    F, D = DR.drug_info(_read(di_url, "BnB Drug Info")) if di_url else (pd.DataFrame(), pd.DataFrame())
    R = DR.ride_drugs(_read(sc_url, "Script Checklist")) if sc_url else pd.DataFrame()
    plan = AF.plan_table(_read(bc_url, "batch codes")) if bc_url else pd.DataFrame()
    status = {"Drug Info": f"{len(F)} forms" if di_url else "not connected",
              "Script Checklist": f"{len(R)} rides" if sc_url else "not connected"}
    return F, D, R, plan, status


ALIAS_COLS = ["written", "same_as", "saved_at"]


def _alias_ws(create=False):
    """'drug_aliases' tab in the roster Sheet (service account), or None."""
    import re
    try:
        import gspread
        from data_io import _service_account_info
        info = _service_account_info()
        m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", str(st.secrets.get("roster_url", "")))
        if not info or not m:
            return None
        sh = gspread.service_account_from_dict(info).open_by_key(m.group(1))
        try:
            ws = sh.worksheet("drug_aliases")
        except gspread.WorksheetNotFound:
            if not create:
                return None
            ws = sh.add_worksheet("drug_aliases", rows=500, cols=len(ALIAS_COLS))
        if not ws.row_values(1):
            ws.update(values=[ALIAS_COLS], range_name="A1")
        return ws
    except Exception as e:
        st.session_state["alias_error"] = str(e)
        return None


@st.cache_data(ttl=300)
def load_aliases() -> pd.DataFrame:
    ws = _alias_ws()
    if ws is not None:
        try:
            return pd.DataFrame(ws.get_all_records(), dtype=str)
        except Exception:
            pass
    url = st.secrets.get("drug_aliases_url")
    return _read(url, "drug aliases") if url else pd.DataFrame()


def _write_aliases(df: pd.DataFrame | None = None, append: list | None = None):
    ws = _alias_ws(create=True)
    if ws is None:
        return (st.session_state.get("alias_error") or "No service account or roster_url in Secrets.") + \
            " Share the roster Sheet with hfc-dashboard@recruitment-hfc-dashboard.iam.gserviceaccount.com as Editor."
    try:
        if append:
            ws.append_rows(append, value_input_option="RAW")
        else:
            ws.clear()
            ws.update(values=[ALIAS_COLS] + df.reindex(columns=ALIAS_COLS).fillna("").astype(str).values.tolist(),
                      range_name="A1")
        return None
    except Exception as e:
        return str(e)


def show(t, css):
    if t is None or t.empty:
        st.info("Nothing to show.")
        return
    st.dataframe(t.style.apply(lambda _: css, axis=None), hide_index=True, width="stretch")


c1, c2 = st.columns([6, 1])
c1.title("💊 Drugs")
if c2.button("🔄 Refresh"):
    load.clear()
    st.rerun()

F, D, R, plan, status = load()
aliases = load_aliases()
DR.set_aliases(aliases)            # 'Ronamox' = 'Ronamox 500' etc., saved below
for df in (F, D, R, plan):
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
st.caption(" · ".join(f"{k}: {v}" for k, v in status.items()))

try:
    start = pd.Timestamp(st.secrets.get("monitor_from", "2026-11-12"))
except Exception:
    start = pd.Timestamp("2026-11-12")
days = sorted({d for df in (F, R, plan) if not df.empty for d in df["date"].dropna()} )
days = [d for d in days if d >= start]
if not days:
    st.info("No Drug Info or Script Checklist forms yet.")
    st.stop()
today = pd.Timestamp(dt.date.today())
past = [i for i, d in enumerate(days) if d <= today]
day = st.selectbox("Day", days, index=(days.index(today) if today in days else (past[-1] if past else 0)),
                   format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y") + (" (today)" if d == today else ""))

# ---- 1 + 2
st.markdown("### 1. Morning drug list and core drugs")
st.caption("From the BnB Drug Info form (check-in 1 = morning). Each drug shows its category. "
           "Core drugs: at least one **Pain**, **Deworming**, **Cold** and **Infection** drug in the bag. "
           "Check-in 2 shows drugs added later in the day.")
mt, mtc = DR.morning_table(F, D, day, plan)
if not mt.empty:
    k = st.columns(3)
    k[0].metric("Marketers", len(mt))
    k[1].metric("Morning form in", int((~mt["Morning form"].str.startswith("✗")).sum()))
    k[2].metric("All 4 core drugs", int((mt["4 core drugs"] == "✓").sum()))
show(mt, mtc)

# ---- 3
st.markdown("### 2. Drugs pitched on each ride (Script Checklist)")
st.caption("The drugs the E4 recorded, in the order pitched (a bundle counts as all its drugs). "
           "**4 core drugs pitched**: at least one of each core category on the ride. "
           "**Order**: compared with the marketer's first ride of the day — red if the order changes, "
           "amber if a drug was dropped or added. **Not in the morning list**: pitched but not in Drug Info.")
rt, rtc = DR.ride_table(R, F, D, day, plan)
if not rt.empty:
    checked = rt[~rt["Pitch"].eq("Sit down") & ~rt["Drugs pitched (in order)"].eq("no Script Checklist yet")]
    k = st.columns(3)
    k[0].metric("Rides with a checklist", len(checked))
    k[1].metric("All 4 core drugs pitched", int((checked["4 core drugs pitched"] == "✓").sum()))
    k[2].metric("Order changed", int(checked["Order"].str.contains("order changed").sum()))
show(rt, rtc)

with st.expander("✏️ Drug names that don't match the morning list — mark as the same drug"):
    st.caption("Names written differently on the Script Checklist ('Ronamox' vs 'Ronamox 500'). Names that clearly "
               "match one drug in the marketer's bag are matched automatically; pick the rest here. Saved to the "
               "*drug_aliases* tab of the roster Sheet and used on every day.")
    um = DR.unmatched_names(R, F, D, day)
    if um.empty:
        st.caption("✓ Every drug pitched matches the morning list.")
    else:
        opts = sorted({o for lst in um["options"] for o in lst})
        ed = um.drop(columns="options").copy()
        ed.insert(0, "Same as", None)
        out = st.data_editor(ed, hide_index=True, width="stretch", key=f"alias_{pd.Timestamp(day).date()}",
                             disabled=[c for c in ed.columns if c != "Same as"],
                             column_config={"Same as": st.column_config.SelectboxColumn(
                                 options=opts, help="The drug from the morning list this really is (leave empty if it "
                                                    "really wasn't in the bag)")})
        if st.button("💾 Save drug names", type="primary"):
            pick = out[out["Same as"].notna()]
            if pick.empty:
                st.info("Choose 'Same as' for at least one row.")
            else:
                now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                err = _write_aliases(append=[[w, s, now] for w, s in zip(pick["Written on the checklist"], pick["Same as"])])
                if err:
                    st.error(f"Could not save: {err}")
                else:
                    load_aliases.clear(); st.rerun()
    if not aliases.empty and {"written", "same_as"} <= set(aliases.columns):
        st.markdown("**Saved — tick Undo? to remove**")
        sv = aliases.reindex(columns=ALIAS_COLS).fillna("")
        sv.insert(0, "Undo?", False)
        new_sv = st.data_editor(sv, hide_index=True, width="stretch", key="alias_saved",
                                disabled=ALIAS_COLS)
        if st.button("💾 Save (remove ticked)", key="alias_undo"):
            err = _write_aliases(df=aliases.reindex(columns=ALIAS_COLS)[~new_sv["Undo?"].values])
            if err:
                st.error(f"Could not save: {err}")
            else:
                load_aliases.clear(); st.rerun()

st.markdown("#### By marketer")
sm = DR.summary_table(rt)
if sm.empty:
    st.info("Nothing to show.")
else:
    st.dataframe(sm, hide_index=True, width="stretch")

with st.expander("All drugs recorded (Drug Info), with category and unit"):
    if D.empty:
        st.info("No Drug Info yet.")
    else:
        dd = D[D["date"] == day][["team", "mkt", "sub_no", "e4", "n", "drug", "category", "unit"]].rename(columns={
            "team": "Team", "mkt": "Marketer", "sub_no": "Check-in", "e4": "E4", "n": "#", "drug": "Drug",
            "category": "Category", "unit": "Unit"})
        st.dataframe(dd, hide_index=True, width="stretch")
        st.download_button("⬇ Download all drugs (CSV)",
                           D.drop(columns=["key"]).to_csv(index=False).encode("utf-8"), "drugs.csv", "text/csv")
    st.caption("Category codes used: " + ", ".join(f"{k} = {v}" for k, v in DR.CAT_LABEL.items())
               + ". Tell the research team if a code is wrong.")
