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
