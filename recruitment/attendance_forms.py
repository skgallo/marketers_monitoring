"""
forms.py — logic for the Forms and Attendance dashboard
=======================================================
Who was expected to work and submit what each day (roster + planned rides), and what actually
came in from SurveyCTO (each form published to its own Google Sheet tab).

Expected forms (per day):
  Recruitment form       E1          1 per planned ride of their team
  Script Checklist       E4          1 per planned ride of their team
  BnB Drug Information   E4          2 per marketer working with their team that day
  BnB Data Entry         E2 and E3   1 each per planned ride of their team
Supervisors' forms are counted (shown) but not expected.

Roster tab columns (one row per person per day):
  date | team | role | name | code (optional: the person's SurveyCTO choice code)
  role = E1, E2, E3, E4, Supervisor (or anything else: shown, nothing expected)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- configuration
# For each form: which columns hold the date, the planned batch code, who filled it in and their role.
# Every value can be overridden in Secrets, e.g.  [forms.bnb_dataentry]  person_col = "enum_name"
FORM_DEFAULTS = {
    "recruitment": {
        "label": "Recruitment",
        "date_cols": ["a01_r", "SubmissionDate"],
        "batch_cols": ["batch_code_pre", "batch_code"],
        "role_col": "role_type",            # 1 = E1, 0 = supervisor
        "role_map": {"1": "E1", "0": "Supervisor"},
        "person_cols": {"E1": ["a02_1_lab", "e1_name", "a02_1"], "Supervisor": ["supervisor_lab", "supervisor"]},
        "team_cols": ["a02a_r"],
        "marketer_cols": [],
        "exclude": {},
        "code_maps": {"Supervisor": {"1": "Daniella", "2": "Fatu", "3": "Joseph", "4": "Kadie", "5": "Sahr"},
                      "E1": {"11": "Misheal Sellu", "12": "Cecil Massaquoi", "13": "Kadija Esther Kamara",
                             "14": "Daniel T G Sesay", "44": "Fatorma Sheriff"}},
    },
    "script": {
        "label": "Script Checklist",
        "date_cols": ["a01_r", "date", "SubmissionDate"],
        "batch_cols": ["batch_code_pre", "batch_code"],
        "role_col": "role_type",            # 1 = E4, 0 = supervisor
        "role_map": {"1": "E4", "0": "Supervisor"},
        "person_cols": {"E4": ["a02_4_lab", "e4_name"], "Supervisor": ["supervisor"]},
        "team_cols": ["a02a_r"],
        "marketer_cols": [],
        "exclude": {"activity": ["1", "2"]},   # office training / observation: not a field ride
        "code_maps": {"Supervisor": {"1": "Daniella Mansaray", "2": "Kadiatu Nabay", "3": "Sahr Fasuluku",
                                     "4": "Joseph Samuels", "5": "Fatu Conteh"},
                      "E4": {"41": "Aminata Kamara", "42": "Davida A L Yambasu", "43": "Mamusu Bangura",
                             "44": "Fatorma Sheriff", "14": "Daniel T G Sesay"}},
    },
    "bnb_druginfo": {
        "label": "BnB Drug Info",
        "date_cols": ["date", "a01_r", "SubmissionDate"],
        "batch_cols": ["batch_code_pre", "batch_code"],
        "role_col": None,
        "role_map": {},
        "person_cols": {"E4": ["a02_4_lab", "e4_name", "enum_name", "enumerator"]},
        "team_cols": ["a02a_r", "team"],
        "marketer_cols": ["mkt_id", "a03_r", "marketer"],
        "exclude": {},
    },
    "bnb_dataentry": {
        "label": "BnB Data Entry",
        "date_cols": ["a01_r", "date", "SubmissionDate"],
        "batch_cols": ["batch_code_pre", "batch_code"],
        "role_col": "enum_role",            # which enumerator entered it: 2 = E2, 3 = E3
        "role_map": {"2": "E2", "3": "E3", "E2": "E2", "E3": "E3"},
        "person_cols": {"E2": ["a02_2_lab", "enum_name"], "E3": ["a02_3_lab", "enum_name"]},
        "team_cols": ["a02a_r", "team"],
        "marketer_cols": [],
        "exclude": {},
    },
}
FORM_ORDER = ["recruitment", "script", "bnb_druginfo", "bnb_dataentry"]
# who is expected to submit what (per ride unless noted)
EXPECTED = {
    "E1": [("recruitment", "ride", 1)],
    "E2": [("bnb_dataentry", "ride", 1)],
    "E3": [("bnb_dataentry", "ride", 1)],
    "E4": [("script", "ride", 1), ("bnb_druginfo", "marketer", 2)],
}
TEST_USERNAMES = {"testing", "test", "skgallo@uchicago.edu"}
_CODE_RE = re.compile(r"^(\d{8})(\d)(\d+)([A-Z]{4})(\d+)([A-Z])$")


# --------------------------------------------------------------------------- helpers
def norm_name(x) -> str:
    """Name key: letters only, lower case ('Daniel T G Sesay' == 'daniel t.g. sesay')."""
    return re.sub(r"[^a-z]", "", str(x).lower()) if x is not None and str(x) not in ("", "nan", "None") else ""


def _code(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return None
    return s[:-2] if s.endswith(".0") else s


def parse_code(code) -> dict:
    m = _CODE_RE.match(str(code or "").strip().upper())
    if not m:
        return {}
    d, team, mkt, route, ride, trt = m.groups()
    return {"date": pd.to_datetime(d, format="%d%m%Y", errors="coerce"), "team": team, "mkt_id": mkt,
            "route": route, "ride": ride, "treatment": trt}


def _naive_dt(s) -> pd.Series:
    """Parse dates/times from any source into plain (time-zone-free) timestamps, so they compare cleanly.
    Text with a time zone is converted to UTC (Sierra Leone time) first."""
    s = pd.Series(s)
    out = pd.to_datetime(s, format="mixed", errors="coerce", utc=True)
    return out.dt.tz_localize(None)


def _first_col(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns and df[c].notna().any():
            return df[c]
    return pd.Series(None, index=df.index, dtype=object)


def plan_table(batch_codes: pd.DataFrame) -> pd.DataFrame:
    """One row per planned ride: code, date, team, ride, marketer."""
    rows = []
    for c in batch_codes.get("batch_code", pd.Series(dtype=str)).dropna().astype(str):
        p = parse_code(c)
        if p and pd.notna(p["date"]):
            rows.append({"code": c.strip(), **p})
    return pd.DataFrame(rows, columns=["code", "date", "team", "mkt_id", "route", "ride", "treatment"])


def clean_roster(roster: pd.DataFrame) -> pd.DataFrame:
    if roster is None or roster.empty:
        return pd.DataFrame(columns=["date", "team", "role", "name", "code", "key"])
    r = roster.copy()
    r.columns = [str(c).strip().lower() for c in r.columns]
    for c in ["date", "team", "role", "name", "code"]:
        if c not in r.columns:
            r[c] = None
    r["date"] = pd.to_datetime(r["date"], format="mixed", dayfirst=True, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    r["team"] = r["team"].map(lambda x: re.sub(r"\D", "", str(x)) if pd.notna(x) else "")
    def _role(x):
        s = str(x).strip()
        if re.fullmatch(r"[eE]\d", s):
            return s.upper()
        return "Supervisor" if s.lower() in ("supervisor", "sup") else s[:1].upper() + s[1:].lower()
    r["role"] = r["role"].map(_role)
    r["name"] = r["name"].astype(str).str.strip()
    r["code"] = r["code"].map(_code)
    r["key"] = r["name"].map(norm_name)
    return r[r["date"].notna() & (r["key"] != "")].reset_index(drop=True)


def submissions(raw: pd.DataFrame, form: str, cfg: dict, roster: pd.DataFrame) -> pd.DataFrame:
    """One row per submitted form: form, date, code, team, role, person (name) and key."""
    out_cols = ["form", "date", "code", "team", "role", "person", "key", "submitted_at", "marketer"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=out_cols)
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "KEY" in df.columns:
        df = df[~(df["KEY"].notna() & df.duplicated(subset="KEY", keep="last"))]
    if "username" in df.columns:
        df = df[~df["username"].astype(str).str.lower().isin(TEST_USERNAMES)]
    for col, vals in (cfg.get("exclude") or {}).items():
        if col in df.columns:
            df = df[~df[col].map(_code).isin(vals)]
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    code = _first_col(df, cfg["batch_cols"]).map(lambda x: str(x).strip() if pd.notna(x) else None)
    p = code.map(parse_code)
    date = _naive_dt(_first_col(df, cfg["date_cols"]))
    date = date.fillna(p.map(lambda d: d.get("date") if d else pd.NaT)).dt.normalize()
    sub = _naive_dt(df.get("SubmissionDate", pd.Series(None, index=df.index)))
    date = date.fillna(sub.dt.normalize())
    team = _first_col(df, cfg["team_cols"]).map(_code)
    team = team.where(team.notna(), p.map(lambda d: d.get("team") if d else None))
    if cfg.get("role_col") and cfg["role_col"] in df.columns:
        role = df[cfg["role_col"]].map(_code).map(cfg["role_map"])
    else:
        role = pd.Series(next(iter(cfg["person_cols"])), index=df.index)
    # who filled it in: the name column for that role; a code is translated with the roster's code column
    code_to_name = {}
    if roster is not None and not roster.empty:
        code_to_name = {c: n for c, n in zip(roster["code"], roster["name"]) if c}
    person = []
    for i in df.index:
        cols = cfg["person_cols"].get(role[i], []) if pd.notna(role[i]) else []
        val = None
        for c in cols:
            if c in df.columns and pd.notna(df.at[i, c]) and str(df.at[i, c]).strip():
                val = str(df.at[i, c]).strip()
                break
        cmap = (cfg.get("code_maps") or {}).get(role[i], {}) if pd.notna(role[i]) else {}
        if val is not None and _code(val) in cmap:
            val = cmap[_code(val)]
        elif val is not None and _code(val) in code_to_name:
            val = code_to_name[_code(val)]
        if val is None and "username" in df.columns:
            val = str(df.at[i, "username"])
        person.append(val)
    mkt = _first_col(df, cfg.get("marketer_cols") or []).map(_code)
    mkt = mkt.where(mkt.notna(), p.map(lambda d: d.get("mkt_id") if d else None))
    out = pd.DataFrame({"form": form, "date": date, "code": code, "team": team, "role": role,
                        "person": person, "submitted_at": sub, "marketer": mkt}, index=df.index)
    out["key"] = out["person"].map(norm_name)
    # match short names to the roster ('Fatu' → 'Fatu Conteh') when the match is unique
    if roster is not None and not roster.empty:
        rkeys = sorted(set(roster["key"]))
        def to_roster(k):
            if not k or k in rkeys:
                return k
            hits = [rk for rk in rkeys if rk.startswith(k) or k.startswith(rk)]
            return hits[0] if len(hits) == 1 else k
        out["key"] = out["key"].map(to_roster)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- views
def expected_for(person_row, plan_day: pd.DataFrame) -> dict:
    """{form: expected count} for one rostered person on one day."""
    rides = plan_day[plan_day["team"] == person_row["team"]] if person_row["team"] else plan_day.iloc[0:0]
    exp = {}
    for form, unit, n in EXPECTED.get(person_row["role"], []):
        if unit == "ride":
            exp[form] = exp.get(form, 0) + n * len(rides)
        elif unit == "marketer":
            exp[form] = exp.get(form, 0) + n * rides["mkt_id"].nunique()
    return exp


def day_people(day, roster: pd.DataFrame, plan: pd.DataFrame, subs: pd.DataFrame):
    """One row per person on `day`: attendance and submitted / expected for each form. Returns (table, css)."""
    day = pd.Timestamp(day).normalize()
    r_day = roster[roster["date"] == day]
    p_day = plan[plan["date"] == day]
    s_day = subs[subs["date"] == day]
    rows, sty = [], []
    seen = set()
    for _, pr in r_day.sort_values(["team", "role", "name"]).iterrows():
        mine = s_day[s_day["key"] == pr["key"]]
        exp = expected_for(pr, p_day)
        row = {"Team": pr["team"] or "—", "Role": pr["role"], "Name": pr["name"]}
        s = {k: "" for k in row}
        total_exp, total_got = 0, 0
        for f in FORM_ORDER:
            lbl = FORM_DEFAULTS[f]["label"]
            got = int((mine["form"] == f).sum())
            e = exp.get(f)
            if e:
                row[lbl] = f"{got}/{e}"
                total_exp += e; total_got += min(got, e)
                s[lbl] = "color:#1E7B34;font-weight:600" if got >= e else (
                    "color:#CC0000;font-weight:600" if got == 0 else "color:#B26B00;font-weight:600")
            else:
                row[lbl] = str(got) if got else "—"
                s[lbl] = "" if got else "color:#888888"
        worked = len(mine) > 0
        row["Attendance"] = "Present" if worked else "Absent / no forms"
        s["Attendance"] = "color:#1E7B34;font-weight:600" if worked else "color:#CC0000;font-weight:600"
        last = mine["submitted_at"].max()
        row["Last form"] = last.strftime("%H:%M") if pd.notna(last) else "—"
        row["All forms in"] = ("✓" if total_got >= total_exp else f"{total_exp - total_got} missing") if total_exp else "—"
        s["All forms in"] = ("color:#1E7B34;font-weight:600" if total_got >= total_exp
                             else "color:#CC0000;font-weight:600") if total_exp else "color:#888888"
        s["Last form"] = ""
        rows.append(row); sty.append(s); seen.add(pr["key"])
    # people who submitted but aren't on the roster
    for k, g in s_day[~s_day["key"].isin(seen) & (s_day["key"] != "")].groupby("key"):
        row = {"Team": ", ".join(sorted(set(x for x in g["team"] if x))) or "—",
               "Role": ", ".join(sorted(set(str(x) for x in g["role"] if pd.notna(x)))) or "—",
               "Name": g["person"].iloc[0]}
        s = {k_: "color:#7B2CBF" for k_ in row}
        for f in FORM_ORDER:
            got = int((g["form"] == f).sum())
            row[FORM_DEFAULTS[f]["label"]] = str(got) if got else "—"
            s[FORM_DEFAULTS[f]["label"]] = ""
        row["Attendance"] = "Not on roster"; s["Attendance"] = "color:#7B2CBF;font-weight:600"
        last = g["submitted_at"].max()
        row["Last form"] = last.strftime("%H:%M") if pd.notna(last) else "—"; s["Last form"] = ""
        row["All forms in"] = "—"; s["All forms in"] = "color:#888888"
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows)
    return t, pd.DataFrame(sty).reindex(columns=t.columns).fillna("")


def day_rides(day, roster: pd.DataFrame, plan: pd.DataFrame, subs: pd.DataFrame):
    """One row per planned ride on `day`: which of the per-ride forms came in. Returns (table, css)."""
    day = pd.Timestamp(day).normalize()
    p_day = plan[plan["date"] == day].copy()
    if p_day.empty:
        return pd.DataFrame(), pd.DataFrame()
    s_day = subs[subs["date"] == day]
    p_day["_r"] = pd.to_numeric(p_day["ride"], errors="coerce")
    rows, sty = [], []
    for _, pr in p_day.sort_values(["team", "_r"]).iterrows():
        on = s_day[s_day["code"] == pr["code"]]
        def cell(form, role=None):
            m = on[on["form"] == form]
            if role:
                m = m[m["role"] == role]
            if m.empty:
                return "✗", "color:#CC0000;font-weight:600"
            names = ", ".join(dict.fromkeys(str(x) for x in m["person"] if x))
            return f"✓ {names}" + (f" (×{len(m)})" if len(m) > 1 else ""), "color:#1E7B34;font-weight:600"
        row = {"Team": pr["team"], "Ride": pr["ride"], "Planned code": pr["code"], "Marketer": pr["mkt_id"]}
        s = {k: "" for k in row}
        for lbl, (f, role) in {"Recruitment (E1)": ("recruitment", "E1"), "Script Checklist (E4)": ("script", "E4"),
                               "Data Entry (E2)": ("bnb_dataentry", "E2"), "Data Entry (E3)": ("bnb_dataentry", "E3")}.items():
            row[lbl], s[lbl] = cell(f, role)
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows)
    return t, pd.DataFrame(sty).reindex(columns=t.columns).fillna("")


def attendance_grid(roster: pd.DataFrame, subs: pd.DataFrame, days: list):
    """Person × day: Present (submitted something) / Absent (rostered, nothing) / — (not rostered).
    Returns (table, css)."""
    people = roster.drop_duplicates("key")[["key", "name"]]
    extra = subs[~subs["key"].isin(people["key"]) & (subs["key"] != "")].drop_duplicates("key")[["key", "person"]]
    people = pd.concat([people, extra.rename(columns={"person": "name"})], ignore_index=True).sort_values("name")
    rows, sty = [], []
    for _, p in people.iterrows():
        row, s = {"Name": p["name"]}, {"Name": ""}
        n_present = n_rostered = 0
        for d in days:
            col = pd.Timestamp(d).strftime("%a %d %b")
            rostered = ((roster["key"] == p["key"]) & (roster["date"] == d)).any()
            worked = ((subs["key"] == p["key"]) & (subs["date"] == d)).any()
            n_rostered += rostered; n_present += worked and rostered
            if worked:
                row[col] = "✓" if rostered else "✓ (not rostered)"
                s[col] = "background-color:#D9F2E6" if rostered else "background-color:#EEEDFE"
            elif rostered:
                row[col], s[col] = "✗ absent", "background-color:#FADBD3"
            else:
                row[col], s[col] = "", ""
        row["Days worked / rostered"] = f"{n_present}/{n_rostered}" if n_rostered else "—"
        s["Days worked / rostered"] = ""
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows)
    return t, pd.DataFrame(sty).reindex(columns=t.columns).fillna("")
