"""
drugs.py — Drug checks for the Recruitment HFC app (page 3 "Drugs")
====================================================================
1. Morning drug list: what the E4 recorded in the BnB Drug Info form (submission 1 = morning).
2. Core drugs: at least one Pain, Deworming, Cold and Infection drug in the marketer's bag.
3. Rides (Script Checklist): the drugs actually pitched on each ride — all 4 core drugs? same order
   as the marketer's first ride of the day?
"""

import re

import numpy as np
import pandas as pd

import attendance_forms as AF

CORE = ["pain", "dew", "cold", "anti"]
CORE_LABEL = {"pain": "Pain", "dew": "Deworming", "cold": "Cold", "anti": "Infection"}
# drug_cat codes in the BnB Drug Info form (choice list "drugcat")
CAT_LABEL = {"1": "Pain Medicine", "2": "Antibiotic", "3": "Antifungal", "4": "Deworming", "5": "Vitamins",
             "6": "Candy", "7": "Cold Medicine", "8": "Antimalarial", "9": "Ulcer or anti acid", "10": "Perfume",
             "-55": "Other"}
# core types: Infection = antibiotic or antifungal (as in the form's core_anti)
CAT_CORE = {"1": "pain", "2": "anti", "3": "anti", "4": "dew", "7": "cold"}
TREATMENT = {"N": "Normal", "P": "Peddling", "S": "Sit down"}
TEST_USERNAMES = AF.TEST_USERNAMES
E4_NAMES = AF.FORM_DEFAULTS["bnb_druginfo"]["code_maps"]["E4"]

GREEN, RED, AMBER, GREY = ("color:#1E7B34;font-weight:600", "color:#CC0000;font-weight:600",
                           "color:#B26B00;font-weight:600", "color:#888888")


ALIASES = {}   # {norm(written name): norm(real name)} — set from the drug_aliases tab by the page


def norm(x) -> str:
    """Drug-name key for matching: lower case, letters and digits only ('Metromax ' == 'metromax');
    names saved as the same drug (drug_aliases) map to one key."""
    k = re.sub(r"[^a-z0-9]", "", str(x).lower()) if x is not None and str(x) not in ("nan", "None") else ""
    return ALIASES.get(k, k)


def set_aliases(df: pd.DataFrame):
    """drug_aliases tab: columns 'written' and 'same_as'."""
    ALIASES.clear()
    if df is None or df.empty or not {"written", "same_as"} <= set(df.columns):
        return
    for w, s in zip(df["written"], df["same_as"]):
        kw = re.sub(r"[^a-z0-9]", "", str(w).lower())
        ks = re.sub(r"[^a-z0-9]", "", str(s).lower())
        if kw and ks and kw != ks:
            ALIASES[kw] = ks


def resolve(name, bag_keys) -> str:
    """Key of `name`, matched to the marketer's bag when one bag drug clearly is the same
    ('Ronamox' → 'Ronamox 500', 'HTC para' → 'HTC-PARA 500')."""
    k = norm(name)
    if not k or k in bag_keys:
        return k
    hits = [b for b in bag_keys if b.startswith(k) or k.startswith(b)]
    return hits[0] if len(hits) == 1 else k


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    # group-path names (a/b/drug_name, rep[2]/drug_name) → drug_name / drug_name_2
    ren = {}
    for c in df.columns:
        s = str(c).strip()
        if "/" in s or "[" in s:
            last = s.split("/")[-1]
            k = re.findall(r"\[(\d+)\]", s)
            base = re.sub(r"\[\d+\]$", "", last)
            new = f"{base}_{k[-1]}" if k and not re.search(r"_\d+$", base) else base
            if new not in df.columns and new not in ren.values():
                ren[c] = new
    df = df.rename(columns=ren)
    if "KEY" in df.columns:
        df = df[~(df["KEY"].notna() & df.duplicated(subset="KEY", keep="last"))]
    if "username" in df.columns:
        df = df[~df["username"].astype(str).str.lower().isin(TEST_USERNAMES)]
    return df.reset_index(drop=True)


def _split_list(s) -> list:
    """'ID1 all cold; ID2 xipain' → ['all cold', 'xipain']."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    out = []
    for part in str(s).split(";"):
        p = re.sub(r"^\s*ID\d+\s*", "", part).strip()
        if p:
            out.append(p)
    return out


# --------------------------------------------------------------------------- 1–2. Drug Info
def drug_info(raw: pd.DataFrame) -> tuple:
    """(one row per Drug Info form, one row per drug). Category from drug_cat_N; for check-in 2 (no new
    drugs) the categories are taken from the same marketer's morning form."""
    df = _clean(raw)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    date = AF._naive_dt(AF._first_col(df, ["date_key", "bc_date_key", "today"])).dt.normalize()
    sub = AF._naive_dt(df.get("SubmissionDate", pd.Series(None, index=df.index)))
    first = AF._first_col(df, ["bc_first", "batch_codes_today"]).map(
        lambda x: str(x).split()[0] if isinstance(x, str) and x.split() else None)
    team = first.map(lambda c: AF.parse_code(c).get("team") if c else None)
    mkt = AF._first_col(df, ["bc_mkt_id", "a03_r"]).map(AF._code)
    forms, drugs = [], []
    for i, r in df.iterrows():
        enum = AF._code(r.get("enumerator"))
        name = E4_NAMES.get(enum, r.get("enum_oth") if enum == "-55" else enum)
        no = AF._code(r.get("submission_no")) or "1"
        lst = _split_list(r.get("drugid_total"))
        f = {"date": date[i], "team": team[i], "mkt": mkt[i], "sub_no": no, "e4": name,
             "submitted_at": sub[i], "key": r.get("KEY"), "drugs": lst,
             "new_drugs": AF._code(r.get("new_drugs_yn"))}
        for c in CORE:
            f[c] = AF._code(r.get(f"core_{c}"))
        forms.append(f)
        for n, d in enumerate(lst, 1):
            cat = AF._code(r.get(f"drug_cat_{n}"))
            if cat == "-55":
                other = r.get(f"drug_cat_oth_{n}")
            else:
                other = None
            unit = r.get(f"unit_{n}")
            drugs.append({"date": date[i], "team": team[i], "mkt": mkt[i], "sub_no": no, "e4": name, "n": n,
                          "drug": d, "cat": cat, "cat_other": other if isinstance(other, str) else None,
                          "unit": unit if isinstance(unit, str) else None, "key": r.get("KEY")})
    F = pd.DataFrame(forms)
    D = pd.DataFrame(drugs)
    if not D.empty:
        # categories missing (check-in 2) → same drug, same marketer, same day, from another form
        kd = D.dropna(subset=["cat"])
        known = {(d, m, norm(x)): c for d, m, x, c in zip(kd["date"], kd["mkt"], kd["drug"], kd["cat"])}
        k = list(zip(D["date"], D["mkt"], D["drug"].map(norm)))
        D["cat"] = [c if c else known.get(kk) for c, kk in zip(D["cat"], k)]
        oth = {(d, m, norm(x)): o for d, m, x, o in zip(kd["date"], kd["mkt"], kd["drug"], kd["cat_other"]) if o}
        D["cat_other"] = [o if o else oth.get(kk) for o, kk in zip(D["cat_other"], k)]
        D["category"] = [CAT_LABEL.get(c, f"Category {c}") if c else "—" for c in D["cat"]]
        D.loc[D["cat"] == "-55", "category"] = "Other" + D.loc[D["cat"] == "-55", "cat_other"].map(
            lambda x: f" ({x})" if x else "")
    return F, D


def bag_for(F: pd.DataFrame, D: pd.DataFrame, date, mkt) -> list:
    """The marketer's drug list for the day (latest check-in, as the Script Checklist uses)."""
    if F.empty:
        return []
    f = F[(F["date"] == date) & (F["mkt"] == mkt)].sort_values(["sub_no", "submitted_at"])
    return list(f.iloc[-1]["drugs"]) if not f.empty else []


def cat_of(D: pd.DataFrame, date, mkt) -> dict:
    """{drug key: category code} for one marketer-day (any day as fallback)."""
    if D.empty:
        return {}
    d = D.dropna(subset=["cat"])
    same = d[(d["date"] == date) & (d["mkt"] == mkt)]
    out = {norm(x): c for x, c in zip(d["drug"], d["cat"])}       # any day
    out.update({norm(x): c for x, c in zip(same["drug"], same["cat"])})
    return out


def morning_table(F: pd.DataFrame, D: pd.DataFrame, day, plan: pd.DataFrame | None = None) -> tuple:
    """One row per marketer working on `day`: morning drug list and the 4 core drugs. Returns (table, css)."""
    day = pd.Timestamp(day).normalize()
    f = F[F["date"] == day] if not F.empty else F
    mkts = set(f["mkt"]) if not f.empty else set()
    if plan is not None and not plan.empty:
        mkts |= set(plan.loc[plan["date"] == day, "mkt_id"])
    rows, sty = [], []
    for m in sorted(x for x in mkts if x):
        fm = f[f["mkt"] == m] if not f.empty else f
        morning = fm[fm["sub_no"] == "1"].sort_values("submitted_at") if not fm.empty else fm
        second = fm[fm["sub_no"] == "2"].sort_values("submitted_at") if not fm.empty else fm
        team = (fm["team"].dropna().iloc[0] if not fm.empty and fm["team"].notna().any() else
                (plan.loc[(plan["date"] == day) & (plan["mkt_id"] == m), "team"].iloc[0]
                 if plan is not None and not plan.empty and ((plan["date"] == day) & (plan["mkt_id"] == m)).any() else "—"))
        row = {"Team": team, "Marketer": m}
        s = {"Team": "", "Marketer": ""}
        if morning.empty:
            row.update({"E4": "—", "Morning form": "✗ not submitted", "Drugs in the bag": "—", "N": "—"})
            s.update({"E4": "", "Morning form": RED, "Drugs in the bag": GREY, "N": GREY})
            for c in CORE:
                row[CORE_LABEL[c]], s[CORE_LABEL[c]] = "—", GREY
            row["4 core drugs"], s["4 core drugs"] = "—", GREY
        else:
            mr = morning.iloc[-1]
            dd = D[(D["key"] == mr["key"])] if not D.empty else D
            cats = {norm(x): c for x, c in zip(dd["drug"], dd["cat"])} if not dd.empty else {}
            lst = [f"{x} ({CAT_LABEL.get(cats.get(norm(x)), 'no category')})" for x in mr["drugs"]]
            row.update({"E4": mr["e4"], "Morning form": mr["submitted_at"].strftime("%H:%M") if pd.notna(mr["submitted_at"]) else "✓",
                        "Drugs in the bag": " · ".join(lst) or "—", "N": str(len(mr["drugs"]))})
            s.update({"E4": "", "Morning form": GREEN, "Drugs in the bag": "", "N": ""})
            have = {c: (mr.get(c) == "1") or any(CAT_CORE.get(cats.get(norm(x))) == c for x in mr["drugs"]) for c in CORE}
            for c in CORE:
                row[CORE_LABEL[c]], s[CORE_LABEL[c]] = ("✓", GREEN) if have[c] else ("✗", RED)
            miss = [CORE_LABEL[c] for c in CORE if not have[c]]
            row["4 core drugs"], s["4 core drugs"] = ("✓", GREEN) if not miss else ("Missing: " + ", ".join(miss), RED)
            if len(morning) > 1:
                row["Morning form"] += f" (×{len(morning)})"
                s["Morning form"] = AMBER
        if second.empty:
            row["Check-in 2"], s["Check-in 2"] = "not yet", GREY
        else:
            sr = second.iloc[-1]
            new = [x for x in sr["drugs"] if norm(x) not in {norm(y) for y in (morning.iloc[-1]["drugs"] if not morning.empty else [])}]
            row["Check-in 2"] = (sr["submitted_at"].strftime("%H:%M") if pd.notna(sr["submitted_at"]) else "✓") + \
                (f" · new: {', '.join(new)}" if new else " · no new drugs")
            s["Check-in 2"] = AMBER if new else GREEN
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows)
    return t, pd.DataFrame(sty).reindex(columns=t.columns).fillna("")


# --------------------------------------------------------------------------- 3. Script Checklist
def _repeat_values(r: pd.Series, field: str) -> list:
    """Values of a repeat field published wide: field_1, field_2, … (in order)."""
    vals = []
    for c in r.index:
        m = re.fullmatch(rf"{field}_(\d+)", str(c))
        if m and pd.notna(r[c]) and str(r[c]).strip():
            vals.append((int(m.group(1)), str(r[c]).strip()))
    if not vals and field in r.index and pd.notna(r[field]) and str(r[field]).strip():
        vals = [(1, str(r[field]).strip())]
    return [v for _, v in sorted(vals)]


MENTIONS = {"name": "Name of the drug", "condition": "Condition / sickness", "dosage": "Dosage",
            "dosage_x": "Dosage (added claims)", "sym_basic": "Symptoms", "sym_narr": "Symptoms (narratives)",
            "predisp": "Predisposing factors", "complications": "Complications", "price": "Price",
            "price_x": "Price (bargaining / promotions)", "work": "Work of the drug", "composition": "Composition",
            "composition_x": "Composition (appeals)", "contra": "Contraindications",
            "contra_x": "Contraindications (claims)", "manufact": "Manufacturer", "manufact_x": "Manufacturer (appeals)",
            "expiry": "Expiry date", "expiry_x": "Expiry (reassurance)", "sidefx": "Side effects",
            "fear": "Fear-based appeals", "hope": "Hope and promises", "scarcity": "Scarcity / call to action",
            "destigma": "Destigmatising", "authority": "Authority / pharmacy", "visual": "Visual aids",
            "testimony": "Testimonies", "local_terms": "Local sayings", "tactile": "Tactile / taste",
            "proof": "Proof of correctness", "icebreaker": "Icebreaker", "mimic": "Mimicking", "teaching": "Teaching",
            "nosidefx": "No side effects", "referral": "Referral", "warranty": "Warranty"}
ASKED_WHO = {"1": "Marketer", "2": "E1", "3": "E2", "4": "E3", "5": "E4", "6": "Supervisor",
             "7": "Another passenger", "-55": "Other"}


def _repeat_map(r: pd.Series, field: str) -> dict:
    """{instance number: value} of a repeat field published wide (field_1, field_2, …)."""
    out = {}
    for c in r.index:
        m = re.fullmatch(rf"{field}_(\d+)", str(c))
        if m and pd.notna(r[c]) and str(r[c]).strip():
            out[int(m.group(1))] = str(r[c]).strip()
    if not out and field in r.index and pd.notna(r[field]) and str(r[field]).strip():
        out[1] = str(r[field]).strip()
    return out


def ride_drugs(script_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per ride (Script Checklist, E4 preferred over supervisor): planned code, ride info and
    the drugs pitched in order (a bundle counts as all its drugs, in the same position)."""
    df = _clean(script_raw)
    if df.empty:
        return pd.DataFrame()
    if "activity" in df.columns:
        act = pd.to_numeric(df["activity"], errors="coerce")
        df = df[act.isna() | (act > 2)]
    code = AF._first_col(df, ["batch_code_pre", "batch_code"]).map(lambda x: str(x).strip() if pd.notna(x) else None)
    rows = []
    for i, r in df.iterrows():
        p = AF.parse_code(code[i])
        if not p:
            continue
        names = _repeat_map(r, "drug_name")
        drugs = [names[k] for k in sorted(names)]
        ran_out = _repeat_map(r, "f12"); asked = _repeat_map(r, "f13")
        qs = _repeat_map(r, "f13b"); mentions = _repeat_map(r, "question_mentions")
        stockout = [names.get(k, f"drug {k}") for k, v_ in sorted(ran_out.items()) if AF._code(v_) == "1"]
        pitch_q = []
        for k, v_ in sorted(asked.items()):
            if AF._code(v_) == "1":
                men = ", ".join(MENTIONS.get(m, m) for m in mentions.get(k, "").split())
                pitch_q.append(f"{names.get(k, f'drug {k}')}: {qs.get(k, '(no text)')}"
                               + (f" [marketer mentioned: {men}]" if men else ""))
        who = [ASKED_WHO.get(c, c) for c in str(r.get("sd_pause_who") or "").split()]
        if "Other" in who and isinstance(r.get("sd_pause_who_oth"), str):
            who = [w if w != "Other" else f"Other ({r['sd_pause_who_oth'].strip()})" for w in who]
        rows.append({"code": code[i], "date": p["date"], "team": p["team"], "mkt": p["mkt_id"],
                     "ride": p["ride"], "treatment": p["treatment"],
                     "role": "E4" if AF._code(r.get("role_type")) != "0" else "Supervisor",
                     "submitted_at": AF._naive_dt(pd.Series([r.get("SubmissionDate")])).iloc[0],
                     "pitched": drugs, "key": r.get("KEY"),
                     "stockout": stockout, "pitch_questions": pitch_q,
                     "pause_q": AF._code(r.get("sd_pause_q")), "pause_who": who,
                     "pause_what": str(r.get("sd_pause_what")).strip() if isinstance(r.get("sd_pause_what"), str) else "",
                     "comment_yn": AF._code(r.get("f14")),
                     "comment": str(r.get("f15")).strip() if isinstance(r.get("f15"), str) else ""})
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    R["_pref"] = (R["role"] == "E4").astype(int)
    R = R.sort_values(["_pref", "submitted_at"]).drop_duplicates("code", keep="last").drop(columns="_pref")
    return R.reset_index(drop=True)


def _items(pitched: list) -> list:
    """Bundles 'A + B' → each drug separately (same position)."""
    out = []
    for p in pitched:
        out += [x.strip() for x in str(p).split(" + ") if x.strip()]
    return out


def ride_table(R: pd.DataFrame, F: pd.DataFrame, D: pd.DataFrame, day, plan: pd.DataFrame | None = None) -> tuple:
    """One row per ride on `day`: drugs pitched (order), 4 core drugs pitched, order vs the marketer's first ride."""
    day = pd.Timestamp(day).normalize()
    r_day = R[R["date"] == day].copy() if not R.empty else pd.DataFrame(columns=["code", "date", "team", "mkt", "ride", "treatment", "pitched"])
    if plan is not None and not plan.empty:
        missing = plan[(plan["date"] == day) & ~plan["code"].isin(r_day["code"])]
        if not missing.empty:
            r_day = pd.concat([r_day, pd.DataFrame({"code": missing["code"], "date": day, "team": missing["team"],
                                                     "mkt": missing["mkt_id"], "ride": missing["ride"],
                                                     "treatment": missing["treatment"], "pitched": None})],
                              ignore_index=True)
    if r_day.empty:
        return pd.DataFrame(), pd.DataFrame()
    r_day["_r"] = pd.to_numeric(r_day["ride"], errors="coerce")
    r_day = r_day.sort_values(["team", "mkt", "_r"])
    first_order = {}
    rows, sty = [], []
    for _, x in r_day.iterrows():
        row = {"Team": x["team"], "Marketer": x["mkt"], "Ride": x["ride"],
               "Pitch": TREATMENT.get(x["treatment"], x["treatment"])}
        s = {k: "" for k in row}
        if not isinstance(x["pitched"], list):
            row.update({"Drugs pitched (in order)": "no Script Checklist yet", "N": "—"})
            for c in CORE:
                row[CORE_LABEL[c]] = "—"
            row.update({"4 core drugs pitched": "—", "Order": "—", "Not in the morning list": "—"})
            s = {k: GREY for k in row}
            s["Drugs pitched (in order)"] = AMBER
            rows.append(row); sty.append(s)
            continue
        if x["treatment"] == "S":
            row.update({"Drugs pitched (in order)": " → ".join(x["pitched"]) or "none (Sit down)",
                        "N": str(len(_items(x["pitched"])))})
            for c in CORE:
                row[CORE_LABEL[c]] = "—"
            row.update({"4 core drugs pitched": "n/a (Sit down)", "Order": "n/a",
                        "Not in the morning list": "—"})
            s = {k: GREY for k in row}
            if x["pitched"]:
                row["4 core drugs pitched"], s["4 core drugs pitched"] = "⚠ drugs recorded on a Sit down ride", RED
            rows.append(row); sty.append(s)
            continue
        items = _items(x["pitched"])
        cats = cat_of(D, day, x["mkt"])
        bag = {norm(b) for b in bag_for(F, D, day, x["mkt"])}
        bag_keys = bag | set(cats)
        keys = [resolve(d, bag_keys) for d in items]
        row["Drugs pitched (in order)"] = " → ".join(f"{i + 1}. {p}" for i, p in enumerate(x["pitched"])) or "none recorded"
        row["N"] = str(len(items))
        have = {c: any(CAT_CORE.get(cats.get(k)) == c for k in keys) for c in CORE}
        for c in CORE:
            row[CORE_LABEL[c]], s[CORE_LABEL[c]] = ("✓", GREEN) if have[c] else ("✗", RED)
        miss = [CORE_LABEL[c] for c in CORE if not have[c]]
        row["4 core drugs pitched"], s["4 core drugs pitched"] = ("✓", GREEN) if not miss else ("Missing: " + ", ".join(miss), RED)
        if not items:
            s["Drugs pitched (in order)"] = RED
        # order vs the marketer's first ride of the day
        seq = keys
        key = (x["mkt"], day)
        if key not in first_order and seq:
            first_order[key] = (x["ride"], seq, items)
            row["Order"], s["Order"] = "reference (first ride)", GREY
        elif seq:
            ref_ride, ref, ref_items = first_order[key]
            common_ref = [d for d in ref if d in seq]
            common_now = [d for d in seq if d in ref]
            notes = []
            if common_ref != common_now:
                notes.append(f"order changed vs ride {ref_ride}")
            dropped = [ri for ri, rk in zip(ref_items, ref) if rk not in seq]
            added = [it for it, k in zip(items, seq) if k not in ref]
            if dropped:
                notes.append("dropped: " + ", ".join(dropped))
            if added:
                notes.append("added: " + ", ".join(added))
            row["Order"] = "✓ same as ride " + ref_ride if not notes else "⚠ " + "; ".join(notes)
            s["Order"] = GREEN if not notes else (RED if common_ref != common_now else AMBER)
        else:
            row["Order"], s["Order"] = "—", GREY
        so = x.get("stockout") if isinstance(x.get("stockout"), list) else []
        row["Ran out (stockout)"] = ", ".join(so) if so else "—"
        s["Ran out (stockout)"] = RED if so else GREY
        not_bag = [d for d, k in zip(items, keys) if bag and k not in bag]
        row["Not in the morning list"] = ", ".join(not_bag) if not_bag else ("✓" if bag else "no Drug Info")
        s["Not in the morning list"] = AMBER if not_bag else (GREEN if bag else GREY)
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows).fillna("—")
    return t, pd.DataFrame(sty).reindex(columns=t.columns).fillna(GREY)


def summary_table(rides: pd.DataFrame) -> pd.DataFrame:
    """Per marketer: rides checked, rides with all 4 core drugs, rides with an order change."""
    if rides is None or rides.empty:
        return pd.DataFrame()
    r = rides[~rides["Pitch"].eq("Sit down") & ~rides["Drugs pitched (in order)"].eq("no Script Checklist yet")]
    if r.empty:
        return pd.DataFrame()
    g = r.groupby(["Team", "Marketer"])
    return pd.DataFrame({
        "Rides checked (Normal / Peddling)": g.size(),
        "Rides with all 4 core drugs (pain, deworming, cold, infection)":
            g["4 core drugs pitched"].apply(lambda s: f"{(s == '✓').sum()} of {len(s)}"),
        "Rides where the order changed": g["Order"].apply(lambda s: int(s.str.contains("order changed").sum())),
        "Rides with a drug dropped or added": g["Order"].apply(lambda s: int(s.str.contains("dropped|added").sum())),
        "Stockouts (drug ran out)": g["Ran out (stockout)"].apply(lambda s: ", ".join(x for x in s if x != "—") or "—")
            if "Ran out (stockout)" in r.columns else "—",
    }).reset_index()


def unmatched_names(R: pd.DataFrame, F: pd.DataFrame, D: pd.DataFrame, day) -> pd.DataFrame:
    """Drug names pitched on `day` that don't match the marketer's morning list: (marketer, written, bag options)."""
    day = pd.Timestamp(day).normalize()
    rows = []
    if R.empty:
        return pd.DataFrame()
    for _, x in R[(R["date"] == day) & (R["treatment"] != "S")].iterrows():
        bag_names = bag_for(F, D, day, x["mkt"])
        bag = {norm(b) for b in bag_names} | set(cat_of(D, day, x["mkt"]))
        for d in _items(x["pitched"] or []):
            if bag and resolve(d, bag) not in bag:
                rows.append({"Marketer": x["mkt"], "Written on the checklist": d,
                             "options": bag_names, "Rides": x["ride"]})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    t["_k"] = t["Written on the checklist"].map(norm)
    t = (t.groupby(["Marketer", "_k"], as_index=False)
          .agg({"Written on the checklist": "first", "options": "first", "Rides": lambda s: ", ".join(sorted(set(s)))}))
    return t.drop(columns="_k")


def reactions_table(R: pd.DataFrame, day) -> tuple:
    """One row per ride on `day` (Script Checklist): questions during the pitch (per drug), questions during the
    Sit down pause (and to whom), and passenger comments. Returns (table, css)."""
    day = pd.Timestamp(day).normalize()
    if R.empty:
        return pd.DataFrame(), pd.DataFrame()
    r_day = R[R["date"] == day].copy()
    if r_day.empty:
        return pd.DataFrame(), pd.DataFrame()
    r_day["_r"] = pd.to_numeric(r_day["ride"], errors="coerce")
    rows, sty = [], []
    for _, x in r_day.sort_values(["team", "mkt", "_r"]).iterrows():
        sit = x["treatment"] == "S"
        row = {"Team": x["team"], "Marketer": x["mkt"], "Ride": x["ride"],
               "Pitch": TREATMENT.get(x["treatment"], x["treatment"])}
        s = {k: "" for k in row}
        pq = x.get("pitch_questions") or []
        row["Questions during the pitch"] = " · ".join(pq) if pq else ("n/a (Sit down)" if sit else "none")
        s["Questions during the pitch"] = AMBER if pq else GREY
        if sit:
            if x.get("pause_q") == "1":
                who = ", ".join(x.get("pause_who") or []) or "?"
                row["Questions during the pause"] = f"Asked: {who} — {x.get('pause_what') or '(no text)'}"
                s["Questions during the pause"] = RED if "Marketer" in (x.get("pause_who") or []) else AMBER
            else:
                row["Questions during the pause"] = "none" if x.get("pause_q") == "0" else "—"
                s["Questions during the pause"] = GREY
        else:
            row["Questions during the pause"], s["Questions during the pause"] = "n/a", GREY
        if x.get("comment_yn") == "1":
            row["Passenger comments"] = x.get("comment") or "(yes, no text)"
            s["Passenger comments"] = RED if sit else AMBER
        else:
            row["Passenger comments"] = "none" if x.get("comment_yn") == "0" else "—"
            s["Passenger comments"] = GREY
        rows.append(row); sty.append(s)
    t = pd.DataFrame(rows)
    return t, pd.DataFrame(sty).reindex(columns=t.columns).fillna("")
