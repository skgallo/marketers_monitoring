"""
data_io.py — Google Sheet reader for the Recruitment HFC dashboard
==================================================================
Reads the submissions and batch_codes tabs from the live Google Sheet
and returns DataFrames ready to pass into rollup().

The Sheet must be shared as "Anyone with the link → Viewer".
Tab URLs (and the app password) live in .streamlit/secrets.toml,
which is never committed to GitHub.

secrets.toml format
-------------------
    app_password          = "your-password"
    submissions_url       = "https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=SUBMISSIONS_GID"
    submissions_url_nov   = "https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=NOV_FORM_GID"   # optional
    batch_codes_url      = "https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=BATCH_CODES_GID"

How to find the GID
-------------------
Open the Sheet, click the tab → look at the URL: #gid=1234567 is the GID.
"""

import re

import streamlit as st
import pandas as pd
from rollup import rollup, prepare_submissions

# ---------------------------------------------------------------------------
# Tab URL helpers
# ---------------------------------------------------------------------------

def _get_url(key: str) -> str:
    """
    Read a tab URL from secrets.toml.
    Raises a clear error if the key is missing so the dashboard fails loudly.
    """
    try:
        return st.secrets[key]
    except KeyError:
        raise RuntimeError(
            f"Missing key '{key}' in .streamlit/secrets.toml.\n"
            f"Add the Google Sheet CSV export URL for this tab."
        )


def _read_tab(url: str, tab_name: str) -> pd.DataFrame:
    """Read one Sheet tab via CSV export URL."""
    try:
        df = pd.read_csv(url, dtype=str)  # read everything as str first
    except Exception as e:
        raise RuntimeError(
            f"Could not read the '{tab_name}' tab.\n"
            f"URL: {url}\n"
            f"Error: {e}\n\n"
            f"Check that the Sheet is shared as 'Anyone with the link → Viewer'."
        )
    # Replace empty strings with None so pd.isna() works downstream
    df = df.replace({"": None, "nan": None})
    # Tidy headers: Google's name-based export can add spaces or merge the first
    # data row into the header (e.g. "batch_code 2210...") — keep the first word.
    df.columns = [str(c).strip() for c in df.columns]
    if "batch_code" not in df.columns:
        for c in df.columns:
            if c.lower().startswith("batch_code"):
                df = df.rename(columns={c: "batch_code"})
                break
    return df


# ---------------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)   # refresh every 5 minutes; user can force with Refresh button
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load submissions and batch_codes from Google Sheets and prepare them.

    Returns
    -------
    submissions_raw : raw DataFrame straight from the Sheet
    batch_codes     : batch codes frame
    subs            : prepare_submissions() output — one row per form (E1 and
                      supervisor), with flags, people and ride dates
    """
    submissions_url  = _get_url("submissions_url")
    batch_codes_url  = _get_url("batch_codes_url")

    # One tab per SurveyCTO form: the Oct form (submissions_url) and, once it is
    # published, the Nov form (submissions_url_nov). Forms are stacked; columns
    # that exist in only one form are left empty for the other.
    parts = [_read_tab(submissions_url, "submissions").assign(form_source="Oct 2026")]
    try:
        nov_url = st.secrets.get("submissions_url_nov")
    except Exception:
        nov_url = None
    if nov_url:
        parts.append(_read_tab(nov_url, "submissions (Nov form)").assign(form_source="Nov 2026"))
    submissions_raw = pd.concat(parts, ignore_index=True, sort=False)
    batch_codes     = _read_tab(batch_codes_url,  "batch_codes")

    subs = prepare_submissions(submissions_raw)
    return submissions_raw, batch_codes, subs


@st.cache_data(ttl=300)
def load_e4() -> pd.DataFrame:
    """
    Script Checklist (Enumerator 4) forms, published by SurveyCTO to their own tab/Sheet.
    Secret: e4_url = "<CSV export link of that tab>". Empty frame if not set yet.
    """
    from rollup import prepare_e4
    try:
        url = st.secrets.get("e4_url")
    except Exception:
        url = None
    if not url:
        return pd.DataFrame()
    try:
        return prepare_e4(_read_tab(url, "Script Checklist (E4)"))
    except RuntimeError as e:
        out = pd.DataFrame()
        out.attrs["error"] = str(e)
        return out


# ---------------------------------------------------------------------------
# Reviews tab — read, and write back from the dashboard
# ---------------------------------------------------------------------------
# Writing needs a Google service account (a "robot" login) in secrets:
#
#   [gcp_service_account]
#   type = "service_account"
#   project_id = "..."
#   private_key_id = "..."
#   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
#   client_email = "hfc-dashboard@PROJECT.iam.gserviceaccount.com"
#   client_id = "..."
#   token_uri = "https://oauth2.googleapis.com/token"
#
# and the Sheet shared with client_email as Editor. The Sheet ID is taken from
# submissions_url (or set sheet_id = "..." explicitly).
# Without it, reviews are read-only from reviews_url (a CSV export link).

REVIEWS_TAB = "reviews"
REVIEW_COLUMNS = ["timestamp", "batch_code", "issue", "decision", "note", "reviewed_by",
                  "form_id", "field", "correct_value"]


def _service_account_info():
    """
    The service-account key, from either
      gcp_service_account_json = '''<the whole JSON key file, pasted as-is>'''   (easiest)
    or the older [gcp_service_account] block. None if neither is set.
    """
    import json
    try:
        raw = st.secrets.get("gcp_service_account_json")
    except Exception:
        raw = None
    if raw:
        return json.loads(raw)
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None


def can_write_reviews() -> bool:
    try:
        return _service_account_info() is not None
    except Exception:
        return False


def _sheet_id() -> str:
    try:
        return st.secrets["sheet_id"]
    except Exception:
        pass
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", _get_url("submissions_url"))
    if not m:
        raise RuntimeError("Could not find the Sheet ID in submissions_url — add sheet_id to secrets.")
    return m.group(1)


@st.cache_resource
def _sheets_client():
    import gspread
    return gspread.service_account_from_dict(_service_account_info())


def _reviews_worksheet():
    """The reviews worksheet, created (with headers) if it doesn't exist yet."""
    import gspread
    sh = _sheets_client().open_by_key(_sheet_id())
    try:
        ws = sh.worksheet(REVIEWS_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(REVIEWS_TAB, rows=1000, cols=len(REVIEW_COLUMNS))
    header = ws.row_values(1)
    missing = [c for c in REVIEW_COLUMNS if c not in header]
    if missing:
        header = header + missing
        if ws.col_count < len(header):
            ws.add_cols(len(header) - ws.col_count)
        ws.update(values=[header], range_name="A1")
    return ws, header


FEEDBACK_TAB = "feedback_log"
FEEDBACK_COLUMNS = ["timestamp", "date", "enumerator", "rides", "supervised", "message", "sent_by"]


def _worksheet(tab: str, columns: list):
    """Any tab in the Sheet, created with headers if it doesn't exist yet."""
    import gspread
    sh = _sheets_client().open_by_key(_sheet_id())
    try:
        ws = sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(tab, rows=1000, cols=len(columns))
    header = ws.row_values(1)
    missing = [c for c in columns if c not in header]
    if missing:
        header = header + missing
        if ws.col_count < len(header):
            ws.add_cols(len(header) - ws.col_count)
        ws.update(values=[header], range_name="A1")
    return ws, header


def append_feedback(row: dict) -> None:
    """Record a daily feedback message as sent (feedback_log tab)."""
    ws, header = _worksheet(FEEDBACK_TAB, FEEDBACK_COLUMNS)
    ws.append_rows([[str(row.get(c, "") or "") for c in header]], value_input_option="RAW")
    load_feedback_log.clear()


@st.cache_data(ttl=300)
def load_feedback_log() -> pd.DataFrame:
    """Messages already marked as sent (empty if saving isn't set up)."""
    if not can_write_reviews():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    try:
        ws, _ = _worksheet(FEEDBACK_TAB, FEEDBACK_COLUMNS)
        values = ws.get_all_values()
        if len(values) <= 1:
            return pd.DataFrame(columns=values[0] if values else FEEDBACK_COLUMNS)
        return pd.DataFrame(values[1:], columns=values[0])
    except Exception:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def append_reviews(rows: list) -> int:
    """Append decision rows (dicts keyed by REVIEW_COLUMNS) to the reviews tab."""
    if not rows:
        return 0
    ws, header = _reviews_worksheet()
    ws.append_rows([[str(r.get(c, "") or "") for c in header] for r in rows],
                   value_input_option="RAW")
    load_reviews.clear()
    return len(rows)


@st.cache_data(ttl=300)
def load_reviews() -> pd.DataFrame:
    """
    Your decisions about flagged differences, one row per decision
    (see REVIEW_COLUMNS). Read through the service account when configured,
    otherwise from reviews_url. Empty frame if neither is set or the tab is empty.
    """
    if can_write_reviews():
        try:
            ws, _ = _reviews_worksheet()
            values = ws.get_all_values()
            if len(values) <= 1:
                return pd.DataFrame(columns=values[0] if values else REVIEW_COLUMNS)
            return pd.DataFrame(values[1:], columns=values[0]).replace({"": None})
        except Exception as e:
            out = pd.DataFrame()
            out.attrs["error"] = f"Could not read the reviews tab: {e}"
            return out
    url = None
    for key in ("reviews_url", "resolutions_url"):
        try:
            url = st.secrets[key]
            break
        except Exception:
            continue
    if not url:
        return pd.DataFrame()
    try:
        return _read_tab(url, "reviews")
    except RuntimeError as e:
        out = pd.DataFrame()
        out.attrs["error"] = str(e)
        return out


# ---------------------------------------------------------------------------
# Patterns export → tabs in a Google Sheet (through the service account)
# ---------------------------------------------------------------------------
# Service accounts can't own files in a personal Google Drive, so instead of uploading an Excel file
# the tables are written as tabs ("P · …") of a Google Sheet the service account can edit:
# patterns_sheet_id in Secrets, or the HFC Sheet itself. File → Download → Excel gives the .xlsx.

def patterns_sheet_id() -> str:
    try:
        return st.secrets.get("patterns_sheet_id") or _sheet_id()
    except Exception:
        return _sheet_id()


def write_patterns(tables: dict, stamp: str) -> str:
    """Overwrite one tab per table (prefixed 'P · ') plus a 'P · README' with the time. Returns the Sheet URL."""
    import gspread
    sid = patterns_sheet_id()
    sh = _sheets_client().open_by_key(sid)
    existing = {ws.title: ws for ws in sh.worksheets()}
    items = [("README", pd.DataFrame({"Patterns snapshot": [f"Saved from the HFC dashboard on {stamp}",
                                                            "Each 'P · ' tab is overwritten on every save."]}))]
    items += list(tables.items())
    for name, t in items:
        title = f"P · {name}"[:99]
        t = t.copy().astype(object).where(pd.notna(t), "")
        values = [list(map(str, t.columns))] + [[str(x) for x in row] for row in t.values.tolist()]
        ws = existing.get(title)
        if ws is None:
            ws = sh.add_worksheet(title, rows=max(len(values) + 10, 50), cols=max(len(t.columns) + 2, 8))
        else:
            ws.clear()
            if ws.row_count < len(values) or ws.col_count < len(t.columns):
                ws.resize(rows=max(len(values) + 10, ws.row_count), cols=max(len(t.columns) + 2, ws.col_count))
        ws.update(values=values, range_name="A1")
    return f"https://docs.google.com/spreadsheets/d/{sid}"


# ---------------------------------------------------------------------------
# SurveyCTO attachments (photos, audio audits) — fetched by the app and shown inside the dashboard
# ---------------------------------------------------------------------------
# Secrets:  surveycto_server = "salone"
#           surveycto_username = "<a SurveyCTO user with permission to download this form's data>"
#           surveycto_password = "<that user's password>"
# Viewers of the dashboard never see these; files are only shown to people logged in to the dashboard.

def can_fetch_media() -> bool:
    try:
        return bool(st.secrets.get("surveycto_server") and st.secrets.get("surveycto_username")
                    and st.secrets.get("surveycto_password"))
    except Exception:
        return False


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=300)
def fetch_attachment(form_id: str, submission_key: str, filename: str) -> tuple[bytes | None, str]:
    """(file bytes or None, message). Tries the API v2 attachment URL with basic, then digest login."""
    import requests
    from urllib.parse import quote
    server = st.secrets["surveycto_server"]
    user, pwd = st.secrets["surveycto_username"], st.secrets["surveycto_password"]
    fn = str(filename).split("/")[-1]
    url = (f"https://{server}.surveycto.com/api/v2/forms/{form_id}/submissions/"
           f"{quote(str(submission_key), safe='')}/attachments/{quote(fn)}")
    last = ""
    for auth in (requests.auth.HTTPBasicAuth(user, pwd), requests.auth.HTTPDigestAuth(user, pwd)):
        try:
            r = requests.get(url, auth=auth, timeout=30)
        except Exception as e:
            last = f"Could not reach SurveyCTO: {e}"
            continue
        if r.status_code == 200 and r.content:
            return r.content, "ok"
        last = f"SurveyCTO answered {r.status_code}: {r.text[:200]}"
    return None, last


# ---------------------------------------------------------------------------
# Diagnostics helper (shown in the dashboard's diagnostics section)
# ---------------------------------------------------------------------------

def diagnostics(submissions_raw: pd.DataFrame, batch_codes: pd.DataFrame, subs: pd.DataFrame) -> dict:
    """Counts shown in the dashboard footer — if something looks off, start here."""
    role = subs.get("role", pd.Series(dtype=str))
    return {
        "submissions_tab_rows" : len(submissions_raw),
        "batch_codes_tab_rows" : len(batch_codes),
        "forms_after_cleaning" : len(subs),
        "e1_forms"             : int((role == "Enumerator 1").sum()),
        "supervisor_forms"     : int((role == "Supervisor").sum()),
        **{f"forms_{k.replace(' ', '_').lower()}": int(n)
           for k, n in subs.get("form_source", pd.Series(dtype=str)).value_counts().items()},
        "dropped_as_test"      : len(submissions_raw) - len(subs),
    }


# ---------------------------------------------------------------------------
# Local dev helper (run directly to smoke-test without Streamlit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For local testing without Streamlit, pass URLs directly as args or env vars.
    import sys, os

    subs_url = os.environ.get("SUBMISSIONS_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    bc_url   = os.environ.get("BATCH_CODES_URL") or (sys.argv[2] if len(sys.argv) > 2 else None)

    if not subs_url or not bc_url:
        print("Usage: SUBMISSIONS_URL=... BATCH_CODES_URL=... python data_io.py")
        print("  or:  python data_io.py <submissions_url> <batch_codes_url>")
        sys.exit(1)

    print("Reading submissions tab…")
    subs = _read_tab(subs_url, "submissions")
    print(f"  {len(subs)} rows × {len(subs.columns)} cols")

    print("Reading batch_codes tab…")
    bc = _read_tab(bc_url, "batch_codes")
    print(f"  {len(bc)} rows × {len(bc.columns)} cols")

    prepared = prepare_submissions(subs)
    print(diagnostics(subs, bc, prepared))
