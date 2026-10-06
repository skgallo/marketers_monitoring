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
    batch_codes_url       = "https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=BATCH_CODES_GID"

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

    submissions_raw = _read_tab(submissions_url,  "submissions")
    batch_codes     = _read_tab(batch_codes_url,  "batch_codes")

    subs = prepare_submissions(submissions_raw)
    return submissions_raw, batch_codes, subs


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


def can_write_reviews() -> bool:
    try:
        return "gcp_service_account" in st.secrets
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
    return gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))


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
