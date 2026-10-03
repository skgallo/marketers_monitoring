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

import streamlit as st
import pandas as pd
from rollup import rollup

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
    Load submissions and batch_codes from Google Sheets, run the rollup,
    and return (submissions_raw, batch_codes, result).

    The Streamlit cache means the Sheet is only re-fetched after 5 minutes
    (or when the user clicks Refresh). The app shows which tab was read and when.

    Returns
    -------
    submissions_raw : raw DataFrame straight from the Sheet
    batch_codes     : batch codes frame
    result          : rollup output — one row per planned batch code
    """
    submissions_url  = _get_url("submissions_url")
    batch_codes_url  = _get_url("batch_codes_url")

    submissions_raw = _read_tab(submissions_url,  "submissions")
    batch_codes     = _read_tab(batch_codes_url,  "batch_codes")

    result = rollup(submissions_raw, batch_codes)
    return submissions_raw, batch_codes, result


# ---------------------------------------------------------------------------
# Diagnostics helper (shown in the dashboard's diagnostics section)
# ---------------------------------------------------------------------------

def diagnostics(submissions_raw: pd.DataFrame, batch_codes: pd.DataFrame, result: pd.DataFrame) -> dict:
    """
    Return a dict of diagnostic counts shown in the dashboard footer.
    Fail loudly: missing columns or unexpected row counts surface here.
    """
    submitted    = result["submission_status"].eq("Submitted").sum()
    no_sub       = result["submission_status"].eq("No submission").sum()

    diag = {
        "submissions_tab_rows"  : len(submissions_raw),
        "batch_codes_tab_rows"  : len(batch_codes),
        "rollup_rows"           : len(result),
        "submitted"             : int(submitted),
        "no_submission"         : int(no_sub),
    }
    return diag


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

    from rollup import rollup
    result = rollup(subs, bc)
    print(f"Rollup: {len(result)} rows")
    print(diagnostics(subs, bc, result))
