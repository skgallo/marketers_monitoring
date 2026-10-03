"""
auth.py — Shared password gate for the Recruitment HFC dashboard
================================================================
Call require_password() at the top of app.py before any content.
The password lives in .streamlit/secrets.toml (never on GitHub):

    app_password = "your-password-here"

If the user enters the wrong password they see nothing else.
The session stays unlocked until the browser tab is closed.
"""

import streamlit as st


def require_password() -> None:
    """
    Block the app until the correct password is entered.
    Safe to call on every page load — uses session_state to avoid
    re-prompting once the user is already authenticated.
    """
    if st.session_state.get("authenticated"):
        return

    st.title("Recruitment HFC Dashboard")
    st.markdown("Enter the password to continue.")

    pw = st.text_input("Password", type="password", key="pw_input")

    if pw:
        try:
            correct = st.secrets["app_password"]
        except KeyError:
            st.error(
                "app_password is not set in .streamlit/secrets.toml. "
                "Add it before running the app."
            )
            st.stop()

        if pw == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
            st.stop()
    else:
        st.stop()
