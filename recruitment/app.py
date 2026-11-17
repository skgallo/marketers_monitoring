"""
Recruitment Monitoring — entry point (Streamlit Cloud runs this file: recruitment/app.py)
=======================================================================================
Only sets the page names shown in the sidebar. The pages themselves:
    recruitment_monitoring.py        main HFC dashboard (Today, Ride verification, …)
    pages/2_Forms_and_Attendance.py  forms and attendance
    pages/3_Drugs.py                 drug checks
"""
import streamlit as st

st.set_page_config(page_title="Recruitment Monitoring", page_icon="🚌", layout="wide")

pg = st.navigation([
    st.Page("recruitment_monitoring.py", title="Recruitment Monitoring", icon="🚌", default=True),
    st.Page("pages/2_Forms_and_Attendance.py", title="Forms and Attendance", icon="🗂️"),
    st.Page("pages/3_Drugs.py", title="Drugs", icon="💊"),
])
pg.run()
