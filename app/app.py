import streamlit as st

from data_module import show_data_module
from risk_module import show_risk_module
from model_module import show_model_module
from reports_module import show_reports_module

st.set_page_config(page_title="Aplikacja Modelowania Ryzyka", layout="wide")
st.title("Aplikacja do eksploracji danych i modelowania ryzyka")

tabs = st.tabs(["Data", "Risk", "Model", "Reports"])
with tabs[0]:
    show_data_module()
with tabs[1]:
    show_risk_module()
with tabs[2]:
    show_model_module()
with tabs[3]:
    show_reports_module()
