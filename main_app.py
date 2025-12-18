# main_app.py
import streamlit as st
from chatbot_rh_safran.app import run as run_chatbot
from 'Analyse et insights'.app import run as run_analysis

# --- SINGLE PAGE CONFIG ---
st.set_page_config(
    page_title="Safran POC Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("POC Safran")
app_choice = st.sidebar.radio(
    "Choisir l'application :",
    ["Chatbot RH", "Analyse T2D"]
)

# --- RUN SELECTED APP ---
if app_choice == "Chatbot RH":
    run_chatbot()
else:
    run_analysis()
