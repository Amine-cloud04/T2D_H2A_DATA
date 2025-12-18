import streamlit as st
import sys
import os
import importlib.util

# --- ADD CHATBOT FOLDER TO PATH ---
chatbot_folder = os.path.join(os.getcwd(), "chatbot_rh_safran")
if chatbot_folder not in sys.path:
    sys.path.insert(0, chatbot_folder)

from chatbot_rh_safran.app import run as run_chatbot

# --- DYNAMIC IMPORT FOR "Analyse et insights" ---
analysis_folder = os.path.join(os.getcwd(), "Analyse et insights")
analysis_app_path = os.path.join(analysis_folder, "app.py")

spec = importlib.util.spec_from_file_location("run_analysis", analysis_app_path)
run_analysis_module = importlib.util.module_from_spec(spec)
sys.modules["run_analysis"] = run_analysis_module
spec.loader.exec_module(run_analysis_module)
run_analysis = run_analysis_module.run

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
