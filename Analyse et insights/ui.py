import os
import logging
import streamlit as st
from core.knowledge_base import KnowledgeBase
from core.chatbot_engine import ChatbotEngine
from core.security_sim import SecuritySimulator

# Désactiver le file watcher pour éviter les erreurs PyTorch
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
logger = logging.getLogger(__name__)


@st.cache_resource
def init_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def init_security_sim():
    return SecuritySimulator()


def chatbot_page():
    # Initialisation
    kb = init_knowledge_base()
    security_sim = init_security_sim()

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatbotEngine(kb)

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if "session_token" not in st.session_state:
        st.session_state.session_token = None

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/fr/thumb/6/6e/Logo_Safran.svg/1280px-Logo_Safran.svg.png",
            width=200,
            caption="POC - Think to Deploy",
        )

        st.markdown("---")
        st.subheader("🔐 Simulation d'Authentification")

        user_choice = st.selectbox(
            "Choisissez un profil utilisateur :",
            ["Non authentifié"]
            + [
                f"{u['user_id']} - {u['name']} ({u['profil']})"
                for u in kb.user_profiles
            ],
            key="user_select_chatbot"
        )

        if st.button("Simuler Connexion", key="login_btn_chatbot"):
            if user_choice != "Non authentifié":
                user_id = user_choice.split(" - ")[0]
                st.session_state.session_token = security_sim.simulate_login(user_id)
                st.session_state.current_user = kb.get_user_profile(user_id)
                st.session_state.conversation = []
                st.rerun()

        if st.session_state.current_user:
            user = st.session_state.current_user
            st.markdown(f"**👤 {user['name']}** ({user['profil']})")
            if st.button("Déconnexion", key="logout_btn_chatbot"):
                st.session_state.current_user = None
                st.session_state.session_token = None
                st.session_state.conversation = []
                st.rerun()

    # ---------- MAIN ----------
    st.title("🤖 Assistant RH Safran")

    selected_tab = st.radio(
        "Navigation",
        ["💬 Chat", "📚 Base de Connaissances", "⚙️ Configuration"],
        horizontal=True
    )

    if selected_tab == "💬 Chat":
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Posez votre question RH")
        if user_input:
            st.session_state.conversation.append({"role": "user", "content": user_input})
            with st.spinner("Recherche..."):
                resp = st.session_state.chatbot.generate_response(
                    user_input, st.session_state.current_user
                )
            st.session_state.conversation.append(
                {"role": "assistant", "content": str(resp)}
            )
            st.rerun()
