def run():
    import os
    import logging
    import streamlit as st
    from core.knowledge_base import KnowledgeBase
    from core.chatbot_engine import ChatbotEngine
    from core.security_sim import SecuritySimulator

    # Désactiver le file watcher pour éviter les erreurs PyTorch
    os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
    logger = logging.getLogger(__name__)

    # Configuration de la page
    st.set_page_config(
        page_title="Safran RH Assistant POC",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    # Initialisation avec cache pour performance
    @st.cache_resource
    def init_knowledge_base():
        return KnowledgeBase()


    @st.cache_resource
    def init_security_sim():
        return SecuritySimulator()


    def main():
        # Initialisation
        kb = init_knowledge_base()
        security_sim = init_security_sim()

        # Initialisation du chatbot dans l'état de session
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = ChatbotEngine(kb)

        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        if "current_user" not in st.session_state:
            st.session_state.current_user = None

        if "session_token" not in st.session_state:
            st.session_state.session_token = None

        # Sidebar - Configuration et informations
        with st.sidebar:
            st.image(
                "https://upload.wikimedia.org/wikipedia/fr/thumb/6/6e/Logo_Safran.svg/1280px-Logo_Safran.svg.png",
                width=200,
                caption="POC - Think to Deploy",
            )

            st.markdown("---")

            # Simulation d'authentification
            st.subheader("🔐 Simulation d'Authentification")

            user_choice = st.selectbox(
                "Choisissez un profil utilisateur :",
                ["Non authentifié"]
                + [
                    f"{u['user_id']} - {u['name']} ({u['profil']})"
                    for u in kb.user_profiles
                ],
                key="user_select"  # CLAÉ POUR ÉVITER LES CONFLITS
            )

            if st.button("Simuler Connexion", type="secondary", key="login_btn"):
                if user_choice != "Non authentifié":
                    user_id = user_choice.split(" - ")[0]
                    # Simulation de login
                    session_token = security_sim.simulate_login(user_id)
                    user_profile = kb.get_user_profile(user_id=user_id)

                    st.session_state.session_token = session_token
                    st.session_state.current_user = user_profile

                    st.success(
                        f"✅ Connecté en tant que {user_profile['name']} ({user_profile['profil']})"
                    )
                    st.session_state.conversation = []  # Nouvelle conversation
                    st.rerun()
                else:
                    st.session_state.current_user = None
                    st.session_state.session_token = None
                    st.info("Mode anonyme activé")
                    st.rerun()

            # Afficher l'utilisateur courant
            if st.session_state.current_user:
                user = st.session_state.current_user
                st.markdown(f"**👤 Utilisateur :** {user['name']}")
                st.markdown(f"**📋 Profil :** {user['profil']}")
                st.markdown(f"**🏢 Département :** {user['department']}")

                if st.button("Déconnexion", type="primary", key="logout_btn"):
                    st.session_state.current_user = None
                    st.session_state.session_token = None
                    st.session_state.conversation = []
                    st.rerun()

            st.markdown("---")

            # Statistiques
            st.subheader("📊 Statistiques")
            st.markdown(f"**Base de connaissances :** {len(kb.df_rh)} Q/R")
            st.markdown(f"**Profils supportés :** {', '.join(kb.get_profiles())}")
            st.markdown(f"**Domaines couverts :** {', '.join(kb.get_domains())}")

            # Rapport de sécurité
            if st.session_state.current_user:
                sec_report = security_sim.get_security_report()
                st.markdown("---")
                st.subheader("🔒 Journal de Sécurité (simulé)")
                st.markdown(f"**Sessions actives :** {sec_report['active_sessions']}")
                st.markdown(f"**Accès refusés :** {sec_report['denied_access_attempts']}")

            st.markdown("---")
            st.caption("POC Think to Deploy - Version 1.0")
            st.caption("Données fictives - Sécurité simulée")

        # Main area - Chatbot
        st.title("🤖 Assistant RH Safran - Proof of Concept")
        st.markdown(
            """
        **Démonstration des fonctionnalités :**
        - 🔍 Recherche sémantique dans la base RH
        - 👥 Réponses adaptées au profil (CDI, CDD, Cadre, Stagiaire...)
        - 🔐 Simulation de sécurité entreprise
        - 💬 Interface intuitive et professionnelle
        """
        )

        # Navigation
        selected_tab = st.radio(
            "Navigation",
            ["💬 Chat", "📚 Base de Connaissances", "⚙️ Configuration"],
            index=0,
            label_visibility="collapsed",
            horizontal=True,
            key="nav_tabs"
        )

        if selected_tab == "💬 Chat":
            # helper: normalize the engine response to a string
            def _extract_response_text(resp):
                if resp is None:
                    return ""
                if isinstance(resp, str):
                    return resp
                if isinstance(resp, dict):
                    return resp.get("reponse") or resp.get("answer") or str(resp)
                if isinstance(resp, list) and resp:
                    first = resp[0]
                    if isinstance(first, dict):
                        return first.get("reponse") or first.get("answer") or str(first)
                    return str(first)
                return str(resp)

            # helper: handle adding a user question and generating assistant answer
            def _handle_question(q: str):
                st.session_state.conversation.append({"role": "user", "content": q})
                with st.spinner("🔍 Recherche..."):
                    resp = st.session_state.chatbot.generate_response(query=q, user_profile=st.session_state.current_user)
                text = _extract_response_text(resp)
                st.session_state.conversation.append({"role": "assistant", "content": text})
                st.rerun()

            # Affichage conversation (top-level, outside of tabs/expanders)
            st.markdown("## Conversation")
            for msg in st.session_state.conversation:
                role = msg.get("role", "assistant")
                with st.chat_message(role):
                    st.write(msg.get("content", ""))

            # Champ d'entrée utilisateur (doit être hors tabs/columns)
            user_input = st.chat_input("Posez votre question RH (congés, paie, transport...)")
            if user_input:
                _handle_question(user_input)

            # Boutons d'actions rapides
            st.markdown("### 🚀 Questions rapides")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗓️ Congés annuels", width="stretch", key="btn_conges"):
                    _handle_question("Comment poser un congé annuel ?")
            with col2:
                if st.button("💰 Salaire", width="stretch", key="btn_salaire"):
                    _handle_question("Quand est versé le salaire ?")
            with col3:
                if st.button("🚌 Transport", width="stretch", key="btn_transport"):
                    _handle_question("Comment s'inscrire au transport ?")

        elif selected_tab == "📚 Base de Connaissances":
            st.subheader("📚 Exploration de la Base de Connaissances RH")

            # Filtres
            col1, col2 = st.columns(2)
            with col1:
                selected_domain = st.selectbox(
                    "Filtrer par domaine", ["Tous"] + kb.get_domains(), key="domain_filter"
                )

            with col2:
                selected_profile = st.selectbox(
                    "Filtrer par profil", ["Tous"] + kb.get_profiles(), key="profile_filter"
                )

            # Application des filtres
            filtered_df = kb.df_rh.copy()

            if selected_domain != "Tous":
                filtered_df = filtered_df[filtered_df["domaine"] == selected_domain]

            if selected_profile != "Tous":
                filtered_df = filtered_df[filtered_df["profil"] == selected_profile]

            # Affichage
            st.dataframe(
                filtered_df[["domaine", "profil", "question", "reponse"]],
                use_container_width=True,
                hide_index=True,
                height=400
            )

            # Statistiques
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions dans la base", len(kb.df_rh))
            with col2:
                st.metric("Questions filtrées", len(filtered_df))

        elif selected_tab == "⚙️ Configuration":
            st.subheader("⚙️ Configuration et Informations Techniques")

            st.markdown(
                """
            **Architecture du POC :**

            ```python
            Chatbot RH Safran POC
            ├── Interface Streamlit (app.py)
            ├── Moteur Chatbot (RAG + règles)
            │   ├── Recherche sémantique
            │   ├── Détection d'intention
            │   └── Personnalisation par profil
            ├── Base de connaissances
            │   ├→ Embeddings multilingues
            │   └→ Filtrage métier
            └── Simulateur de sécurité
                ├→ Authentification simulée
                ├→ Contrôle d'accès
                └→ Journalisation
            ```

            **Points clés pour Safran :**
            1. **🔒 Isolation des données** : Aucun accès direct au SI Safran
            2. **🛡️ Sécurité by design** : Authentification, journalisation, contrôle d'accès
            3. **👥 Personnalisation** : Réponses adaptées au profil (CDI, Cadre, Stagiaire...)
            4. **📈 Évolutivité** : Architecture prête pour l'industrialisation
            """
            )

            # Tester la recherche
            st.markdown("---")
            st.subheader("🔍 Tester la recherche manuellement")

            test_query = st.text_input("Entrez une requête de test :", "congé annuel")
            if st.button("Tester la recherche", type="primary"):
                with st.spinner("Recherche en cours..."):
                    results = kb.semantic_search(test_query, top_k=3)

                    if results:
                        st.success(f"✅ {len(results)} résultat(s) trouvé(s)")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Résultat {i} - Score: {result['similarity']:.2f}"):
                                st.markdown(f"**Question :** {result['question']}")
                                st.markdown(f"**Réponse :** {result['reponse']}")
                                st.markdown(f"**Profil :** {result['profil']}")
                                st.markdown(f"**Domaine :** {result['domaine']}")
                    else:
                        st.warning("⚠️ Aucun résultat trouvé")


    main()
