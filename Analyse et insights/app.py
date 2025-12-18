def run() :
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import sys

    # Ajout du dossier 'src' au chemin système pour l'importation des modules sécurisés
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

    # Importation des fonctions depuis les composants isolés (Bloc 2: Sandbox)
    from data_processor import load_data, get_mean_per_criterion, get_mean_per_formation_type, get_time_evolution, get_insights
    from security import anonymize_evaluations


    # --- BLOC 1 & 2 : SOURCING ET SÉCURITÉ (Isolation et Anonymisation) ---
    @st.cache_data
    def load_and_secure():
        """
        Assure le sourcing des données (Bloc 1) et leur anonymisation (Bloc 2).
        Garantit qu'aucune donnée nominative ne sort du SI Safran.
        """
        #  1. Import des données fictives fournies (Sourcing)
        raw_df = load_data('evaluation_formation.csv')
    
    # 2. Application de la couche d'anonymisation (Sécurité)
    # On filtre les colonnes comme 'formateur_id' pour ne garder que la valeur métier
        return anonymize_evaluations(raw_df)

    try:
    # Chargement des données traitées par la sandbox de sécurité
        df = load_and_secure()
        st.sidebar.success("Statut : SaaS Isolé / Données Anonymisées ✅")
        st.sidebar.info(f"Évaluations chargées : {len(df)}")
    except Exception as e:
        st.error(f"Erreur d'accès aux données : {e}")
        st.stop()

# --- TITRE ET CONTEXTE ---
    st.title("💡 Analyse IA des Évaluations de Formation")
    st.markdown("""
        **Conformité Industrielle :** Ce POC démontre une capacité à traiter les données RH de manière sécurisée[cite: 12]. 
        L'architecture est isolée du SI Safran pour prévenir toute fuite de données[cite: 69].
    """)

    st.divider()

    # --- BLOC 3 & 4 : ANALYSE ET SORTIE (L'Usine IA) ---
    # Calcul des KPIs obligatoires (Moyennes par critère et Évolution temporelle)
    mean_criterion = get_mean_per_criterion(df)
    time_evolution = get_time_evolution(df)

    st.header("1. Indicateurs Clés de Performance (KPIs)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Satisfaction Moyenne par Critère")
    
    # KPI : Satisfaction globale moyenne
        st.metric(label="Satisfaction Globale", value=f"{round(mean_criterion['satisfaction'], 2)}/5")
    
        # Visualisation : Graphique simple par critère [cite: 86]
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=mean_criterion.index, y=mean_criterion.values, palette="viridis", ax=ax1)
        ax1.set_title('Performance (Échelle 1-5)')
        ax1.set_ylim(1, 5)
        st.pyplot(fig1)

    with col2:
        st.subheader("Évolution dans le Temps")
    
    # Visualisation : Évolution temporelle [cite: 81]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.lineplot(x='year_month', y='satisfaction', data=time_evolution, marker='o', color='blue', ax=ax2)
        ax2.set_title('Tendance de la Satisfaction Mensuelle')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.divider()

# --- SIGNAUX FAIBLES ET INSIGHTS (Génération automatique) ---
    st.header("2. Insights IA et Signaux Faibles")
    st.markdown("Analyse automatique pour identifier les axes d'amélioration exploitables par Safran[cite: 54].")

    insights = get_insights(df, mean_criterion)
    for insight in insights:
    # Affichage des insights lisibles (ex: Logistique = critère le plus critiqué) [cite: 82]
        st.info(f"🔍 {insight}")

    st.divider()

# --- ANALYSE DÉTAILLÉE (Tableau de synthèse) ---
    st.header("3. Performance par Type de Formation")
    st.markdown("Tableau de synthèse montrant la performance moyenne par domaine[cite: 87].")

    mean_per_type = get_mean_per_formation_type(df)
# Affichage interactif avec mise en évidence des scores
    st.dataframe(
        mean_per_type.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ffb3b3'),
        use_container_width=True
    )

    st.divider()
    st.caption("POC T2D (Think To Deploy) - Phase 1 - 1ère Édition")
