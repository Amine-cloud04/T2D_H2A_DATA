# 🛡️ Safran POC Dashboard : Intelligence RH & Analyse IA

![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![NLP](https://img.shields.io/badge/IA-NLP%20%26%20Chatbot-blue?style=for-the-badge)

## 📝 Présentation Générale
Ce projet est un **Proof of Concept (POC)** unifié développé pour **Safran**. Il s'agit d'une plateforme décisionnelle qui combine l'intelligence artificielle conversationnelle et l'analyse de données avancée (Business Intelligence augmentée) pour optimiser la gestion des formations et le support RH.

L'application agit comme un hub central permettant de basculer entre un assistant virtuel pour les collaborateurs et un outil d'audit stratégique pour les gestionnaires de formation.

---

## 🏗️ Architecture du Projet
Le projet repose sur une architecture modulaire avec un point d'entrée principal qui charge dynamiquement les sous-applications :

* **Portail Principal :** Gestion de la navigation et de la configuration globale.
* **Module Chatbot RH :** Assistant conversationnel dédié aux thématiques RH de Safran.
* **Module Analyse T2D (Think to Deploy) :** Moteur d'analyse NLP et statistique des évaluations de formation.

---

## ✨ Fonctionnalités Clés

### 1. 🤖 Chatbot RH Safran
Un agent intelligent capable de simuler des interactions humaines pour :
* Répondre aux questions fréquentes des collaborateurs.
* Orienter les employés vers les bonnes ressources RH.
* Offrir un support disponible 24h/24.

### 2. 📊 Analyse et Insights (T2D)
Ce module transforme les questionnaires de satisfaction en indicateurs stratégiques :
* **Analyse de Sentiment NLP :** Classification automatique des commentaires (Positif, Neutre, Négatif) avec détection de la confiance.
* **Clustering Intelligent :** Regroupement automatique des retours par thématiques (Logistique, Pédagogie, Contenu) via machine learning.
* **Détection de Signaux Faibles :** Identification proactive des problèmes critiques isolés (ex: alertes sur un formateur spécifique ou une infrastructure).
* **Dashboards Interactifs :** Visualisation des KPIs (Satisfaction moyenne, taux de complétion, évolution temporelle) via Plotly.
* **Recommandations IA :** Génération automatique de plans d'action basés sur les données analysées.

---

## 🛠️ Stack Technique
* **Frontend :** Streamlit (Multi-page dynamique via imports reflexifs).
* **Backend :** FastAPI (Serveur de données et logique métier sur le port 8000).
* **Analyse de données :** Pandas, NumPy.
* **Visualisation :** Plotly (Graphiques complexes et interactifs).
* **IA/NLP :** Traitement du langage naturel pour l'analyse textuelle et le clustering.

---



### Structure des Dossiers
Pour que la plateforme fonctionne, assurez-vous de respecter l'arborescence suivante :
```text
.
├── main_app.py                 # Point d'entrée principal
├── chatbot_rh_safran/          # Dossier du module Chatbot
│   └── app.py
├── Analyse et insights/       # Dossier du module d'analyse
│   └── app.py
└── requirements.txt            # Dépendances du projet
```
## Installation des bibliothèques nécessaires
pip install -r requirements.txt
# 1. Lancez votre API FastAPI (Backend)
python -m main.py
# 2. Lancez le Dashboard Streamlit :
streamlit run streamlit_app.py
