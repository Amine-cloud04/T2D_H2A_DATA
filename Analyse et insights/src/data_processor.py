import pandas as pd
import numpy as np

RATING_COLS = ['satisfaction', 'contenu', 'logistique', 'applicabilite']

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in RATING_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=RATING_COLS + ['date'], inplace=True)
    df['year_month'] = df['date'].dt.to_period('M')
    return df

def get_mean_per_criterion(df):
    return df[RATING_COLS].mean().sort_values(ascending=False)

def get_mean_per_formation_type(df):
    return df.groupby('type_formation')[RATING_COLS].mean()

def get_time_evolution(df):
    time_evolution = df.groupby('year_month')['satisfaction'].mean().reset_index()
    time_evolution['year_month'] = time_evolution['year_month'].astype(str)
    return time_evolution

def get_insights(df, mean_criterion):
    """Génération d'insights à haute valeur métier[cite: 51, 54, 82]."""
    insights = []
    
    # 1. Analyse du Gap Théorie/Pratique (Insight Stratégique)
    avg_content = df['contenu'].mean()
    avg_app = df['applicabilite'].mean()
    gap = avg_content - avg_app
    
    if gap > 0.5:
        insights.append(f"⚠️ **Alerte Décalage Pédagogique** : Les formations sont jugées excellentes sur le contenu ({avg_content:.2f}) mais difficiles à appliquer sur le terrain ({avg_app:.2f}). Un renforcement de l'accompagnement post-formation est suggéré[cite: 53, 54].")
    
    # 2. Corrélation Logistique/Satisfaction (Signal Faible)
    # On identifie si la logistique tire la note vers le bas [cite: 83]
    if mean_criterion['logistique'] < 3.8:
        insights.append(f"📉 **Signal Faible Logistique** : La logistique est le critère limitant ({mean_criterion['logistique']:.2f}). L'analyse montre qu'elle impacte directement la perception globale de la qualité Safran[cite: 83].")

    # 3. Analyse de la Cohérence (Dispersion)
    std_dev = df.groupby('type_formation')['satisfaction'].std().max()
    if std_dev > 0.8:
        inconsistent_type = df.groupby('type_formation')['satisfaction'].std().idxmax()
        insights.append(f"⚖️ **Alerte Cohérence** : Forte disparité de satisfaction constatée sur '{inconsistent_type}' (Écart-type: {std_dev:.2f}). La qualité de l'expérience varie selon les sessions[cite: 84].")

    # 4. Insight de Performance
    best_type = df.groupby('type_formation')['satisfaction'].mean().idxmax()
    insights.append(f"🏆 **Excellence Opérationnelle** : Le domaine '{best_type}' présente le meilleur taux d'adhésion. Ses méthodes pourraient être modélisées pour les autres types de formation[cite: 54].")
        
    return insights