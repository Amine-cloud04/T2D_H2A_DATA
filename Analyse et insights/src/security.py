import pandas as pd

def anonymize_evaluations(df):
    """
    BLOC 2 : Couche de sécurité[cite: 61, 62]. 
    Garantit la conformité RGPD en filtrant les colonnes autorisées[cite: 43, 68].
    """
    safe_columns = [
        'formation_id', 'type_formation', 'satisfaction', 
        'contenu', 'logistique', 'applicabilite', 'commentaire', 
        'date', 'year_month'
    ]
    return df[safe_columns].copy() 