import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from loguru import logger
from models.evaluation import EvaluationBase, Langue, FormationType

class DataLoader:
    """Charge et valide les données d'évaluation"""
    
    def __init__(self):
        self.required_columns = [
            'evaluation_id', 'formation_id', 'type_formation',
            'formateur_id', 'satisfaction', 'contenu',
            'logistique', 'applicabilite', 'commentaire',
            'langue', 'date'
        ]
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Charge un fichier CSV"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return self.validate_and_clean(df)
        except UnicodeDecodeError:
            # Essayer avec latin-1 si utf-8 échoue
            df = pd.read_csv(file_path, encoding='latin-1')
            return self.validate_and_clean(df)
    
    def load_excel(self, file_path: str) -> pd.DataFrame:
        """Charge un fichier Excel"""
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return self.validate_and_clean(df)
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie les données"""
        # Vérifier les colonnes requises
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        # Nettoyer les données
        df = df.copy()
        
        # Supprimer les lignes avec satisfaction manquante
        df = df.dropna(subset=['satisfaction'])
        
        # Convertir les scores en entiers
        score_columns = ['satisfaction', 'contenu', 'logistique', 'applicabilite']
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(3)  # Valeur par défaut: neutre
            df[col] = df[col].clip(1, 5).astype(int)
        
        # Convertir la date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Nettoyer les commentaires
        df['commentaire'] = df['commentaire'].fillna('')
        df['commentaire'] = df['commentaire'].astype(str).str.strip()
        
        # Valider les types de formation
        valid_formations = [f.value for f in FormationType]
        df['type_formation'] = df['type_formation'].apply(
            lambda x: x if x in valid_formations else FormationType.AUTRE.value
        )
        
        logger.info(f"Cleaned data: {len(df)} valid rows")
        return df
    
    def to_evaluation_models(self, df: pd.DataFrame) -> List[EvaluationBase]:
        """Convertit le DataFrame en modèles Pydantic"""
        evaluations = []
        for _, row in df.iterrows():
            try:
                eval_model = EvaluationBase(
                    evaluation_id=str(row['evaluation_id']),
                    formation_id=str(row['formation_id']),
                    type_formation=row['type_formation'],
                    formateur_id=str(row['formateur_id']),
                    satisfaction=int(row['satisfaction']),
                    contenu=int(row['contenu']),
                    logistique=int(row['logistique']),
                    applicabilite=int(row['applicabilite']),
                    commentaire=row['commentaire'],
                    langue=row.get('langue', 'FR'),
                    date=row['date']
                )
                evaluations.append(eval_model)
            except Exception as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue
        
        return evaluations