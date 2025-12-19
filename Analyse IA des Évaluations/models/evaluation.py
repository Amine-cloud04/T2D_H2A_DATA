from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from enum import Enum

class FormationType(str, Enum):
    LEAN_SIX_SIGMA = "Lean Six Sigma"
    SAP = "SAP"
    PROCESSUS_METIER = "Processus Métier"
    SOFT_SKILLS = "Soft Skills"
    AUTRE = "Autre"

class Langue(str, Enum):
    FR = "FR"
    DARIJA = "Darija"
    AR = "AR"

class EvaluationBase(BaseModel):
    """Modèle d'une évaluation de formation"""
    evaluation_id: str
    formation_id: str
    type_formation: FormationType
    formateur_id: str
    satisfaction: int = Field(ge=1, le=5)
    contenu: int = Field(ge=1, le=5)
    logistique: int = Field(ge=1, le=5)
    applicabilite: int = Field(ge=1, le=5)
    commentaire: Optional[str] = None
    langue: Langue = Langue.FR
    date: date

class SentimentResult(BaseModel):
    """Résultat d'analyse de sentiment"""
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float
    confidence: float

class ThemeResult(BaseModel):
    """Thème extrait"""
    theme: str
    count: int
    sentiment_score: float
    keywords: List[str]

class AnalysisResult(BaseModel):
    """Résultat complet d'analyse"""
    total_evaluations: int
    avg_satisfaction: float
    avg_contenu: float
    avg_logistique: float
    avg_applicabilite: float
    sentiment_distribution: dict
    top_themes: List[ThemeResult]
    signaux_faibles: List[dict]
    kpis: dict

class FileUploadResponse(BaseModel):
    """Réponse après upload de fichier"""
    filename: str
    rows_processed: int
    analysis_id: str
    status: str