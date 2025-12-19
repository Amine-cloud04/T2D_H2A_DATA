from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional
import shutil

from config import settings
from models.evaluation import AnalysisResult, FileUploadResponse
from services.data_loader import DataLoader
from services.sentiment_analyzer import SentimentAnalyzer
from services.topic_extractor import TopicExtractor
from services.clustering import CommentClustering, AnomalyDetector
from services.kpi_calculator import KPICalculator
from utils.logger import setup_logger
from utils.anonymization import anonymize_data




# Setup
logger = setup_logger()
app = FastAPI(
    title=settings.API_TITLE,
    version="1.0.0",
    description="API d'analyse automatisée des évaluations de formation - Safran"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser les services
data_loader = DataLoader()
sentiment_analyzer = SentimentAnalyzer()
topic_extractor = TopicExtractor()
kpi_calculator = KPICalculator()

# Stockage temporaire des analyses
analyses_cache = {}

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Safran Evaluation Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze/{analysis_id}",
            "kpis": "/api/kpis/{analysis_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "sentiment_analyzer": "ready",
            "topic_extractor": "ready",
            "kpi_calculator": "ready"
        }
    }

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload et traite un fichier CSV d'évaluations
    """
    logger.info(f"Receiving file: {file.filename}")
    
    # Vérifier l'extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non supportée. Accepté: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Vérifier la taille (optionnel)
    # Note: UploadFile ne donne pas directement la taille
    
    # Créer un ID unique pour cette analyse
    analysis_id = str(uuid.uuid4())
    
    # Sauvegarder temporairement le fichier
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{analysis_id}{file_ext}"
    
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Charger les données
        if file_ext == '.csv':
            df = data_loader.load_csv(str(temp_file))
        else:
            df = data_loader.load_excel(str(temp_file))
        
        # Anonymiser les données (RGPD)
        df = anonymize_data(df)
        
        # Stocker dans le cache
        analyses_cache[analysis_id] = {
            'dataframe': df,
            'filename': file.filename,
            'upload_time': datetime.now(),
            'status': 'uploaded',
            'rows': len(df)
        }
        
        logger.info(f"File processed successfully: {analysis_id}, {len(df)} rows")
        
        # Nettoyer le fichier temporaire
        temp_file.unlink()
        
        return FileUploadResponse(
            filename=file.filename,
            rows_processed=len(df),
            analysis_id=analysis_id,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/{analysis_id}")
async def analyze_evaluations(analysis_id: str):
    """
    Analyse complète d'un fichier uploadé
    """
    if analysis_id not in analyses_cache:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    try:
        logger.info(f"Starting analysis for: {analysis_id}")
        
        df = analyses_cache[analysis_id]['dataframe']
        
        # 1. Analyse de sentiment
        logger.info("Step 1/5: Sentiment analysis...")
        comments = df['commentaire'].fillna('').astype(str).tolist()
        sentiments = sentiment_analyzer.batch_analyze(comments)
        df['sentiment'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        
        # 2. Extraction de thèmes
        logger.info("Step 2/5: Topic extraction...")
        keywords = topic_extractor.extract_keywords_tfidf(comments, top_n=20)
        themes = topic_extractor.extract_themes_lda(comments, n_topics=5)
        
        # 3. Clustering
        logger.info("Step 3/5: Clustering...")
        clustering = CommentClustering(n_clusters=5)
        df['cluster'] = clustering.fit_predict(comments)
        cluster_analysis = clustering.analyze_clusters(df.copy())
        
        # 4. Détection de signaux faibles
        logger.info("Step 4/5: Anomaly detection...")
        anomaly_detector = AnomalyDetector()
        weak_signals = anomaly_detector.detect_weak_signals(df)
        outliers = anomaly_detector.detect_outliers_by_formateur(df)
        
        # 5. Calcul des KPIs
        logger.info("Step 5/5: KPI calculation...")
        kpis = kpi_calculator.calculate_all_kpis(df, sentiments)
        
        # Préparer le résultat
        result = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_evaluations': len(df),
                'avg_satisfaction': round(df['satisfaction'].mean(), 2),
                'completion_rate': kpis['global']['completion_rate']
            },
            'sentiment_analysis': {
                'distribution': sentiment_analyzer.get_sentiment_distribution(sentiments),
                'details': kpis['sentiment']
            },
            'topics': {
                'keywords': [{'word': kw[0], 'score': float(kw[1])} for kw in keywords[:15]],
                'themes': themes
            },
            'clustering': {
                'summary': cluster_analysis.to_dict('records'),
                'n_clusters': clustering.n_clusters
            },
            'weak_signals': weak_signals,
            'outliers': outliers,
            'kpis': kpis,
            'scores_by_criteria': df[['satisfaction', 'contenu', 'logistique', 'applicabilite']].describe().to_dict()
        }
        
        # Mettre à jour le cache
        analyses_cache[analysis_id]['analysis_result'] = result
        analyses_cache[analysis_id]['status'] = 'analyzed'
        
        logger.info(f"Analysis completed for: {analysis_id}")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kpis/{analysis_id}")
async def get_kpis(analysis_id: str):
    """
    Récupère uniquement les KPIs d'une analyse
    """
    if analysis_id not in analyses_cache:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    if 'analysis_result' not in analyses_cache[analysis_id]:
        raise HTTPException(
            status_code=400, 
            detail="Analysis not completed yet. Run /api/analyze first"
        )
    
    result = analyses_cache[analysis_id]['analysis_result']
    return JSONResponse(content=result['kpis'])

@app.get("/api/export/{analysis_id}")
async def export_results(analysis_id: str, format: str = "json"):
    """
    Exporte les résultats d'analyse
    """
    if analysis_id not in analyses_cache:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    if 'analysis_result' not in analyses_cache[analysis_id]:
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    result = analyses_cache[analysis_id]['analysis_result']
    
    if format == "json":
        return JSONResponse(content=result)
    
    elif format == "csv":
        # Exporter le DataFrame avec les résultats d'analyse
        df = analyses_cache[analysis_id]['dataframe']
        csv_data = df.to_csv(index=False)
        
        from fastapi.responses import Response
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.csv"}
        )
    
    else:
        raise HTTPException(status_code=400, detail="Format not supported")

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Supprime une analyse du cache
    """
    if analysis_id in analyses_cache:
        del analyses_cache[analysis_id]
        return {"message": "Analysis deleted", "analysis_id": analysis_id}
    else:
        raise HTTPException(status_code=404, detail="Analysis ID not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )