from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List
from loguru import logger
import re

class SentimentAnalyzer:
    """Analyse le sentiment des commentaires (FR + Darija)"""
    
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        logger.info(f"Loading sentiment model: {model_name}")
        
        # Charger le modèle multilingue
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Sentiment model loaded successfully")
    
    def preprocess_text(self, text: str) -> str:
        """Nettoie le texte avant analyse"""
        if not text:
            return ""
        
        # Supprimer les URLs
        text = re.sub(r'http\S+', '', text)
        
        # Supprimer les emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r'[^\w\s\.,!?àâäéèêëïîôùûüÿç]', ' ', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze(self, text: str) -> Dict:
        """Analyse le sentiment d'un texte"""
        if not text or len(text.strip()) < 3:
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "confidence": 0.0
            }
        
        # Prétraiter
        clean_text = self.preprocess_text(text)
        
        # Tronquer si trop long (max 512 tokens BERT)
        if len(clean_text) > 500:
            clean_text = clean_text[:500]
        
        try:
            # Analyser
            result = self.classifier(clean_text)[0]
            
            # Convertir le label (1-5 stars) en POSITIVE/NEGATIVE/NEUTRAL
            stars = int(result['label'].split()[0])
            
            if stars >= 4:
                label = "POSITIVE"
            elif stars <= 2:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            return {
                "label": label,
                "score": result['score'],
                "confidence": result['score'],
                "stars": stars
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "confidence": 0.0
            }
    
    def batch_analyze(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Analyse un batch de textes"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze(text) for text in batch]
            results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
        
        return results
    
    def get_sentiment_distribution(self, sentiments: List[Dict]) -> Dict:
        """Calcule la distribution des sentiments"""
        total = len(sentiments)
        if total == 0:
            return {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        
        distribution = {
            "POSITIVE": sum(1 for s in sentiments if s['label'] == 'POSITIVE'),
            "NEUTRAL": sum(1 for s in sentiments if s['label'] == 'NEUTRAL'),
            "NEGATIVE": sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
        }
        
        # Convertir en pourcentages
        distribution = {
            k: round((v / total) * 100, 1) 
            for k, v in distribution.items()
        }
        
        return distribution