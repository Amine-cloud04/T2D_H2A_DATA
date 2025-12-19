from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configuration centralisée"""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Safran Evaluation Analysis API"
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/safran"
    
    # NLP Models
    SENTIMENT_MODEL: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    FRENCH_MODEL: str = "fr_core_news_sm"
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".csv", ".xlsx", ".xls"]
    
    # Security
    JWT_SECRET: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()