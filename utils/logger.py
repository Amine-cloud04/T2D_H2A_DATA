# utils/logger.py
from loguru import logger
import sys
from pathlib import Path

def setup_logger(log_file: str = "logs/app.log", level: str = "INFO"):
    """
    Configure le système de logging
    """
    # Créer le dossier de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuration de loguru
    logger.remove()  # Supprimer le handler par défaut
    
    # Handler console
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Handler fichier
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level
    )
    
    logger.info("Logger initialized")
    return logger

# Export par défaut
__all__ = ['setup_logger', 'logger']
