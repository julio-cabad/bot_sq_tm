import os
import logging
from logging.handlers import RotatingFileHandler
from config.settings import LOG_FILE
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    # Asegurarse de que el directorio de logs exista
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    
    # Si el logger ya tiene handlers, no agregar más
    if logger.handlers:
        return logger
        
    logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
    
    # Crear el formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Configurar el handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Agregar los handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Evitar la propagación de logs
    logger.propagate = False
    
    return logger
