import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from trading.bot import TBot
from utils.logger import setup_logger

def main():
    """Punto de entrada principal del bot"""
    logger = setup_logger("Main")
    
    try:  
        bot = TBot(timeframe="1m")
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()