import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de Binance
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'kwfKlJS6T7reidknCL0ElxoC0yk4ihwDDFOQBiMyAVqJol55EKE5dMAmYIz4niPa')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'FcNzA5YiafpOo9EwGEP2XyLFAFf2yBBk9g6MbfZnnU8xrGENUlEG4EffRQO2wvGU')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'

# Configuración de Trading
INITIAL_CAPITAL = 100  # Capital inicial en USDT
MAX_POSITIONS = 5  # Número máximo de posiciones simultáneas
POSITION_SIZE = INITIAL_CAPITAL / MAX_POSITIONS  # 20 USDT por posición
LEVERAGE = 10  # Apalancamiento
TOTAL_LEVERAGED_VALUE = INITIAL_CAPITAL * LEVERAGE  # 1000 USDT

# Comisiones de Binance
TAKER_FEE = 0.0005  # 0.05% para entrada
MAKER_FEE = 0.0004  # 0.04% para salida

# Niveles de Take Profit y Stop Loss
TAKE_PROFIT_PERCENTAGE = 0.04  # 4%
STOP_LOSS_PERCENTAGE = 0.02    # 2%
TOTAL_PROFIT_TARGET = 0.005    # 0.5% del total apalancado

# Modo de simulación
SIMULATION_MODE = True  # True para solo simular operaciones sin abrir posiciones reales

# Trading Pairs - Solo los pares más líquidos disponibles en testnet
CRYPTOCURRENCY_LIST = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT','TIAUSDT', 'SUIUSDT',
    'SOLUSDT', 'TRXUSDT', 'TONUSDT', 'LTCUSDT', 'DOTUSDT', 'AVAXUSDT', 'FETUSDT',
    'LINKUSDT', 'XLMUSDT', 'BCHUSDT', 'XMRUSDT', 'ATOMUSDT', 'LDOUSDT', 'HBARUSDT',
    'ICPUSDT', 'APTUSDT', 'ARBUSDT', 'NEARUSDT', 'VETUSDT', 'GRTUSDT', 'QNTUSDT', 'FTMUSDT',
    'OPUSDT', 'RUNEUSDT', 'AAVEUSDT', 'FLOWUSDT', 'CHZUSDT', 'KAVAUSDT', 'AXSUSDT', 'IOTAUSDT'
]

# Configuración de Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/trading_bot.log'
