import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
from typing import Optional
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET
from binance.um_futures import UMFutures
from utils.logger import setup_logger
import os
from dotenv import load_dotenv

class RobotBinance:
    def __init__(self, pair: str, temporality: str):
        """Initialize Binance client with API credentials."""
        self.pair = pair.upper()
        self.temporality = temporality
        self.symbol = self.pair
        self.logger = setup_logger("RobotBinance")
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize and configure Binance client."""
        try:
            load_dotenv()
            
            client = UMFutures(
                key=os.getenv('BINANCE_API_KEY'),
                secret=os.getenv('BINANCE_API_SECRET'),
                base_url="https://fapi.binance.com"  # Usando mainnet directamente
            )
            
            # Verificar la conexión
            time_res = client.time()
            self.logger.debug(f"Connected to Binance mainnet")
            self.logger.debug(f"Server time: {time_res['serverTime']}")
            
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {str(e)}")
            raise

    def binance_client(self):
        return self.client

    def _request(self, func, **kwargs):
        """Wrapper para manejar errores en las peticiones a Binance."""
        retries = 0
        max_retries = 3
        delay = 5  # seconds

        while retries < max_retries:
            try:
                return func(**kwargs)
            except Exception as e:
                self.logger.error(f"Error en petición a Binance: {str(e)}")
                retries += 1
                if retries < max_retries:
                    self.logger.info(f"Reintentando en {delay} segundos... (Intento {retries}/{max_retries})")
                    time.sleep(delay)
                else:
                    self.logger.error("Se alcanzó el máximo de intentos")
                    raise

    def binance_account(self) -> dict:
        """Obtiene información de la cuenta"""
        return self._request(self.client.account)

    def symbol_price(self) -> float:
        """Get current price for the symbol."""
        try:
            ticker = self.client.ticker_price(symbol=self.symbol)
            if ticker and isinstance(ticker, dict) and 'price' in ticker:
                price = float(ticker['price'])
                if price > 0:
                    return price
            self.logger.warning(f"Invalid price data for {self.symbol}: {ticker}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {self.symbol}: {str(e)}")
            return None

    def open_orders(self, pair: str):
        """Obtiene las órdenes abiertas"""
        return self._request(self.client.get_open_orders, symbol=pair)

    def candlestick(self) -> pd.DataFrame:
        """Get candlestick data for the symbol."""
        try:
            klines = self.client.klines(
                symbol=self.symbol,
                interval=self.temporality,
                limit=400
            )
            
            if not klines:
                raise ValueError(f"No candlestick data received for {self.symbol}")
                
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir columnas a tipos numéricos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Verificar si hay valores NaN
            if df[numeric_columns].isna().any().any():
                self.logger.warning(f"Found NaN values in {self.symbol} data")
                df = df.dropna(subset=numeric_columns)
                
            if len(df) < 2:
                raise ValueError(f"Insufficient data for {self.symbol} after cleaning")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting candlestick data for {self.symbol}: {str(e)}")
            raise

    def change_leverage(self, symbol: str, leverage: int):
        """Cambia el apalancamiento para un símbolo"""
        return self._request(
            self.client.change_leverage,
            symbol=symbol,
            leverage=leverage
        )

    def place_order(self, symbol: str, side: str, quantity: float):
        """
        Coloca una orden de mercado
        Args:
            symbol: Par de trading
            side: 'BUY' o 'SELL'
            quantity: Cantidad a operar
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'recvWindow': 60000
        }
        
        return self._request(self.client.new_order, **params)
