import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from typing import Dict, List
from datetime import datetime
from utils.logger import setup_logger
from bnc.binance import RobotBinance
from indicators.indicators import Indicators
from config.settings import CRYPTOCURRENCY_LIST, SIMULATION_MODE
import pandas as pd
from src.strategies.dual_momentum import DualMomentumStrategy


class TBot:
    def __init__(self, timeframe: str = "1m"):
        self.logger = setup_logger("TrendMagicBot")
        self.timeframe = timeframe
        self.strategy = DualMomentumStrategy()
        self.positions = {}  # Diccionario para trackear posiciones abiertas
        self.clients = {}    # Diccionario para los clientes de cada par
        self.max_positions = 5
        self.initial_capital = 100  # USDT
        self.position_size = self.initial_capital / self.max_positions  # 20 USDT por posición
        self.leverage = 10
        self.taker_fee = 0.0005  # 0.05% para entrada
        self.maker_fee = 0.0004  # 0.04% para salida
        
        # Inicializar clientes para cada par
        for symbol in CRYPTOCURRENCY_LIST:
            self.clients[symbol] = RobotBinance(pair=symbol, temporality=timeframe)
        
        print("\033[2J\033[H", end='')
        
    def calculate_pnl(self, entry_price: float, current_price: float, side: str) -> tuple:
        position_value = self.position_size * self.leverage
        
        # Calcular comisiones
        entry_fee = position_value * self.taker_fee
        exit_fee = position_value * self.maker_fee
        total_fees = entry_fee + exit_fee
        
        if side == "LONG":
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            pnl_usdt = (current_price - entry_price) * (position_value / entry_price)
        else:  # SHORT
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
            pnl_usdt = (entry_price - current_price) * (position_value / entry_price)
        
        # Aplicar apalancamiento y restar comisiones
        pnl_percent = pnl_percent * self.leverage
        pnl_usdt = pnl_usdt - total_fees
        
        return pnl_percent, pnl_usdt
        
    def run(self):
        TEMPLATE = """=== TrendMagic Bot Status === {}
==================================================
Initial Capital: {:.2f} USDT | Positions: {}/{}
==================================================

Active Positions:
{}
==================================================
Total PNL: {}
=================================================="""

        # Códigos de color ANSI
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        while True:
            try:
                positions_info = []
                total_pnl_usdt = 0
                total_pnl_percent = 0
                
                # Verificar y limpiar posiciones duplicadas
                if len(self.positions) > self.max_positions:
                    self.logger.warning(f"Found {len(self.positions)} positions, cleaning excess...")
                    # Mantener solo las primeras max_positions
                    excess_positions = list(self.positions.keys())[self.max_positions:]
                    for symbol in excess_positions:
                        del self.positions[symbol]
                
                for symbol in CRYPTOCURRENCY_LIST:
                    try:
                        data = self.clients[symbol].candlestick()
                        indicators = Indicators(data)
                        current_price = float(data['close'].iloc[-1])
                        
                        tm_color, tm_value = indicators.trend_magic()
                        sqz = indicators.squeeze_momentum()
                        signal = self.strategy.analyze(tm_color, tm_value, sqz, log=False)
                        
                        # Actualizar posiciones existentes
                        if symbol in self.positions:
                            pos = self.positions[symbol]
                            pnl_percent, pnl_usdt = self.calculate_pnl(pos['entry_price'], current_price, pos['side'])
                            positions_info.append({
                                'symbol': symbol,
                                'side': pos['side'],
                                'entry_time': pos['timestamp'],
                                'entry_price': pos['entry_price'],
                                'current_price': current_price,
                                'pnl_percent': pnl_percent,
                                'pnl_usdt': pnl_usdt
                            })
                        # Abrir nuevas posiciones solo si hay espacio
                        elif signal.type != "NONE" and len(self.positions) < self.max_positions:
                            self.positions[symbol] = {
                                'side': signal.type,
                                'entry_price': current_price,
                                'timestamp': datetime.now()
                            }
                            self.logger.info(f"New position: {symbol} {signal.type} @ {current_price}")
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                
                # Ordenar posiciones por PNL
                positions_info.sort(key=lambda x: x['pnl_percent'], reverse=True)
                
                # Formatear información de posiciones
                positions_display = []  # Cambiar a lista para mejor control
                for pos in positions_info:
                    color = GREEN if pos['pnl_usdt'] >= 0 else RED
                    
                    # Formatear cada línea de posición
                    line = (
                        f"{pos['entry_time'].strftime('%H:%M:%S')} | "
                        f"{pos['symbol']:<8} {pos['side']:<5} | "
                        f"Entry: {pos['entry_price']:<10.2f} | "
                        f"Current: {pos['current_price']:<10.2f} | "
                        f"PNL: {color}${pos['pnl_usdt']:>7.2f} ({pos['pnl_percent']:>6.2f}%){RESET}"
                    )
                    positions_display.append(line)
                    
                    total_pnl_usdt += pos['pnl_usdt']
                    total_pnl_percent += pos['pnl_percent']
                
                # Unir las líneas con saltos de línea
                positions_text = "\n".join(positions_display) if positions_display else "No open positions"
                
                # Formatear el total
                color = GREEN if total_pnl_usdt >= 0 else RED
                total_display = f"{color}${total_pnl_usdt:>7.2f} ({total_pnl_percent:>6.2f}%){RESET}"
                
                # Limpiar pantalla y mostrar
                print("\033[H" + TEMPLATE.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.initial_capital,
                    len(self.positions),
                    self.max_positions,
                    positions_text,
                    total_display
                ))
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Fatal error: {str(e)}")
                break
