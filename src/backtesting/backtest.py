import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import List, Dict
import pandas as pd
from src.strategies.dual_momentum import DualMomentumStrategy

class Backtest:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000):
        self.data = data
        self.capital = initial_capital
        self.strategy = DualMomentumStrategy()
        self.positions: List[Dict] = []
        
    def run(self):
        """Execute backtest"""
        print('=-=-=-=-=-==-=-=-=-=-=-=-')
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            
            # Get indicator values
            tm_color = row['tm_color']
            tm_value = row['tm_value']
            sqz = row['sqz']
            
            # Get signal
            signal = self.strategy.analyze(tm_color, tm_value, sqz)
            
            if signal.type != "NONE":
                self.positions.append({
                    'timestamp': row.name,
                    'type': signal.type,
                    'price': signal.price,
                    'reason': signal.reason
                })
                
    def get_results(self):
        """Calculate and return backtest results"""
        return pd.DataFrame(self.positions) 