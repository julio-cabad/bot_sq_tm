from dataclasses import dataclass
from typing import Literal, Optional
from utils.logger import setup_logger

@dataclass
class Signal:
    type: Literal["LONG", "SHORT", "CLOSE", "NONE"]
    price: float
    reason: str

class DualMomentumStrategy:
    def __init__(self):
        self.logger = setup_logger("DualMomentumStrategy")
        self.previous_tm_value = None
        self.current_positions = {}  # Trackear posiciones por símbolo

    def analyze(self, tm_color: str, tm_value: float, sqz: str, log: bool = True) -> Signal:
        signal = Signal(type="NONE", price=tm_value, reason="No signal")
        
        if log:
            self.logger.info(f"Analyzing - TM Color: {tm_color}, TM Value: {tm_value}, SQZ: {sqz}")
        
        # Check for long entry
        if (tm_color == "blue" and sqz == "up"):
            signal = Signal(
                type="LONG",
                price=tm_value,
                reason="TM Blue + SQZ Up"
            )
            if log:
                self.logger.info(f"LONG Signal Generated: {signal.reason}")

        # Check for short entry
        elif (tm_color == "red" and sqz == "down"):
            signal = Signal(
                type="SHORT",
                price=tm_value,
                reason="TM Red + SQZ Down"
            )
            if log:
                self.logger.info(f"SHORT Signal Generated: {signal.reason}")

        self.previous_tm_value = tm_value
        return signal 