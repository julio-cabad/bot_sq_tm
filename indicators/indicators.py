import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import talib
import numpy as np
import pandas as pd
import pandas_ta as ta
from utils.logger import setup_logger
from pandas_ta.utils import unsigned_differences
from numpy import nan as npNaN
from pandas import DataFrame
from pandas_ta.momentum import mom
from pandas_ta.overlap import ema, linreg, sma
from pandas_ta.trend import decreasing, increasing
from pandas_ta.volatility import bbands, kc
from pandas_ta.utils import get_offset
from pandas_ta.utils import unsigned_differences, verify_series

class Indicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.open = data['open']
        self.volume = data['volume']
        self.logger = setup_logger("Indicators")

    def rsi(self, period=14):
        try:
            return talib.RSI(self.close, timeperiod=period).iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculando RSI: {str(e)}")
            raise

    def macd(self):
        try:
            macd, signal, hist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
            return {
                'MACD': macd.iloc[-1],
                'Signal': signal.iloc[-1],
                'MACD_Hist': hist.iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculando MACD: {str(e)}")
            raise

    def bollinger_bands(self, timeperiod: int):
        upper, middle, lower = talib.BBANDS(self.close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0)
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]

    def ema(self, timeperiod: int):
        return round(talib.EMA(self.close, timeperiod=timeperiod).iloc[-1], 5)
        
    def sma(self, timeperiod: int):
        return round(talib.SMA(self.close, timeperiod=timeperiod).iloc[-1], 5)

    def trend_magic(self):
        color = 'green'
        cci = None
        period = 20
        coeff = 1
        ap = 5

        if cci is None:
            cci = talib.CCI(self.high, self.low, self.close, timeperiod=period)

        tr = talib.ATR(self.high, self.low, self.close, timeperiod=ap)

        up = self.low - tr * coeff
        down = self.high + tr * coeff

        magic_trend = pd.Series([0.0] * len(self.data))

        for i in range(len(self.data)):
            # Define el color de la línea MagicTrend.
            color = 'blue' if cci[i] > 0 else 'red'

            if cci[i] >= 0:
                if not math.isnan(up[i]):
                    magic_trend[i] = up[i] if i == 0 else max(up[i], magic_trend[i - 1])
                else:
                    magic_trend[i] = magic_trend[i - 1] if i > 0 else np.nan
            else:
                if not math.isnan(down[i]):
                    magic_trend[i] = down[i] if i == 0 else min(down[i], magic_trend[i - 1])
                else:
                    magic_trend[i] = magic_trend[i - 1] if i > 0 else np.nan

        mt = magic_trend.iloc[-1], 2

        if mt is not None:
            return color, round(magic_trend.iloc[-1], 3)

    def squeeze(self, bb_length=None, bb_std=None, kc_length=None, kc_scalar=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs):

        """Indicator: Squeeze Momentum (SQZ)"""
        print("Starting squeeze calculation")
        # Validate arguments
        bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
        bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
        kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
        kc_scalar = float(kc_scalar) if kc_scalar and kc_scalar > 0 else 1.5
        mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
        mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6
        _length = max(bb_length, kc_length, mom_length, mom_smooth)
        print("Using length:", _length)
        high = verify_series(self.high, _length)
        print("High after verify series:", high)


        low = verify_series(self.low, _length)
        print("Low after verify series:", low)


        close = verify_series(self.close, _length)
        print("Close after verify series:", close)

        offset = get_offset(offset)

        if high is None or low is None or close is None:
            print("Error: One or more input series is None")
            return


        use_tr = kwargs.setdefault("tr", True)
        asint = kwargs.pop("asint", True)
        detailed = kwargs.pop("detailed", False)
        lazybear = kwargs.pop("lazybear", False)
        mamode = mamode if isinstance(mamode, str) else "sma"

        def simplify_columns(df, n=3):
            df.columns = df.columns.str.lower()
            return [c.split("_")[0][n - 1:n] for c in df.columns]

        # Calculate Result
        bbd = ta.bbands(close, length=bb_length, std=bb_std, mamode=mamode)
        kch = ta.kc(high, low, close, length=kc_length, scalar=kc_scalar, mamode=mamode, tr=use_tr)

        # Simplify KC and BBAND column names for dynamic access
        bbd.columns = simplify_columns(bbd)
        kch.columns = simplify_columns(kch)

        if lazybear:
            highest_high = high.rolling(kc_length).max()
            lowest_low = low.rolling(kc_length).min()
            avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kch.b

            squeeze = linreg(close - avg_, length=kc_length)

        else:
            momo = mom(close, length=mom_length)
            if mamode.lower() == "ema":
                squeeze = ema(momo, length=mom_smooth)
            else: # "sma"
                squeeze = sma(momo, length=mom_smooth)

        # Classify Squeezes
        squeeze_on = (bbd.l > kch.l) & (bbd.u < kch.u)
        squeeze_off = (bbd.l < kch.l) & (bbd.u > kch.u)
        no_squeeze = ~squeeze_on & ~squeeze_off

        # Offset
        if offset != 0:
            squeeze = squeeze.shift(offset)
            squeeze_on = squeeze_on.shift(offset)
            squeeze_off = squeeze_off.shift(offset)
            no_squeeze = no_squeeze.shift(offset)

        # Handle fills
        if "fillna" in kwargs:
            squeeze.fillna(kwargs["fillna"], inplace=True)
            squeeze_on.fillna(kwargs["fillna"], inplace=True)
            squeeze_off.fillna(kwargs["fillna"], inplace=True)
            no_squeeze.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            squeeze.fillna(method=kwargs["fill_method"], inplace=True)
            squeeze_on.fillna(method=kwargs["fill_method"], inplace=True)
            squeeze_off.fillna(method=kwargs["fill_method"], inplace=True)
            no_squeeze.fillna(method=kwargs["fill_method"], inplace=True)

        # Name and Categorize it
        _props = "" if use_tr else "hlr"
        _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
        _props += "_LB" if lazybear else ""
        squeeze.name = f"SQZ{_props}"

        data = {
            squeeze.name: squeeze,
            f"SQZ_ON": squeeze_on.astype(int) if asint else squeeze_on,
            f"SQZ_OFF": squeeze_off.astype(int) if asint else squeeze_off,
            f"SQZ_NO": no_squeeze.astype(int) if asint else no_squeeze,
        }
        df = DataFrame(data)
        df.name = squeeze.name
        df.category = squeeze.category = "momentum"

        # Detailed Squeeze Series
        if detailed:
            pos_squeeze = squeeze[squeeze >= 0]
            neg_squeeze = squeeze[squeeze < 0]

            pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
            neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

            pos_inc *= squeeze
            pos_dec *= squeeze
            neg_dec *= squeeze
            neg_inc *= squeeze

            pos_inc.replace(0, npNaN, inplace=True)
            pos_dec.replace(0, npNaN, inplace=True)
            neg_dec.replace(0, npNaN, inplace=True)
            neg_inc.replace(0, npNaN, inplace=True)

            sqz_inc = squeeze * increasing(squeeze)
            sqz_dec = squeeze * decreasing(squeeze)
            sqz_inc.replace(0, npNaN, inplace=True)
            sqz_dec.replace(0, npNaN, inplace=True)

            # Handle fills
            if "fillna" in kwargs:
                sqz_inc.fillna(kwargs["fillna"], inplace=True)
                sqz_dec.fillna(kwargs["fillna"], inplace=True)
                pos_inc.fillna(kwargs["fillna"], inplace=True)
                pos_dec.fillna(kwargs["fillna"], inplace=True)
                neg_dec.fillna(kwargs["fillna"], inplace=True)
                neg_inc.fillna(kwargs["fillna"], inplace=True)
            if "fill_method" in kwargs:
                sqz_inc.fillna(method=kwargs["fill_method"], inplace=True)
                sqz_dec.fillna(method=kwargs["fill_method"], inplace=True)
                pos_inc.fillna(method=kwargs["fill_method"], inplace=True)
                pos_dec.fillna(method=kwargs["fill_method"], inplace=True)
                neg_dec.fillna(method=kwargs["fill_method"], inplace=True)
                neg_inc.fillna(method=kwargs["fill_method"], inplace=True)

            df[f"SQZ_INC"] = sqz_inc
            df[f"SQZ_DEC"] = sqz_dec
            df[f"SQZ_PINC"] = pos_inc
            df[f"SQZ_PDEC"] = pos_dec
            df[f"SQZ_NDEC"] = neg_dec
            df[f"SQZ_NINC"] = neg_inc

        print("Squeeze calculation completed")
        return df
    
    def squeeze_momentum(self):
        length = 20
        mult = 2.0
        length_KC = 20
        mult_KC = 1.5

        # Calculate BB
        m_avg = self.close.rolling(window=length).mean()
        m_std = self.close.rolling(window=length).std(ddof=0) * mult

        self.data['upper_BB'] = m_avg + m_std
        self.data['lower_BB'] = m_avg - m_std

        # Calculate TRue Range

        self.data['tr0'] = abs(self.high - self.low)
        self.data['tr1'] = abs(self.high - self.close.shift())
        self.data['tr2'] = abs(self.high - self.low.shift())
        self.data['tr'] = self.data[['tr0', 'tr1', 'tr2']].max(axis=1)

        # Calculate KC
        range_ma = self.data['tr'].rolling(window=length_KC).mean()
        self.data['upper_KC'] = m_avg + range_ma * mult_KC
        self.data['lower_KC'] = m_avg - range_ma * mult_KC

        # Calculate SQZ
        highest = self.high.rolling(window=length_KC).max()
        lowest = self.low.rolling(window=length_KC).min()
        m1 = (highest + lowest) / 2
        self.data['SQZ'] = self.close - (m1 + m_avg) / 2

        # Perform linear regression
        y = self.data['SQZ'].values
        x = np.arange(len(y))

        A = np.vstack([x, np.ones(len(x))]).T
        coefficients, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        # Calculate

        highest = self.high.rolling(window=length_KC).max()
        lowest = self.low.rolling(window=length_KC).min()
        m1 = (highest + lowest) / 2
        self.data['SQZ'] = self.close - (m1 + m_avg) / 2

        y = np.array(range(0, length_KC))
        func = lambda x: np.polyfit(y, x, 1)[0] * (length_KC - 1) + np.polyfit(y, x, 1)[1]
        self.data['SQZ'] = self.data['SQZ'].rolling(window=length_KC).apply(func, raw=True)
        self.data['price_change'] = self.data['SQZ'].diff()
        self.data['price_increase'] = self.data['SQZ'] > self.data['SQZ'].shift(1)
        self.data['price_decrease'] = self.data['SQZ'] < self.data['SQZ'].shift(1)
        self.data['price_trend'] = np.where(self.data['SQZ'] > self.data['SQZ'].shift(1), 'up',
                                            np.where(self.data['SQZ'] < self.data['SQZ'].shift(1), 'down',
                                                     'sin cambio'))

        return self.data['price_trend'].iloc[-1]