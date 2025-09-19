import talib
import pandas as pd

def calculate_rsi(data, period=14):
    """RSI hesaplama fonksiyonu"""
    return talib.RSI(data['close'], timeperiod=period)

def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD hesaplama fonksiyonu"""
    macd, macdsignal, macdhist = talib.MACD(data['close'], fastperiod, slowperiod, signalperiod)
    return macd, macdsignal, macdhist

def calculate_ema(data, period=14):
    """EMA hesaplama fonksiyonu"""
    return talib.EMA(data['close'], timeperiod=period)

def calculate_atr(data, period=14):
    """ATR hesaplama fonksiyonu"""
    return talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)