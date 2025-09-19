import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from utils.utils import calculate_rsi, calculate_macd, calculate_ema, calculate_atr

# .env dosyasından API anahtarlarını yükle
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# Binance API'sine bağlan
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'urls': {
        'api': 'https://testnet.binancefuture.com/api',  # Testnet URL'si
    }
})

def fetch_initial_data(symbol, timeframe):
    # İlk 700 mum verisini al (başlangıç verisi)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=700)
    ohlcv_df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Göstergeleri hesapla
    ohlcv_df['RSI'] = calculate_rsi(ohlcv_df)
    ohlcv_df['MACD'], ohlcv_df['MACD_signal'], _ = calculate_macd(ohlcv_df)
    ohlcv_df['EMA'] = calculate_ema(ohlcv_df)
    ohlcv_df['ATR'] = calculate_atr(ohlcv_df)
    
    return ohlcv_df

def append_new_data(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1)  # son mum
    latest_data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    file_path = f"data/ohlcv/{symbol}_{timeframe}.parquet"
    
    if os.path.exists(file_path):
        existing_data = pd.read_parquet(file_path)
        combined_data = pd.concat([existing_data, latest_data], ignore_index=True)
        combined_data.to_parquet(file_path, index=False)
    else:
        latest_data.to_parquet(file_path, index=False)