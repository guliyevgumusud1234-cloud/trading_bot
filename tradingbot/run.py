import time
from src.datafeed.ws_client import fetch_initial_data, append_new_data
from src.trading.exchange import save_open_position, close_position
from src.rl.dqn_agent import model
from src.telegram_bot import send_trade_open, send_trade_close, send_balance_update

# Coinler ve zaman dilimleri
coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'LTC/USDT', 'UNI/USDT', 'KAS/USDT', 'LINK/USDT', 'TRX/USDT', 'MATIC/USDT', 'BMT/USDT']
timeframes = ['1m', '5m', '15m', '1h', '4h']

# State verisini oluşturma
def get_state_from_data(symbol, timeframe):
    data = fetch_initial_data(symbol, timeframe)  # Veriyi çek
    rsi = data['RSI']  # RSI göstergesi
    macd = data['MACD']  # MACD göstergesi
    macd_signal = data['MACD_signal']  # MACD sinyal hattı
    ema = data['EMA']  # EMA göstergesi
    atr = data['ATR']  # ATR göstergesi
    state = [data['close'][-5:], rsi[-5:], macd[-5:], ema[-5:], atr[-5:]]
    return state

# Bot başlatma fonksiyonu
def start_trading():
    while True:
        for symbol in coins:
            for timeframe in timeframes:
                append_new_data(symbol, timeframe)
                state = get_state_from_data(symbol, timeframe)
                
                action = model(state)
                if action == 0:  # long
                    save_open_position({"symbol": symbol, "side": "long"})
                    send_trade_open(symbol, 50000, "long", 1)
                elif action == 1:  # short
                    save_open_position({"symbol": symbol, "side": "short"})
                    send_trade_open(symbol, 50000, "short", 1)
        
        time.sleep(60)  # Bir dakika bekle

if __name__ == "__main__":
    start_trading()