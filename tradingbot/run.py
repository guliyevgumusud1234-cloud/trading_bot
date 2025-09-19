import time
from src.datafeed.ws_client import fetch_initial_data, append_new_data
from src.trading.exchange import save_open_position, close_position
from src.rl.dqn_agent import model
from src.telegram_bot import send_trade_open, send_trade_close, send_balance_update
import torch

# State verisini oluşturma
def get_state_from_data(symbol, timeframe):
    data = fetch_initial_data(symbol, timeframe)  # Veriyi çek
    rsi = calculate_rsi(data)
    macd, _, _ = calculate_macd(data)
    ema = calculate_ema(data)
    atr = calculate_atr(data)
    
    # Son 5 mum verisini alabiliriz
    state = [data['close'][-5:], rsi[-5:], macd[-5:], ema[-5:], atr[-5:]]
    
    return state

# Bot başlatma fonksiyonu
def start_trading():
    symbol = 'BTC/USDT'
    timeframe = '1m'
    
    while True:
        # Veriyi çek ve kaydet
        append_new_data(symbol, timeframe)
        
        # State verisini oluştur
        state = get_state_from_data(symbol, timeframe)
        
        # İşlem kararını al
        action = model(state)  # model ile işlem kararı al
        if action == 0:  # long
            save_open_position({"symbol": symbol, "side": "long"})
            send_trade_open(symbol, 50000, "long", 1)
        elif action == 1:  # short
            save_open_position({"symbol": symbol, "side": "short"})
            send_trade_open(symbol, 50000, "short", 1)
        
        time.sleep(60)

if __name__ == "__main__":
    start_trading()