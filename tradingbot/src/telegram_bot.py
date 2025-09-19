import os
from telegram import Bot
from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler

# .env dosyasından Telegram API token'ı yükle
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")  # Telegram chat ID

# Binance API'sine bağlan
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'urls': {
        'api': 'https://testnet.binancefuture.com/api',  # Testnet URL'si
    }
})

# Telegram botu başlat
bot = Bot(token=TELEGRAM_TOKEN)

# Telegram botu ile mesaj gönderme fonksiyonu
def send_telegram_message(message):
    bot.send_message(chat_id=chat_id, text=message)

# Telegram komutları
def start(update, context):
    update.message.reply_text("Ticaret botu başlatıldı. İşlemleri takip ediyorsunuz.")

# Botu başlat
def run_bot():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    run_bot()