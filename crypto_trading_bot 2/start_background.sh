#!/bin/bash

# Arka planda bot'u başlatmak için script
# tmux kullanarak veya nohup ile çalıştırır

cd "$(dirname "$0")"
source venv/bin/activate

# Log dosyası
LOG_FILE="logs/trading.log"
mkdir -p logs

echo "🚀 Trading Bot arka planda başlatılıyor..."
echo "📝 Loglar: $LOG_FILE"
echo ""

# tmux session adı
SESSION_NAME="trading_bot"

# Eğer session zaten varsa, önce kapat
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
    echo "⚠️  Mevcut session bulundu, kapatılıyor..."
    tmux kill-session -t $SESSION_NAME
    sleep 2
fi

# Yeni tmux session oluştur ve bot'u başlat
tmux new-session -d -s $SESSION_NAME -c "$(pwd)" \
    "source venv/bin/activate && python main.py --paper 2>&1 | tee -a $LOG_FILE"

sleep 3

# Session durumunu kontrol et
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✅ Bot başarıyla başlatıldı!"
    echo ""
    echo "📋 Kullanışlı komutlar:"
    echo "   tmux attach -t $SESSION_NAME    # Bot'u görmek için"
    echo "   tmux kill-session -t $SESSION_NAME  # Bot'u durdurmak için"
    echo "   tail -f $LOG_FILE                # Logları izlemek için"
    echo ""
    echo "🔍 Bot durumunu kontrol et:"
    tmux list-sessions | grep $SESSION_NAME
else
    echo "❌ Bot başlatılamadı! Logları kontrol edin:"
    echo "   tail -20 $LOG_FILE"
fi

