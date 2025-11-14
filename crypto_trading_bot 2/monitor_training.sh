#!/bin/bash

# Training ilerlemesini izlemek için script

cd "$(dirname "$0")"

TRAINING_LOG="logs/full_training.log"
TRAINING_PID=$(pgrep -f "python train.py" | head -1)

if [ -z "$TRAINING_PID" ]; then
    echo "❌ Training çalışmıyor!"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "📊 TRAINING İLERLEME MONİTÖRÜ"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "🔄 Process: PID $TRAINING_PID"
echo "📝 Log: $TRAINING_LOG"
echo ""

# Sürekli izle
while ps -p $TRAINING_PID > /dev/null 2>&1; do
    clear
    echo "═══════════════════════════════════════════════════════════"
    echo "📊 TRAINING İLERLEME - $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Process durumu
    ps -p $TRAINING_PID -o etime,pcpu,pmem | tail -1 | awk '{print "⏱️  Çalışma Süresi: " $1 " | CPU: " $2 "% | RAM: " $3 "%"}'
    echo ""
    
    # Son adım
    echo "📋 Son Adım:"
    tail -20 $TRAINING_LOG | grep -E "(Step|Training|Engineering|Saving)" | tail -1 | sed 's/.*INFO.*- //' || echo "   Bekleniyor..."
    echo ""
    
    # Eğitilen modeller
    echo "✅ Eğitilen Modeller:"
    tail -100 $TRAINING_LOG | grep -E "✅.*trained|Saved.*to" | tail -5 | sed 's/.*INFO.*- //' || echo "   Henüz yok"
    echo ""
    
    # Hatalar
    ERROR_COUNT=$(tail -50 $TRAINING_LOG | grep -c "ERROR" || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  Son Hatalar:"
        tail -50 $TRAINING_LOG | grep "ERROR" | tail -3 | sed 's/.*ERROR.*- //'
        echo ""
    fi
    
    # Son 5 satır
    echo "📝 Son Loglar:"
    tail -5 $TRAINING_LOG | sed 's/.*INFO.*- //' | tail -3
    
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🔄 5 saniye sonra güncellenecek... (Ctrl+C ile çık)"
    
    sleep 5
done

# Training tamamlandı
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ TRAINING TAMAMLANDI!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📊 Sonuçlar:"
tail -30 $TRAINING_LOG | grep -E "(✅|Saved|TRAINING PIPELINE COMPLETE)" | tail -10
echo ""
echo "📁 Kaydedilen Modeller:"
ls -lh models/trained/*.{pkl,cbm,txt} 2>/dev/null | awk '{print "   • " $9 " (" $5 ")"}'
echo ""
echo "═══════════════════════════════════════════════════════════"

