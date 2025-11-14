#!/bin/bash

# Training'i arka planda başlatmak için script
# Veriyi çeker, hazırlar ve modelleri eğitir

cd "$(dirname "$0")"
source venv/bin/activate

# Log dosyası
TRAINING_LOG="logs/training.log"
mkdir -p logs
mkdir -p models/trained

echo "═══════════════════════════════════════════════════════════"
echo "🚀 TRAINING BAŞLATILIYOR..."
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📋 Yapılacaklar:"
echo "   1️⃣  Veri çekme (2-3 yıl geçmiş veri)"
echo "   2️⃣  Feature engineering (200+ özellik)"
echo "   3️⃣  Model training (5 strateji)"
echo "   4️⃣  Walk-forward validation"
echo "   5️⃣  Model kaydetme"
echo ""
echo "⏱️  Tahmini süre: 3-8 saat"
echo "📝 Loglar: $TRAINING_LOG"
echo ""

# Training parametreleri
START_DATE="${1:-2022-01-01}"  # Varsayılan: 2022-01-01
NO_OPTIMIZE="${2:-}"  # --no-optimize flag'i

# Eğer --no-optimize verilmişse
OPTIMIZE_FLAG=""
if [ "$NO_OPTIMIZE" == "--no-optimize" ] || [ "$NO_OPTIMIZE" == "no-optimize" ]; then
    OPTIMIZE_FLAG="--no-optimize"
    echo "⚡ Hızlı mod: Hyperparameter optimization kapalı"
    echo "   (Daha hızlı ama daha az optimize)"
else
    echo "🎯 Tam mod: Hyperparameter optimization açık"
    echo "   (Daha yavaş ama daha iyi sonuçlar)"
fi

echo ""
echo "📅 Başlangıç tarihi: $START_DATE"
echo ""

# Eğer training zaten çalışıyorsa uyar
if pgrep -f "python train.py" > /dev/null; then
    echo "⚠️  Training zaten çalışıyor!"
    echo "   PID: $(pgrep -f 'python train.py' | head -1)"
    echo ""
    read -p "Yine de devam etmek istiyor musunuz? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ İptal edildi"
        exit 1
    fi
fi

# Training'i arka planda başlat
echo "🚀 Training başlatılıyor..."
echo ""

if [ -n "$OPTIMIZE_FLAG" ]; then
    nohup python train.py --start-date "$START_DATE" $OPTIMIZE_FLAG > "$TRAINING_LOG" 2>&1 &
else
    nohup python train.py --start-date "$START_DATE" > "$TRAINING_LOG" 2>&1 &
fi

TRAINING_PID=$!

sleep 3

# Process kontrolü
if ps -p $TRAINING_PID > /dev/null; then
    echo "✅ Training başarıyla başlatıldı!"
    echo ""
    echo "📊 Bilgiler:"
    echo "   • PID: $TRAINING_PID"
    echo "   • Log dosyası: $TRAINING_LOG"
    echo "   • Başlangıç tarihi: $START_DATE"
    echo ""
    echo "📋 Kullanışlı komutlar:"
    echo "   tail -f $TRAINING_LOG              # Logları izle"
    echo "   ps -p $TRAINING_PID                 # Process durumu"
    echo "   kill $TRAINING_PID                  # Training'i durdur"
    echo ""
    echo "🔍 İlerlemeyi izlemek için:"
    echo "   tail -f $TRAINING_LOG | grep -E '(Step|Training|Iteration)'"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
else
    echo "❌ Training başlatılamadı!"
    echo "   Logları kontrol edin: tail -20 $TRAINING_LOG"
    exit 1
fi

