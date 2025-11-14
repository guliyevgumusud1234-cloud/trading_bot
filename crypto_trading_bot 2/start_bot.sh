#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 KRİPTO TRADING BOT BAŞLATMA                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Virtual environment kontrolü
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment bulunamadı!"
    echo "💡 Lütfen önce şunu çalıştırın:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Virtual environment'ı aktif et
source venv/bin/activate

echo "🔍 Sistem kontrolü yapılıyor..."
echo ""

# .env dosyası kontrolü
if [ ! -f ".env" ]; then
    echo "❌ .env dosyası bulunamadı!"
    echo "💡 Lütfen .env dosyasını oluşturun:"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

echo "✅ .env dosyası bulundu"

# PostgreSQL kontrolü
if brew services list | grep -q "postgresql@15.*started"; then
    echo "✅ PostgreSQL çalışıyor"
else
    echo "⚙️  PostgreSQL başlatılıyor..."
    brew services start postgresql@15 2>/dev/null || echo "⚠️  PostgreSQL başlatılamadı (Manuel başlatmanız gerekebilir)"
fi

# Redis kontrolü
if brew services list | grep -q "redis.*started"; then
    echo "✅ Redis çalışıyor"
else
    echo "⚙️  Redis başlatılıyor..."
    brew services start redis 2>/dev/null || echo "⚠️  Redis başlatılamadı (Manuel başlatmanız gerekebilir)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Trading bot başlatılıyor..."
echo ""
echo "📊 Mod: PAPER TRADING (Gerçek para kullanılmaz)"
echo "🌐 Network: TESTNET"
echo ""
echo "⏹️  Durdurmak için: Ctrl+C"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Bot'u başlat
python main.py --paper --testnet

