#!/bin/bash
echo "🚀 Bot'u API Key olmadan başlatıyoruz (sınırlı mod)..."
echo ""
echo "⚠️  NOT: Gerçek zamanlı veri çekemez ama sistem yapısını test eder"
echo ""
source venv/bin/activate
python main.py --paper --testnet 2>&1 | head -50
