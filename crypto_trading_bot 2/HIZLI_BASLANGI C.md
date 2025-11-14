# 🚀 HIZLI BAŞLANGIÇ REHBERİ

## ✅ Kurulum Tamamlandı! Şimdi Ne Yapmalı?

Sistemi çalıştırmak için gereken **5 adım**:

---

## 1️⃣ Binance API Anahtarı Alma (10 dakika)

### Adım 1: Binance Hesabı
1. https://www.binance.com/ adresine gidin
2. Hesap oluşturun (eğer yoksa)
3. KYC doğrulamasını tamamlayın

### Adım 2: API Anahtarı Oluşturma

**⚠️ ÖNEMLİ: ÖNCE TESTNET İLE BAŞLAYIN!**

#### Testnet (Önerilen - Gerçek para risk yok):
1. https://testnet.binancefuture.com/ adresine gidin
2. GitHub ile giriş yapın
3. API Key oluşturun
4. ✅ **Enable Reading** - AÇIK
5. ✅ **Enable Futures** - AÇIK
6. ❌ **Enable Withdrawals** - KAPALI (GÜVENLİK!)

#### Ana Hesap (Sadece testnet başarılı olduktan sonra):
1. Binance → Profil → API Management
2. "Create API" butonuna tıklayın
3. Güvenlik doğrulamasını yapın
4. İzinleri ayarlayın:
   - ✅ **Enable Reading** - AÇIK
   - ✅ **Enable Futures** - AÇIK
   - ❌ **Enable Withdrawals** - KAPALI (ÇOK ÖNEMLİ!)
5. IP Whitelist ekleyin (opsiyonel ama önerilen)

### Adım 3: Anahtarları Kaydedin
```
API Key: xxxxxxxxxxxxxxxxxxxxxxx
Secret Key: yyyyyyyyyyyyyyyyyyyyyyy
```

**🔒 GÜVENLİK NOTLARI:**
- Bu anahtarları ASLA paylaşmayın
- Withdrawal iznini ASLA açmayın
- IP whitelist kullanın
- 2FA'yı aktif edin

---

## 2️⃣ Environment Variables Ayarlama (5 dakika)

### Adım 1: .env Dosyası Oluşturma

```bash
cd /Users/huseynli99/Downloads/cursor/crypto_trading_bot
cp .env.example .env
nano .env  # veya VSCode/TextEdit ile açın
```

### Adım 2: Gerekli Bilgileri Doldurma

```bash
# ==============================================
# BINANCE API (ZORUNLU)
# ==============================================
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here
BINANCE_TESTNET=true  # İlk başta true yapın!

# ==============================================
# DATABASE (Docker kullanıyorsanız değiştirmeyin)
# ==============================================
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=postgres
DB_PASSWORD=secure_password_123  # Güçlü şifre seçin

# ==============================================
# REDIS (Docker kullanıyorsanız değiştirmeyin)
# ==============================================
REDIS_HOST=localhost
REDIS_PORT=6379

# ==============================================
# TELEGRAM ALERTS (Opsiyonel ama önerilen)
# ==============================================
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# ==============================================
# MONITORING (Docker kullanıyorsanız)
# ==============================================
GRAFANA_PASSWORD=admin123
GRAFANA_USER=admin

# ==============================================
# SYSTEM SETTINGS
# ==============================================
ENVIRONMENT=development
LOG_LEVEL=INFO
TIMEZONE=UTC
```

---

## 3️⃣ Telegram Bot Kurulumu (10 dakika - Opsiyonel)

### Neden Telegram?
- Gerçek zamanlı trade bildirimler
- Risk uyarıları
- Günlük özet raporlar
- Sistem hataları anında bildirim

### Kurulum:

#### Adım 1: Bot Oluşturma
1. Telegram'da **@BotFather** bulun
2. `/newbot` komutunu gönderin
3. Bot için isim seçin (örn: "My Trading Bot")
4. Kullanıcı adı seçin (örn: "my_trading_bot")
5. **Token**'ı kaydedin: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`

#### Adım 2: Chat ID Alma
1. Bot'unuza mesaj gönderin (Start yapın)
2. https://api.telegram.org/bot<TOKEN>/getUpdates adresine gidin
   (TOKEN yerine bot token'ınızı yazın)
3. `"chat":{"id":123456789}` değerini bulun ve kaydedin

#### Adım 3: .env'e Ekleyin
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

---

## 4️⃣ Database Kurulumu

### Seçenek A: Docker ile (ÖNERİLEN - Kolay)

```bash
cd /Users/huseynli99/Downloads/cursor/crypto_trading_bot

# Database container'ı başlat
docker-compose up -d timescaledb redis

# Database'in hazır olmasını bekleyin (30 saniye)
sleep 30

# Database başarılı mı kontrol edin
docker-compose ps
```

### Seçenek B: Manuel Kurulum (macOS)

```bash
# PostgreSQL yükle
brew install postgresql@15
brew services start postgresql@15

# Redis yükle
brew install redis
brew services start redis

# Database oluştur
createdb trading_db

# Schema'yı yükle
psql trading_db < database/init.sql
```

---

## 5️⃣ İlk Çalıştırma (Paper Trading)

### Adım 1: Virtual Environment Aktif Et
```bash
cd /Users/huseynli99/Downloads/cursor/crypto_trading_bot
source venv/bin/activate
```

### Adım 2: Yapılandırmayı Kontrol Et
```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('🔍 Yapılandırma Kontrolü:\n')
print('✅ API Key:', 'VAR' if os.getenv('BINANCE_API_KEY') else '❌ YOK')
print('✅ Secret:', 'VAR' if os.getenv('BINANCE_API_SECRET') else '❌ YOK')
print('✅ Testnet:', os.getenv('BINANCE_TESTNET', 'false'))
print('✅ Telegram:', 'VAR' if os.getenv('TELEGRAM_BOT_TOKEN') else '❌ YOK (opsiyonel)')
print('\n✅ Yapılandırma hazır!' if os.getenv('BINANCE_API_KEY') else '\n❌ .env dosyasını düzenleyin!')
"
```

### Adım 3: Database Bağlantısını Test Et
```bash
python -c "
from utils.database import DatabaseManager
try:
    db = DatabaseManager()
    print('✅ Database bağlantısı başarılı!')
except Exception as e:
    print('❌ Database bağlantı hatası:', e)
    print('💡 Docker ile database başlatın: docker-compose up -d timescaledb')
"
```

### Adım 4: Exchange Bağlantısını Test Et
```bash
python -c "
from execution.exchange_interface import ExchangeInterface
import os
from dotenv import load_dotenv
load_dotenv()

try:
    exchange = ExchangeInterface(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    )
    
    # API latency test
    latency = exchange.get_api_latency()
    print(f'✅ Exchange bağlantısı başarılı!')
    print(f'📊 API Latency: {latency:.2f}ms')
    
    # Balance check
    balance = exchange.fetch_balance()
    print(f'💰 USDT Balance: {balance.get(\"USDT\", {}).get(\"free\", 0):.2f}')
    
except Exception as e:
    print(f'❌ Exchange bağlantı hatası: {e}')
    print('💡 API anahtarlarını kontrol edin!')
"
```

### Adım 5: Paper Trading Başlat! 🚀

```bash
# Paper trading modu (gerçek para yok)
python main.py --paper --testnet

# Logları izleyin (başka bir terminal'de)
tail -f logs/trading.log
```

---

## 📊 İzleme ve Monitoring

### Log Dosyaları
```bash
# Ana trading log
tail -f logs/trading.log

# Risk log
tail -f logs/risk.log

# Execution log
tail -f logs/execution.log

# Hata log
tail -f logs/error.log
```

### Grafana Dashboard (Docker ile)
```bash
# Tüm stack'i başlat
docker-compose up -d

# Grafana'ya eriş
# URL: http://localhost:3000
# Kullanıcı: admin
# Şifre: .env dosyasındaki GRAFANA_PASSWORD
```

---

## ⚠️ İLK ÇALIŞTIRMA ÖNCESİ KONTROL LİSTESİ

### Zorunlu:
- [ ] ✅ Binance API anahtarları alındı (TESTNET!)
- [ ] ✅ .env dosyası oluşturuldu ve dolduruldu
- [ ] ✅ BINANCE_TESTNET=true yapıldı
- [ ] ✅ Database kuruldu (Docker veya manuel)
- [ ] ✅ Virtual environment aktif
- [ ] ✅ Exchange bağlantısı test edildi

### Önerilen:
- [ ] ✅ Telegram bot kuruldu
- [ ] ✅ Güçlü database şifresi seçildi
- [ ] ✅ API withdrawal izni kapalı
- [ ] ✅ IP whitelist yapılandırıldı
- [ ] ✅ 2FA aktif edildi

### İleri Seviye:
- [ ] ⚙️ Modeller eğitildi (`python train.py`)
- [ ] ⚙️ Backtesting yapıldı
- [ ] ⚙️ Risk limitleri özelleştirildi
- [ ] ⚙️ Grafana dashboard kuruldu

---

## 🎯 İlk Paper Trading Testi (7 Gün)

```bash
# Gün 1-2: Sistem stabilitesi
- Bot'un çalıştığından emin olun
- Log dosyalarını kontrol edin
- Hata olmadığını doğrulayın

# Gün 3-4: Performans izleme
- Trade sinyallerini gözlemleyin
- Risk limitlerinin çalıştığını kontrol edin
- Telegram bildirimlerini inceleyin

# Gün 5-7: Optimizasyon
- Strategy performansını analiz edin
- Risk parametrelerini ayarlayın
- Sonuçları kaydedin

# BAŞARILI İSE → Testnet ile devam
# SORUN VARSA → Ayarları düzeltin ve tekrar test edin
```

---

## 🆘 Sık Karşılaşılan Sorunlar

### ❌ "Invalid API Key"
```bash
# Çözüm:
1. API anahtarlarını kontrol edin
2. .env dosyasında tırnak işareti KULLANMAYIN
3. Testnet için testnet anahtarı, main için main anahtarı kullanın
```

### ❌ "Database connection failed"
```bash
# Çözüm:
# Docker ile:
docker-compose up -d timescaledb
sleep 30

# Manuel ile:
brew services start postgresql@15
```

### ❌ "Module not found"
```bash
# Çözüm:
source venv/bin/activate
pip install -r requirements.txt
```

### ❌ "Permission denied"
```bash
# Çözüm:
chmod +x main.py train.py
```

---

## 📞 Yardım ve Destek

### Logları Kontrol Edin
```bash
# Son 100 satır
tail -n 100 logs/trading.log

# Hataları filtrele
grep "ERROR" logs/trading.log

# Belirli bir tarihteki loglar
grep "2024-01-15" logs/trading.log
```

### Debug Modu
```bash
# Daha detaylı loglar için
LOG_LEVEL=DEBUG python main.py --paper
```

### Test Komutları
```bash
# Database test
python -c "from utils.database import DatabaseManager; print('OK')"

# Exchange test
python -c "from execution.exchange_interface import ExchangeInterface; print('OK')"

# Strategy test
python -c "from models.meta_strategy import MetaOrchestrator; print('OK')"
```

---

## 🎓 Öğrenme Kaynakları

### Dokümantasyon
- `README.md` - Genel bakış
- `DEPLOYMENT_GUIDE.md` - Detaylı kurulum
- `PROJECT_SUMMARY.md` - Sistem özeti
- `KURULUM.md` - Türkçe kurulum

### Örnek Komutlar
```bash
# Paper trading (test)
python main.py --paper --testnet

# Model training
python train.py --start-date 2022-01-01

# Backtesting
python -c "from backtest.backtester import RealisticBacktester; print('Backtest ready')"

# Docker ile tüm sistem
docker-compose up -d
```

---

## ⏭️ Sonraki Adımlar

### Kısa Vadede (1-2 Hafta):
1. ✅ Paper trading ile 7 gün test
2. ✅ Sonuçları analiz et
3. ✅ Risk parametrelerini optimize et
4. ✅ Telegram bildirimleri kur

### Orta Vadede (1 Ay):
1. ⚙️ Modelleri gerçek veri ile eğit
2. ⚙️ Backtesting yap
3. ⚙️ Walk-forward validation
4. ⚙️ Grafana dashboard'u kur

### Uzun Vadede (1+ Ay):
1. 🎯 Küçük sermaye ile live test ($100-500)
2. 🎯 Performansı günlük izle
3. 🎯 Başarılı olursa kademeli artır
4. 🎯 Otomatik retraining kur

---

## 🎉 BAŞARILAR!

Sisteminiz artık hazır! İlk paper trading testinizde başarılar dileriz.

**Unutmayın:**
- 🧪 İlk hafta SADECE test
- 💰 Küçük başlayın
- 📊 Aktif izleyin
- 🎓 Sürekli öğrenin
- 🛡️ Risk yönetimi #1 öncelik

**🚀 Let's trade!** 📈

