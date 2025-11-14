# 📦 Bağımlılık Kurulum Rehberi

## Sistem Gereksinimleri

- **Python**: 3.11 (zorunlu)
- **İşletim Sistemi**: macOS, Linux, veya Windows
- **RAM**: Minimum 8GB
- **Disk**: 10GB+ boş alan

---

## Kurulum Seçenekleri

### Seçenek 1: Docker ile Kurulum (ÖNERİLEN) ✅

Docker kullanırsanız, tüm bağımlılıklar otomatik yüklenir!

```bash
cd crypto_trading_bot

# Docker ile build
docker-compose build

# Tamamlandı! Tüm bağımlılıklar container içinde hazır
```

**Avantajları:**
- ✅ Tek komutla tüm bağımlılıklar yüklenir
- ✅ TA-Lib otomatik kurulur
- ✅ Sistem bağımsız çalışır
- ✅ Production ortamı ile aynı

---

### Seçenek 2: Manuel Python Kurulumu

#### macOS (Sizin sisteminiz)

```bash
# 1. Homebrew ile sistem bağımlılıklarını yükle
brew install ta-lib
brew install python@3.11

# 2. Virtual environment oluştur
cd crypto_trading_bot
python3.11 -m venv venv
source venv/bin/activate

# 3. Pip'i güncelle
pip install --upgrade pip setuptools wheel

# 4. Tüm Python bağımlılıklarını yükle
pip install -r requirements.txt

# Bu işlem 5-10 dakika sürebilir
```

#### Linux (Ubuntu/Debian)

```bash
# 1. Sistem bağımlılıkları
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
sudo apt-get install -y build-essential wget

# 2. TA-Lib kaynak koddan kur
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib*

# 3. Virtual environment
cd crypto_trading_bot
python3.11 -m venv venv
source venv/bin/activate

# 4. Python paketleri
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows

```powershell
# 1. Python 3.11'i python.org'dan indir ve kur

# 2. TA-Lib için önceden derlenmiş wheel indir
# https://github.com/cgohlke/talib-build/releases
# TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl dosyasını indir

# 3. PowerShell'de:
cd crypto_trading_bot
python -m venv venv
.\venv\Scripts\Activate

pip install --upgrade pip

# 4. İndirilen TA-Lib wheel'i kur
pip install path\to\TA_Lib-0.4.28-cp311-cp311-win_amd64.whl

# 5. Diğer bağımlılıkları kur
pip install -r requirements.txt
```

---

## Kurulum Doğrulama

### Tüm paketlerin yüklendiğini kontrol et:

```bash
# Virtual environment'ı aktif et (Docker kullanmıyorsanız)
source venv/bin/activate  # macOS/Linux
# veya
.\venv\Scripts\Activate  # Windows

# Python'u başlat
python
```

```python
# Python konsolunda test et:
import numpy
import pandas
import ccxt
import lightgbm
import xgboost
import catboost
import torch
import stable_baselines3
import talib  # Bu önemli!
import sqlalchemy
import redis
import aiohttp

print("✅ Tüm bağımlılıklar başarıyla yüklendi!")
```

Hata yoksa, kurulum başarılı! 🎉

---

## Olası Sorunlar ve Çözümler

### ❌ TA-Lib kurulamıyor

**macOS:**
```bash
# M1/M2 Mac için:
arch -arm64 brew install ta-lib
export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
pip install TA-Lib
```

**Linux:**
```bash
# Tekrar kaynak koddan kur:
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
pip install TA-Lib
```

### ❌ PyTorch kurulamıyor

```bash
# CPU only versiyon (daha küçük):
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### ❌ Permission hatası

```bash
# pip install'da --user kullan:
pip install --user -r requirements.txt
```

### ❌ Memory hatası

```bash
# Paketleri tek tek kur:
pip install numpy pandas scipy
pip install ccxt websocket-client
pip install lightgbm xgboost catboost
# ... devam et
```

---

## Kurulum Sonrası Test

```bash
# Ana dizinde:
cd crypto_trading_bot

# Test komutları:
python -c "from data.fetcher import DataFetcher; print('✅ Data modülü çalışıyor')"
python -c "from models.strategy_trend import TrendFollowingStrategy; print('✅ Strategy modülü çalışıyor')"
python -c "from risk.risk_manager import RiskManager; print('✅ Risk modülü çalışıyor')"
python -c "from utils.logger import get_logger; print('✅ Utils modülü çalışıyor')"

echo "🎉 Tüm modüller başarıyla yüklendi!"
```

---

## Disk Kullanımı

Kurulum sonrası yaklaşık disk kullanımı:
- **Sadece Python paketleri**: ~2.5 GB
- **Docker image**: ~3 GB
- **Toplam (Docker dahil)**: ~5-6 GB

---

## Güncelleme

Bağımlılıkları güncellemek için:

```bash
# Virtual environment aktif olmalı
pip install --upgrade -r requirements.txt
```

---

## Yardım

Sorun yaşıyorsanız:
1. Python versiyonunu kontrol edin: `python --version` (3.11 olmalı)
2. Pip versiyonunu güncelleyin: `pip install --upgrade pip`
3. Virtual environment kullandığınızdan emin olun
4. Docker kullanmayı deneyin (en kolay yol)

---

**Önemli Not:** Docker kullanıyorsanız bu adımların hiçbirine gerek yok! 
`docker-compose build` komutu her şeyi halleder. 🚀

