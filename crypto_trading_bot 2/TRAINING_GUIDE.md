# 📚 Model Training Rehberi

## 🎯 Training Nedir?

Modelleri geçmiş verilerle eğiterek gelecekteki fiyat hareketlerini tahmin edebilmelerini sağlar. Eğitilmemiş modeller rastgele tahminler yapar, eğitilmiş modeller ise geçmişteki pattern'leri öğrenerek daha iyi sinyaller üretir.

## 📋 Training Süreci (7 Adım)

### 1️⃣ Historical Data Loading
- 2-3 yıllık OHLCV verisi çekilir
- Tüm semboller için (BTC, ETH, BNB, SOL, ARB, MATIC, AVAX, LINK)
- Tüm timeframe'ler için (5m, 15m, 1h, 4h, 1d)
- **Süre:** 10-30 dakika

### 2️⃣ Feature Engineering
- 200+ özellik hesaplanır:
  - **Price features:** Returns, volatility, ATR, Bollinger Bands
  - **Volume features:** VWAP, volume profile, OBV
  - **Technical indicators:** EMA, MACD, RSI, ADX, Supertrend
  - **Microstructure:** Bid-ask spread, order book imbalance
  - **Derivatives:** Funding rate, open interest, liquidations
  - **Market regime:** Trend strength, volatility regime
  - **Correlation:** BTC correlation, cross-asset correlation
  - **Time-based:** Hour, day of week, seasonality
- **Süre:** 30-60 dakika

### 3️⃣ Train/Val/Test Split
- **Train:** 2 yıl (öğrenme için)
- **Validation:** 3 ay (hyperparameter tuning için)
- **Test:** 3 ay (final performans testi için)

### 4️⃣ Strategy Training
Her strateji ayrı ayrı eğitilir:

- **Trend Following (LightGBM)**
  - Gradient boosting model
  - EMA crossovers, ADX, Supertrend features
  - **Süre:** 10-20 dakika

- **Mean Reversion (XGBoost)**
  - Extreme gradient boosting
  - Bollinger Bands, Z-score, RSI features
  - **Süre:** 10-20 dakika

- **Momentum (CatBoost)**
  - Categorical boosting
  - Consolidation detection, volume breakout features
  - **Süre:** 10-20 dakika

- **Arbitrage (Statistical)**
  - Funding rate extremes detection
  - **Süre:** 5 dakika

- **Deep RL (PPO)**
  - Proximal Policy Optimization
  - 1M timesteps training
  - **Süre:** 2-4 saat (en uzun!)

### 5️⃣ Walk-Forward Validation
- 90 gün train, 30 gün test
- Sliding window ile test
- Overfitting kontrolü
- **Süre:** 30-60 dakika

### 6️⃣ Model Saving
- Tüm modeller `models/trained/` klasörüne kaydedilir
- Her strateji için ayrı model dosyası

### 7️⃣ Training Report
- Performans metrikleri (Sharpe, Sortino, Win Rate, vb.)
- `models/trained/training_report.yaml` oluşturulur

## 🚀 Nasıl Çalıştırılır?

### Temel Kullanım

```bash
# Virtual environment'ı aktif et
source venv/bin/activate

# Training'i başlat
python train.py
```

Bu komut:
- 2022-01-01'den bugüne kadar veri çeker
- Tüm stratejileri eğitir
- Hyperparameter optimization yapar
- Modelleri kaydeder

### Özelleştirilmiş Kullanım

```bash
# Belirli tarih aralığı
python train.py --start-date 2023-01-01 --end-date 2024-01-01

# Hyperparameter optimization olmadan (daha hızlı)
python train.py --no-optimize

# Sadece eğit, kaydetme (test için)
python train.py --no-save

# Kombinasyon
python train.py --start-date 2023-01-01 --no-optimize
```

## ⏱️ Tahmini Süre

| Adım | Süre |
|------|------|
| Veri çekme | 10-30 dakika |
| Feature engineering | 30-60 dakika |
| Model training (LightGBM/XGBoost/CatBoost) | 30-60 dakika |
| Model training (RL) | 2-4 saat |
| Walk-forward validation | 30-60 dakika |
| **TOPLAM** | **3-8 saat** |

## 📦 Gereksinimler

- ✅ Internet bağlantısı (veri çekmek için)
- ✅ Yeterli disk alanı (~5-10 GB)
- ✅ Yeterli RAM (8+ GB önerilir)
- ✅ Bot çalışıyor olmasına gerek yok (training ayrı çalışır)

## 📊 Training Sonrası

Training tamamlandıktan sonra:

1. **Modeller kaydedilir:** `models/trained/` klasöründe
2. **Rapor oluşturulur:** `models/trained/training_report.yaml`
3. **Bot otomatik kullanır:** Bir sonraki başlatmada eğitilmiş modeller yüklenir

## 🔍 Training İlerlemesini İzleme

Training sırasında loglar `logs/trading.log` dosyasına yazılır:

```bash
# Canlı logları izle
tail -f logs/trading.log

# Sadece training loglarını filtrele
tail -f logs/trading.log | grep -E "(Training|training|TRAINING)"
```

## ⚠️ Önemli Notlar

1. **İlk training uzun sürer:** 3-8 saat arası
2. **RL training en uzun:** 2-4 saat sürebilir
3. **Internet gerekli:** Veri çekmek için
4. **Disk alanı:** ~5-10 GB gerekli
5. **Training sırasında bot çalışabilir:** Ama gerekli değil

## 🎯 Hızlı Test (Küçük Veri Seti)

Eğer hızlı test etmek isterseniz:

```bash
# Son 6 ay veri ile (daha hızlı)
python train.py --start-date 2024-05-01 --no-optimize
```

Bu yaklaşık 1-2 saat sürer.

## 📈 Training Sonrası Performans

Training tamamlandıktan sonra bot'u yeniden başlatın:

```bash
# Bot'u durdur
tmux kill-session -t trading_bot

# Yeniden başlat
./start_background.sh
```

Bot artık eğitilmiş modelleri kullanacak ve daha iyi sinyaller üretecektir!

