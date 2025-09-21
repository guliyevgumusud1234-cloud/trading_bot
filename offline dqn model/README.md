# Offline DQN Futures Trading

Tamamen çevrimdışı veriyle bir DQN ajanı eğitmek için minimal ama üretime hazır bir iskelet. Ajan, geçmiş fiyat verilerini kullanarak saatlik zaman diliminde (varsayılan) kısa/flat/uzun pozisyon kararları verir, işlem maliyetlerini dikkate alır ve sağlam bir geriye dönük değerlendirme yapar.

## Dizinin Yapısı

```
offline dqn model/
├── .venv/                 # Sanal ortam (oluşturuldu)
├── config/config.yaml     # Varsayılan hiperparametreler ve yollar
├── data/                  # CSV verinizi buraya koyun (ör. futures.csv)
├── src/trader/            # Paket kodu (özellikler, ortam, DQN, eğitim)
├── train.py               # Eğitim giriş noktası
├── requirements.txt       # Pip bağımlılıkları
└── README.md
```

## Kurulum

1. Sanal ortamı oluşturup aktif hale getirin:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
   (Proje kurulurken `numpy`, `pandas`, `torch`, `pyyaml`, `tqdm` zaten kuruldu.)

## Veri Beklentileri

- `config/config.yaml` içindeki `data.csv_path` alanına giden bir CSV sağlayın.
- En azından şu sütunları içermeli:
  - `timestamp` (ISO formatında tarih-saat, varsayılan `time_col`)
  - `close`
- Opsiyonel ama desteklenen sütunlar:
- `open`, `high`, `low`, `volume`

Modül varsayılan olarak kısa/orta/uzun vadeli momentum & volatilite (1h, 4h, 3 günlük), çoklu RSI, ATR, Bollinger bant genişliği, hacim volatilitesi ve OBV tabanlı trend göstergeleri ile trend/volatilite rejim skorları üretir.
- Dosya zaman sırasına göre sıralanmalı; değilse `time_col` üzerinden otomatik sıralanır.

## Konfigürasyon

- `config/config.yaml` dosyası eğitim parametriklerini tutar. Ana başlıklar:
  - `experiment`: isim, çıktı dizini, rastgelelik tohumu
  - `data`: veri yolu, pencere boyutu (`window`), train/val/test oranları, purged bar sayısı
  - `environment`: işlem maliyeti (bps), aksiyon seviyeleri, ödül ölçeği, işlem cooldown’ı, maksimum drawdown, trend filtresi (`trend_filter_abs_return`) ve volatilite tavanı (`volatility_ceiling`).
  - `model`: GRU tabanlı temsil + dueling MLP katmanları, dropout, opsiyonel dağılımsal başlık
  - `training`: öğrenme oranı, epsilon programı (warmup + cycle), hedef ağ güncellemesi, erken durdurma, Prioritized Replay parametreleri, öğrenme oranı scheduler’ı (`lr_scheduler`).
  - `log`: yazdırma, değerlendirme ve checkpoint sıklıkları

Tüm çıktı artefaktları `runs/<experiment>/<timestamp>/` altında saklanır (config kopyası, metrikler, checkpoint’ler, scaler vb.).

## Eğitimi Başlatma

1. Varsayılan yapılandırma `data/btcusdt_1h_ext.csv` dosyasını kullanır; kendi CSV’nizi koyuyorsanız `config/config.yaml` içindeki `data.csv_path` alanını güncelleyin.
2. Sanal ortam aktifken aşağıdaki komutu çalıştırın:
   ```bash
   PYTHONPATH=src python train.py --config config/config.yaml
   ```
3. Çıktı sonunda konsolda çalıştırılan run klasörü yazdırılır (ör. `runs/futures_dqn_1h/20250101_123000/`).

Eğitim esnasında:
- Her `print_freq` adımında konsola güncel epsilon / loss yazılır.
- Her `eval_interval` adımında doğrulama setinde Sharpe, getiri ve max DD ölçülür.
- En iyi doğrulama Sharpe skoruna sahip checkpoint `checkpoints/model_best.pt` olarak saklanır.
- Eğitim sonunda test sonuçları `test_summary.json` dosyasına yazılır.

## Çıktı Artefaktları

- `metrics.csv`: Adım bazında epsilon, loss ve doğrulama metrikleri.
- `checkpoints/`: Periyodik checkpoint’ler, en iyi model (`model_best.pt`), son model (`model_last.pt`).
- `artifacts/feature_scaler.npz`: Özellik z-score parametreleri.
- `artifacts/feature_names.json`: Kullanılan özelliklerin sıralı listesi.
- `config_used.yaml`: Çalışmada kullanılan konfigürasyonun tam kopyası.
- `test_summary.json`: Eğitim bitiminde test sonuçları.
- `scripts/evaluate_checkpoint.py`: Herhangi bir checkpoint’i train/val/test bölmesinde yeniden değerlendirmek için CLI.

## Modeli Kullanma

- En iyi modeli yüklemek için checkpoint’i `torch.load(".../model_best.pt", weights_only=False)` ile açıp `model_state_dict`’i `QNetwork` nesnesine yükleyin (GRU parametreleri otomatik oluşturulur).
- Özellik ölçekleyicisi ve özellik listesi artefakt dizininde bulunur.
- Canlı kullanımda son 128 bar için özellikleri üretip aynı ölçekleyiciyle z-score’layın ve modeli ileri besleyin. `inference.hysteresis_threshold` ve `cooldown_bars` konfigürasyonu canlı işlem katmanında uygulanabilir.
- `scripts/evaluate_checkpoint.py --config ... --checkpoint ... --split test` komutu ile modele dair Sharpe/getiri/drawdown raporu alabilirsiniz.
- `scripts/infer_signal.py --checkpoint ... --csv ...` ile güncel veri setinden tek adım aksiyon ve Q değerlerini alabilirsiniz.
- `scripts/ensemble_manage.py` çoklu checkpoint’i değerlendirip `ensemble_manifest.json` oluşturur.

## Otomasyon Scriptleri

- `scripts/setup_vm.sh`: Debian tabanlı VM’de Python sanal ortamı, pip bağımlılıkları ve log klasörlerini hazırlar. Tek sefer çalıştırmanız yeterli.
- `scripts/fetch_data.py`: `config/coins.yaml` içindeki core/extended semboller için 1 saatlik mumları çeker ve `data/market/` altına yazar.
- `scripts/train_loop.py`: Seçilen coin grubunu sırayla eğitir; her koşuyu `logs/train_*.log` dosyasına kaydeder ve özetini `logs/train_summary.jsonl` dosyasına ekler.
- `scripts/run_cycle.sh`: Tam bir döngü (`fetch_data` → `train_loop` → son `model_best.pt` ile toplu backtest) çalıştırır.
- `scripts/auto_loop.sh`: `run_cycle.sh` komutunu belirlediğiniz aralıkta (varsayılan 180 dk) sonsuz döngüde çalıştırır. `tmux` içinde başlatmanız önerilir.
- `scripts/backtest_top_coins.py`: Belirli bir checkpoint’i tüm coin listesi üzerinde test eder; raporu stdout’a yazar.
- `scripts/notify.py`: `ALERT_WEBHOOK_URL` tanımlıysa başarısız otomasyon döngülerinde webhook’a bildirim gönderir.
- `scripts/prune_runs.py`: Eski run klasörlerini temizlemek için yardımcı skript.
- `scripts/ensemble_manage.py`: Çoklu checkpoint’i değerlendirip en iyi modelleri seçer.
- `scripts/infer_signal.py`: Tek adım sinyal çıkarımı yapar (canlı prototip).

### Örnek VM Kurulumu ve Sürekli Eğitim

```bash
git clone <repo>
cd offline\ dqn\ model
bash scripts/setup_vm.sh             # sanal ortam ve bağımlılık kurulumunu yapar
source .venv/bin/activate
tmux new -s auto
  # tmux içinde aşağıdakini çalıştırın
  INTERVAL_MINUTES=240 TRAIN_EXTENDED=1 EXTENDED_WINDOW_DAYS=30 bash scripts/auto_loop.sh
```

`tmux` oturumundan `Ctrl+b` ardından `d` ile çıkabilirsiniz; eğitim arka planda devam eder. Geri dönmek için `tmux attach -t auto`.
`ALERT_WEBHOOK_URL` ortam değişkeni tanımlıysa, `auto_loop.sh` başarısız döngüler için `scripts/notify.py` aracılığıyla webhook’a mesaj gönderir.

### Log ve Metrik İzleme

- `logs/train_summary.jsonl`: Her eğitim koşusunun zaman damgası, sembol, durum ve run klasörü.
- `logs/train_<symbol>_*.log`: İlgili eğitim sürecinin tam stdout/stderr’i.
- `runs/<symbol>.../metrics.csv`: Eğitim sırasında kaydedilen metrikler; `tail -f` ile takip edilebilir.
- `runs/<symbol>.../metrics.jsonl`: JSON formatında lr/Sharpe/epsilon kayıtları.
- `logs/auto_loop.log`: `auto_loop.sh` tarafından çalıştırılan döngülerin zaman çizelgesi.
- `logs/fetch_summary.jsonl`: Veri çekme işlemlerinin özet kayıtları.

### Veri ve Coin Yapılandırması

- `config/coins.yaml`: core ve extended listelerini yönetir; `scripts/fetch_data.py` ve diğer scriptler burayı kullanır.
- Veri klasörü varsayılan olarak `data/market/<symbol>_1h.csv`; scriptler mevcut dosyaları sıralı şekilde günceller.
- `scripts/prune_runs.py` ve `cleanup` rutinleri disk alanını yönetmek için kullanılabilir.

## Ek Özelleştirmeler

- Öğrenme takvimi, epsilon süresi, replay kapasitesi vb. parametreleri `config.yaml` üzerinden güncelleyebilirsiniz.
- `src/trader/model.py` içinde GRU tabanlı bir model veya dağılımsal başlık eklemek için MLP yapısını genişletebilirsiniz.
- Risk yönetimi overlay’i (maks DD, trade limiti) canlı sistem katmanında konfigüre edilmek üzere ayrı tutulmuştur.

> **Not:** Bu repo yalnızca teknik amaçlıdır; herhangi bir yatırım tavsiyesi içermez. Gerçek sermaye ile işlem öncesinde kapsamlı doğrulama, slippage ve maliyet testleri, risk kontrolleri yapılmalıdır.
