# Cloud / VM Quickstart

Bu belge, Google Cloud benzeri bir VM üzerinde projeyi 7/24 çalıştırmak için gereken adımları özetler.

## 1. Makine Hazırlığı
- Debian 12 tabanlı, en az 2 vCPU / 7 GB RAM (c4-standard-2 benzeri) ve 30 GB disk yeterlidir.
- Gerekirse `sudo apt-get update && sudo apt-get install -y git python3 python3-venv tmux`.

## 2. Repo ve Kurulum Scripti
```bash
git clone <repo>
cd offline\ dqn\ model
bash scripts/setup_vm.sh
source .venv/bin/activate
```

## 3. Veri Çekme ve İlk Eğitim
```bash
python scripts/fetch_data.py --group all
python scripts/train_loop.py --group core
```
- Çıktılar `data/market/`, `logs/`, `runs/` klasörlerine kaydedilir.

## 4. Sürekli Döngü (tmux önerilir)
```bash
# tmux içinde
ALERT_WEBHOOK_URL=https://example.com/webhook \
TRAIN_EXTENDED=1 EXTENDED_WINDOW_DAYS=30 INTERVAL_MINUTES=240 bash scripts/auto_loop.sh
```
- Döngü, her 4 saatte bir `run_cycle.sh` (veri çek + eğitim + toplu backtest) çalıştırır. `TRAIN_EXTENDED=1` ayarı extended coin grubunu da periyodik olarak eğitir. `ALERT_WEBHOOK_URL` tanımlıysa başarısız döngüler `scripts/notify.py` tarafından belirtilen webhook’a bildirilir.

## 5. Loglar
- `logs/auto_loop.log`: Her döngünün başlangıç/bitiş kayıtları.
- `logs/train_<symbol>_*.log`: Eğitim sürecinin tüm çıktısı.
- `logs/train_summary.jsonl`: Sembollere göre özet kayıt.
- `logs/fetch_summary.jsonl`: Veri çekme sonuçları.

## 6. Backtest ve Raporlama
Son `model_best.pt` üzerinde tüm coinlerde rapor almak:
```bash
latest_ckpt=$(tail -n1 logs/train_summary.jsonl | python -c 'import sys,json,os;data=json.loads(sys.stdin.read());print(data["info"])')
python scripts/backtest_top_coins.py --checkpoint "$latest_ckpt" --coins-config config/coins.yaml --data-dir data
```

En iyi modelleri seçmek:
```bash
python scripts/ensemble_manage.py --config config/config.yaml --checkpoints runs --top 5
```

Sinyal gözlemlemek:
```bash
python scripts/infer_signal.py --checkpoint "$latest_ckpt" --csv data/market/btcusdt_1h.csv --position 0
```

## 7. Güvenlik / Erişim
- API anahtarlarını `.env` veya Secret Manager’da saklayın (gerekiyorsa).
- Dashboard için HTTP servisi açacaksanız firewall kurallarını güncellemeyi unutmayın.

Bu adımlarla VM üzerinde eğitim döngüsü otomatik çalışır; tmux veya log dosyaları üzerinden süreci izleyebilirsiniz.
