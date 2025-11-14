# Docker Kurulum ve Başlatma Rehberi

## 1. Docker Kurulumu

### macOS için:

**Seçenek 1: Docker Desktop (Önerilen)**
1. Docker Desktop'ı indirin: https://www.docker.com/products/docker-desktop
2. `.dmg` dosyasını açın ve Docker'ı Applications klasörüne sürükleyin
3. Docker Desktop'ı başlatın
4. Docker'ın çalıştığını doğrulayın: `docker ps`

**Seçenek 2: Homebrew**
```bash
brew install --cask docker
```

## 2. Docker Kurulumunu Test Etme

```bash
# Docker versiyonunu kontrol et
docker --version

# Docker Compose versiyonunu kontrol et
docker compose version

# Docker'ın çalıştığını test et
docker ps
```

## 3. Sistem Başlatma

### Adım 1: .env Dosyasını Kontrol Et
`.env` dosyasında gerekli değişkenlerin olduğundan emin ol:
- `DB_PASSWORD`
- `BINANCE_API_KEY` (opsiyonel, paper trading için gerekli değil)
- `BINANCE_API_SECRET` (opsiyonel)
- `TELEGRAM_BOT_TOKEN` (opsiyonel)
- `TELEGRAM_CHAT_ID` (opsiyonel)
- `GRAFANA_PASSWORD`
- `PGADMIN_PASSWORD` (opsiyonel)

### Adım 2: Docker Compose ile Sistemleri Başlat

```bash
cd crypto_trading_bot

# Tüm servisleri başlat
docker compose up -d

# Logları izle
docker compose logs -f

# Sadece belirli servisleri başlat
docker compose up -d timescaledb redis
```

### Adım 3: Servis Durumunu Kontrol Et

```bash
# Tüm container'ların durumunu kontrol et
docker compose ps

# Her servisin loglarını kontrol et
docker compose logs trading_bot
docker compose logs timescaledb
docker compose logs redis
docker compose logs prometheus
docker compose logs grafana
```

## 4. Servis Erişim Bilgileri

- **TimescaleDB**: `localhost:5432`
- **Redis**: `localhost:6379`
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / [GRAFANA_PASSWORD])
- **PgAdmin**: http://localhost:5050 (opsiyonel)

## 5. Sistem Durdurma

```bash
# Tüm servisleri durdur
docker compose down

# Servisleri durdur ve volume'ları sil
docker compose down -v
```

## 6. Sorun Giderme

### Docker Desktop çalışmıyor
- Docker Desktop'ı başlatın
- System Preferences > Security & Privacy'de Docker'a izin verin

### Port çakışması
- 5432, 6379, 9090, 3000 portlarının kullanılmadığından emin olun
- `docker-compose.yml` dosyasında portları değiştirebilirsiniz

### Container'lar başlamıyor
```bash
# Logları kontrol et
docker compose logs

# Container'ları yeniden oluştur
docker compose up -d --build --force-recreate
```

