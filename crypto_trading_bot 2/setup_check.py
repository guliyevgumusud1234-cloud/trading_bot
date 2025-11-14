#!/usr/bin/env python3
"""
Sistem Kurulum Kontrol Script'i
Bu script sisteminizin çalışmaya hazır olup olmadığını kontrol eder.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_check(name, status, message=""):
    icon = "✅" if status else "❌"
    print(f"{icon} {name:30s} {message}")
    return status

def check_env_file():
    """Environment dosyasını kontrol et"""
    print_header("1. ENVIRONMENT VARIABLES KONTROLÜ")
    
    env_exists = os.path.exists('.env')
    print_check(".env dosyası", env_exists)
    
    if not env_exists:
        print("\n💡 Çözüm:")
        print("   cp .env.example .env")
        print("   nano .env  # ve kendi değerlerinizi girin")
        return False
    
    load_dotenv()
    
    checks = {
        "BINANCE_API_KEY": os.getenv('BINANCE_API_KEY'),
        "BINANCE_API_SECRET": os.getenv('BINANCE_API_SECRET'),
        "DB_PASSWORD": os.getenv('DB_PASSWORD'),
    }
    
    all_good = True
    for key, value in checks.items():
        exists = value and value != f"your_{key.lower()}_here"
        print_check(key, exists, "AYARLI" if exists else "EKSİK")
        all_good = all_good and exists
    
    testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    print_check("Testnet modu", testnet, "AÇIK (güvenli)" if testnet else "KAPALI")
    
    telegram = bool(os.getenv('TELEGRAM_BOT_TOKEN'))
    print_check("Telegram alerts", telegram, "VAR" if telegram else "YOK (opsiyonel)")
    
    return all_good

def check_dependencies():
    """Python paketlerini kontrol et"""
    print_header("2. PYTHON PAKETLER KONTROLÜ")
    
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('ccxt', 'CCXT (Exchange)'),
        ('lightgbm', 'LightGBM (ML)'),
        ('torch', 'PyTorch (Deep Learning)'),
        ('stable_baselines3', 'Stable-Baselines3 (RL)'),
        ('sqlalchemy', 'SQLAlchemy (Database)'),
        ('redis', 'Redis'),
        ('fastapi', 'FastAPI'),
    ]
    
    all_good = True
    for module, name in packages:
        try:
            __import__(module)
            print_check(name, True, "YÜKLÜ")
        except ImportError:
            print_check(name, False, "EKSİK")
            all_good = False
    
    # TA-Lib opsiyonel
    try:
        import talib
        print_check("TA-Lib (opsiyonel)", True, "YÜKLÜ")
    except ImportError:
        print_check("TA-Lib (opsiyonel)", False, "YOK (brew install ta-lib)")
    
    return all_good

def check_project_structure():
    """Proje yapısını kontrol et"""
    print_header("3. PROJE YAPISINI KONTROLÜ")
    
    important_dirs = [
        'config', 'data', 'models', 'risk', 'execution', 
        'backtest', 'utils', 'logs'
    ]
    
    all_good = True
    for dir_name in important_dirs:
        exists = os.path.isdir(dir_name)
        print_check(f"{dir_name}/ dizini", exists)
        all_good = all_good and exists
    
    important_files = [
        'main.py', 'train.py', 'requirements.txt',
        'config/config.yaml', 'config/strategy_params.yaml'
    ]
    
    for file_name in important_files:
        exists = os.path.exists(file_name)
        print_check(file_name, exists)
        all_good = all_good and exists
    
    return all_good

def check_database():
    """Database bağlantısını kontrol et"""
    print_header("4. DATABASE BAĞLANTISI")
    
    try:
        from utils.database import DatabaseManager
        load_dotenv()
        
        # Bağlantıyı dene
        db = DatabaseManager(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'trading_db'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        print_check("Database bağlantısı", True, "BAŞARILI")
        return True
        
    except Exception as e:
        print_check("Database bağlantısı", False, f"HATA: {str(e)[:40]}")
        print("\n💡 Çözüm:")
        print("   docker-compose up -d timescaledb")
        print("   # veya")
        print("   brew services start postgresql@15")
        return False

def check_exchange():
    """Exchange bağlantısını kontrol et"""
    print_header("5. EXCHANGE (BINANCE) BAĞLANTISI")
    
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or api_key == 'your_api_key_here':
        print_check("API Key", False, "AYARLANMAMIŞ")
        print("\n💡 Binance API anahtarı gerekli!")
        print("   Testnet: https://testnet.binancefuture.com/")
        print("   Ana hesap: https://www.binance.com/ (API Management)")
        return False
    
    try:
        from execution.exchange_interface import ExchangeInterface
        
        testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        exchange = ExchangeInterface(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        latency = exchange.get_api_latency()
        print_check("Exchange bağlantısı", True, f"BAŞARILI ({latency:.0f}ms)")
        
        try:
            balance = exchange.fetch_balance()
            usdt = balance.get('USDT', {}).get('free', 0)
            print_check("Balance kontrolü", True, f"${usdt:.2f} USDT")
        except:
            print_check("Balance kontrolü", False, "İzin hatası?")
        
        return True
        
    except Exception as e:
        print_check("Exchange bağlantısı", False, f"HATA: {str(e)[:40]}")
        print("\n💡 API anahtarlarını kontrol edin!")
        return False

def print_summary(checks):
    """Özet rapor"""
    print_header("ÖZET RAPOR")
    
    all_passed = all(checks.values())
    
    for name, status in checks.items():
        print_check(name, status)
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ SİSTEM HAZIR!")
        print("\n🚀 Şimdi çalıştırabilirsiniz:")
        print("   python main.py --paper --testnet")
    else:
        print("❌ BAZI SORUNLAR VAR")
        print("\nLütfen yukarıdaki hataları düzeltin ve tekrar deneyin.")
        print("Yardım için: HIZLI_BASLANGIÇ.md dosyasına bakın")
    
    print("="*60 + "\n")
    
    return all_passed

def main():
    print("\n" + "🤖 " + "="*56)
    print("  KRİPTO TRADING BOT - SİSTEM KONTROL PROGRAMI")
    print("="*58 + " 🤖\n")
    
    # Çalışma dizinini kontrol et
    if not os.path.exists('main.py'):
        print("❌ HATA: Lütfen proje dizininde çalıştırın!")
        print(f"   Şu anda: {os.getcwd()}")
        print(f"   Olması gereken: .../crypto_trading_bot/")
        sys.exit(1)
    
    checks = {
        "Environment Variables": check_env_file(),
        "Python Paketleri": check_dependencies(),
        "Proje Yapısı": check_project_structure(),
        "Database": check_database(),
        "Exchange Bağlantısı": check_exchange(),
    }
    
    result = print_summary(checks)
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Kullanıcı tarafından iptal edildi.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Beklenmeyen hata: {e}")
        sys.exit(1)

