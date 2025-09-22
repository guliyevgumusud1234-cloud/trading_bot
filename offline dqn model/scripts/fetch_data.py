#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "coins.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "market"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FETCH_LOG = PROJECT_ROOT / "logs" / "fetch_summary.jsonl"

PRIMARY_ENDPOINTS = [
    "https://api-gcp.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
]
ARCHIVE_ENDPOINT = "https://data.binance.vision/data/spot/daily/klines"


def load_coin_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunk_symbols(symbols: Iterable[str], chunk_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for sym in symbols:
        batch.append(sym)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_klines_from_endpoint(url: str, params: dict) -> pd.DataFrame:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    req = f"{url}?{query}"
    with urlopen(req) as resp:
        data = json.load(resp)
    if not data:
        raise RuntimeError("empty response")
    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_klines(symbol: str, interval: str, limit: int, base_url: str, end_time: int | None = None) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    }
    if end_time is not None:
        params["endTime"] = str(end_time)

    candidate_urls = [base_url] if base_url else []
    for endpoint in PRIMARY_ENDPOINTS:
        if endpoint not in candidate_urls:
            candidate_urls.append(endpoint)

    last_error: Exception | None = None
    for url in candidate_urls:
        try:
            return fetch_klines_from_endpoint(url, params)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(f"All primary endpoints failed for {symbol}: {last_error}")


def _validate(new_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    issues = []
    if (new_df[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append("non-positive price detected; rows filtered")
        new_df = new_df[(new_df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if (new_df["high"] < new_df["low"]).any():
        issues.append("high < low encountered; swapped")
        mask = new_df["high"] < new_df["low"]
        new_df.loc[mask, ["high", "low"]] = new_df.loc[mask, ["low", "high"]].values
    new_df = new_df.dropna()
    if issues:
        print(f"[fetch] {symbol} warnings: {', '.join(issues)}", file=sys.stderr)
    return new_df


def merge_save(symbol: str, new_df: pd.DataFrame) -> Path:
    new_df = _validate(new_df, symbol)
    path = DATA_DIR / f"{symbol.lower()}_1h.csv"
    if path.exists():
        old = pd.read_csv(path, parse_dates=["timestamp"])
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    else:
        combined = new_df.sort_values("timestamp")
    combined.to_csv(path, index=False)
    return path


def fetch_from_archive(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    # Archive provides daily zipped files; download recent days until we reach limit
    now = datetime.utcnow()
    rows: list[pd.DataFrame] = []
    day = 0
    while sum(len(df) for df in rows) < limit and day < 10:
        target_date = (now - timedelta(days=day)).strftime("%Y-%m-%d")
        symbol_upper = symbol.upper()
        url = f"{ARCHIVE_ENDPOINT}/{symbol_upper}/{interval}/{symbol_upper}-{interval}-{target_date}.zip"
        try:
            with urlopen(url) as resp:
                content = resp.read()
        except Exception as exc:  # noqa: BLE001
            day += 1
            continue
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    df = pd.read_csv(f, header=None)
                    df.columns = [
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base",
                        "taker_buy_quote",
                        "ignore",
                    ]
                    rows.append(df)
        day += 1
    if not rows:
        raise RuntimeError("archive data unavailable")
    df_concat = pd.concat(rows, ignore_index=True).sort_values("open_time")
    # keep only needed rows
    df_concat = df_concat.tail(limit)
    df_concat["timestamp"] = pd.to_datetime(df_concat["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df_concat[numeric_cols] = df_concat[numeric_cols].astype(float)
    return df_concat[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_symbol(symbol: str, interval: str, limit: int, base_url: str, retries: int = 3, sleep: float = 1.0) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = fetch_klines(symbol, interval, limit, base_url)
        except (HTTPError, URLError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"[fetch] {symbol} attempt {attempt} failed: {exc}, retrying...", file=sys.stderr)
            time.sleep(sleep * attempt)
        else:
            return merge_save(symbol, df)

    # fallback to archive download if primary endpoints failed
    try:
        df = fetch_from_archive(symbol, interval, limit)
        print(f"[fetch] {symbol} fallback archive used", file=sys.stderr)
        return merge_save(symbol, df)
    except Exception as exc:
        if last_error is None:
            last_error = exc
        raise RuntimeError(f"Failed to fetch {symbol}: {last_error}")


def throttled_fetch(symbols: list[str], interval: str, limit: int, base_url: str, rate_limit_per_minute: int) -> dict:
    results = {}
    if not symbols:
        return results
    per_request_delay = 60.0 / max(rate_limit_per_minute, 1)
    for symbol in symbols:
        start = time.time()
        try:
            path = fetch_symbol(symbol, interval, limit, base_url)
            results[symbol] = {"status": "ok", "path": str(path)}
        except Exception as exc:  # noqa: BLE001
            results[symbol] = {"status": "error", "error": str(exc)}
        elapsed = time.time() - start
        sleep = per_request_delay - elapsed
        if sleep > 0:
            time.sleep(sleep)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly kline data for configured symbols")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to coins.yaml")
    parser.add_argument("--group", type=str, choices=["core", "extended", "all"], default="all", help="Symbol group to fetch")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files (otherwise append/merge)")
    args = parser.parse_args()

    cfg = load_coin_config(Path(args.config))
    symbols = []
    if args.group in ("core", "all"):
        symbols.extend(cfg["symbols"]["core"])
    if args.group in ("extended", "all"):
        symbols.extend(cfg["symbols"]["extended"])

    interval = cfg["settings"]["interval"]
    limit = int(cfg["settings"].get("lookback_limit", 720))
    base_url = cfg["settings"]["base_url"]
    rate_limit = int(cfg["settings"].get("rate_limit_per_minute", 1100))

    if args.force:
        for symbol in symbols:
            path = DATA_DIR / f"{symbol.lower()}_1h.csv"
            if path.exists():
                path.unlink()

    results = throttled_fetch(symbols, interval, limit, base_url, rate_limit)
    success = [s for s, info in results.items() if info["status"] == "ok"]
    failures = {s: info for s, info in results.items() if info["status"] != "ok"}

    print("[fetch] completed")
    print("  success:", ", ".join(success))
    if failures:
        print("  failures:")
        for sym, info in failures.items():
            print(f"    - {sym}: {info['error']}")

    try:
        FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FETCH_LOG.open("a", encoding="utf-8") as f:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "group": args.group,
                "success": success,
                "failures": failures,
            }
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] unable to write summary log: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
