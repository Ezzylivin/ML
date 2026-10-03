"""
edge_discovery.py — autonomous edge search.

Walk-forward tests a GRID of (symbol x timeframe x strategy-set x direction x
combo-rule) and ranks each by OUT-OF-SAMPLE consistency, so the bot can find
configs that actually clear costs instead of trusting one lucky backtest.

A config "has edge" when, across N out-of-sample folds, it is profitable in
>= MIN_PROFITABLE_FOLDS of them, with positive mean ROI and enough trades.

Higher timeframes (4h/1d) are RESAMPLED from the cached 1h CSV, so no extra data
fetch is needed. Runs ML-off (pure strategy edge) for speed; the winners can then
be re-checked with the ML gate.

Output: data/discovered_edges.json   (also returned by discover()).
Run standalone:  python edge_discovery.py
"""
import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config2 import DATA_DIR
from app.backtest2 import Backtester
from app.verify.engineer_and_train import apply_mega_features

log = logging.getLogger("edge_discovery")
OUT_PATH = os.path.join(DATA_DIR, "discovered_edges.json")

# ---- search space (edit to widen/narrow the hunt) ---------------------------
SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "UNIVERSE_SYMBOLS",
    "BTC-USD,ETH-USD,SOL-USD,XRP-USD,DOGE-USD,ADA-USD,SUI-USD,PEPE-USD,SHIB-USD"
).split(",") if s.strip()]
TIMEFRAMES = ["1h", "4h", "1d"]
STRAT_SETS = {
    "trend":      [{"code": "supertrend"}, {"code": "ema_cloud"}, {"code": "sma_crossover"}],
    "momentum":   [{"code": "macd_crossover"}, {"code": "atr_breakout"}, {"code": "vol_profile"}],
    "meanrev":    [{"code": "rsi_threshold"}, {"code": "bb_fade"}, {"code": "stoch"}],
    "supertrend": [{"code": "supertrend"}],
    "ema_cloud":  [{"code": "ema_cloud"}],
    "macd":       [{"code": "macd_crossover"}],
    # ---- NEW INPUT sets (cross-asset / regime + perp funding) ----
    "btc_regime":   [{"code": "btc_regime"}],
    "rel_strength": [{"code": "rel_strength"}],
    "xasset":       [{"code": "btc_regime"}, {"code": "rel_strength"}],
    "trend_regime": [{"code": "supertrend"}, {"code": "ema_cloud"}, {"code": "btc_regime"}],
    "funding":      [{"code": "funding_extreme"}],
    "funding_trend":[{"code": "funding_extreme"}, {"code": "supertrend"}],
}
DIRECTIONS = ["LONG", "BOTH"]
RULES      = ["OR"]

FOLDS                = 6
MIN_ROWS             = 250   # prepared bars needed to bother testing
MIN_TRADES_PER_FOLD  = 4
MIN_PROFITABLE_FOLDS = 4     # out of FOLDS
_RESAMPLE = {"4h": "4h", "1d": "1D"}


def _read_indexed(path):
    """Read an OHLCV CSV into a clean datetime-indexed frame. Handles BOTH integer
    unix timestamps (s or ms) and ISO strings (parsing ints without a unit makes
    pandas read them as NANOSECONDS -> all ~1970 -> one resample bucket). None on
    failure."""
    try:
        df = pd.read_csv(path)
        tcol = next((c for c in df.columns if c.lower() in ("timestamp", "time", "date")), None)
        if tcol is None:
            return None
        ts_num = pd.to_numeric(df[tcol], errors="coerce")
        if ts_num.notna().mean() > 0.9:
            fv = float(ts_num.dropna().iloc[0])
            unit = "ms" if fv > 1e11 else "s"
            df[tcol] = pd.to_datetime(ts_num, unit=unit, utc=True, errors="coerce")
        else:
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df[~df.index.duplicated(keep="first")]
    except Exception as e:
        log.warning(f"read {path} failed: {e}")
        return None


def _load_tf(symbol: str, tf: str):
    """Load bars for a timeframe. Prefers a dedicated {symbol}-{tf}.csv (e.g. a long
    TRUE-daily file from ensure_history) when present and long enough; otherwise
    resamples the 1h CSV."""
    direct = os.path.join(DATA_DIR, f"{symbol}-{tf}.csv")
    if os.path.exists(direct):
        d = _read_indexed(direct)
        if d is not None and len(d) >= MIN_ROWS:
            log.info(f"{symbol} {tf}: loaded {len(d)} bars from dedicated csv")
            return d
    base = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    if not os.path.exists(base):
        log.warning(f"{symbol}: no 1h csv at {base}")
        return None
    df = _read_indexed(base)
    if df is None:
        return None
    if tf == "1h":
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    rules = ["4h", "4H"] if tf == "4h" else ["1D", "1d"]
    out = None
    for rule in rules:
        try:
            cand = df.resample(rule).agg(agg).dropna()
            if cand is not None and len(cand) > 0:
                out = cand
                break
        except Exception as e:
            log.warning(f"{symbol} {tf}: resample '{rule}' failed: {e}")
    log.info(f"{symbol} {tf}: resampled to {0 if out is None else len(out)} bars from {len(df)} 1h")
    return out


# ============================================================
# 🔗 NEW-INPUT ATTACHERS (cross-asset BTC reference + perp funding)
# ============================================================
_BTC_CACHE = {}
_FUNDING_CACHE = {}

def _attach_btc(feat_df, symbol, tf):
    """Add a 'btc_close' column aligned to feat_df.index for cross-asset/regime
    strategies. BTC uses its own close. No-op if BTC data is unavailable."""
    try:
        if symbol.upper().startswith("BTC"):
            out = feat_df.copy(); out["btc_close"] = out["close"].values
            return out
        key = ("BTC-USD", tf)
        if key not in _BTC_CACHE:
            _BTC_CACHE[key] = _load_tf("BTC-USD", tf)
        btc = _BTC_CACHE[key]
        if btc is None or "close" not in btc.columns:
            return feat_df
        out = feat_df.copy()
        out["btc_close"] = btc["close"].reindex(out.index, method="ffill").values
        return out
    except Exception as e:
        log.warning(f"attach_btc {symbol} {tf} failed: {e}")
        return feat_df


def _fetch_funding(symbol):
    """Perp funding-rate history from Bybit (USDT perp): DataFrame indexed by ts
    with a 'funding' column, or None. Bybit has broad access + long history."""
    try:
        import ccxt
        ex = ccxt.bybit({"enableRateLimit": True})
        market = f"{symbol.split('-')[0]}/USDT:USDT"
        rows = []
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365 * 3)).timestamp() * 1000)
        for _ in range(80):
            batch = ex.fetch_funding_rate_history(market, since=since, limit=200)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1]["timestamp"] + 1
            if len(batch) < 200:
                break
        if not rows:
            return None
        df = pd.DataFrame([{"ts": r["timestamp"], "funding": r.get("fundingRate")} for r in rows])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.dropna().drop_duplicates("ts").set_index("ts").sort_index()
    except Exception as e:
        log.warning(f"fetch_funding {symbol} failed: {e}")
        return None


def _attach_funding(feat_df, symbol, tf):
    """Add a 'funding' column (perp funding rate, forward-filled to the bar index).
    No-op if funding can't be fetched."""
    try:
        if symbol not in _FUNDING_CACHE:
            _FUNDING_CACHE[symbol] = _fetch_funding(symbol)
        fnd = _FUNDING_CACHE[symbol]
        if fnd is None or not len(fnd):
            return feat_df
        out = feat_df.copy()
        out["funding"] = fnd["funding"].reindex(out.index, method="ffill").values
        return out
    except Exception as e:
        log.warning(f"attach_funding {symbol} failed: {e}")
        return feat_df


def _score_config(feat_df, symbol, tf, strats, direction, rule):
    """Walk-forward a single config over FOLDS slices of a prepared frame."""
    n = len(feat_df)
    if n < MIN_ROWS:
        return None
    edges = [int(n * k / FOLDS) for k in range(FOLDS + 1)]
    cfg = {
        "symbol": symbol, "timeframe": tf, "initialBalance": 1000.0,
        "risk_percentage": 1.0, "strategies": strats, "trade_direction": direction,
        "mlModel": "off", "params": {"hybridMode": rule},
        "comboConfig": {"combinationRule": rule},
    }
    bt = Backtester(cfg)
    rois, wrs, trades = [], [], 0
    for k in range(FOLDS):
        sl = feat_df.iloc[edges[k]:edges[k + 1]]
        if len(sl) < 30:
            continue
        try:
            res = bt._simulate(sl, None)   # ML off -> ml_probs None
        except Exception as e:
            log.warning(f"{symbol} {tf} sim fold {k} failed: {e}")
            continue
        m = res.get("metrics", {})
        rois.append(float(m.get("roi", 0.0)))
        wrs.append(float(m.get("win_rate", 0.0)))
        trades += int(m.get("total_trades", 0) or 0)
    if not rois:
        return None
    prof = sum(1 for r in rois if r > 0)
    mean_roi = sum(rois) / len(rois)
    has_edge = (prof >= MIN_PROFITABLE_FOLDS and mean_roi > 0
                and trades >= MIN_TRADES_PER_FOLD * len(rois))
    return {
        "symbol": symbol, "timeframe": tf, "set": None,
        "strategies": [s["code"] for s in strats], "direction": direction, "rule": rule,
        "folds": len(rois), "profitable_folds": prof,
        "mean_roi": round(mean_roi, 2), "worst_roi": round(min(rois), 2),
        "best_roi": round(max(rois), 2), "total_trades": trades,
        "mean_win_rate": round(sum(wrs) / len(wrs), 1), "edge": bool(has_edge),
    }


def discover(symbols=None, timeframes=None, top=20, write=True):
    """Run the full grid search. Returns a ranked report dict and (optionally)
    writes it to data/discovered_edges.json."""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    results = []
    for symbol in symbols:
        for tf in timeframes:
            raw = _load_tf(symbol, tf)
            if raw is None or len(raw) < MIN_ROWS:
                log.info(f"skip {symbol} {tf}: raw rows={0 if raw is None else len(raw)} < {MIN_ROWS}")
                continue
            try:
                feat_df, _ = apply_mega_features(raw.copy())
            except Exception as e:
                log.warning(f"{symbol} {tf}: feature build failed: {e}")
                continue
            if feat_df is None or len(feat_df) < MIN_ROWS:
                log.info(f"skip {symbol} {tf}: feature rows={0 if feat_df is None else len(feat_df)} < {MIN_ROWS}")
                continue
            # Attach the new inputs (cross-asset BTC ref + perp funding) so the
            # btc_regime / rel_strength / funding_extreme strategies can vote.
            feat_df = _attach_btc(feat_df, symbol, tf)
            feat_df = _attach_funding(feat_df, symbol, tf)
            for name, strats in STRAT_SETS.items():
                for direction in DIRECTIONS:
                    for rule in RULES:
                        r = _score_config(feat_df, symbol, tf, strats, direction, rule)
                        if r:
                            r["set"] = name
                            results.append(r)

    # rank: real edge first, then # profitable folds, then mean ROI
    results.sort(key=lambda x: (x["edge"], x["profitable_folds"], x["mean_roi"]), reverse=True)
    edges = [r for r in results if r["edge"]]
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "tested": len(results), "edges_found": len(edges),
        "criteria": {"folds": FOLDS, "min_profitable_folds": MIN_PROFITABLE_FOLDS,
                     "min_trades_per_fold": MIN_TRADES_PER_FOLD},
        "edges": edges, "top": results[:top],
    }
    if write:
        try:
            with open(OUT_PATH, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"edge_discovery: {len(edges)} edge(s) / {len(results)} tested -> {OUT_PATH}")
        except Exception as e:
            log.warning(f"edge_discovery write failed: {e}")
    return report


# ============================================================
# 📦 DATA SUFFICIENCY — fetch enough history per coin/timeframe
# ============================================================
def _fetch_history(symbol, timeframe, years):
    """Fetch up to `years` of OHLCV from Binance.US (sync). Reachable from this
    server (not geo-blocked like Bybit/Binance-global) and supports NATIVE 4h + 1d
    with deep history, so we no longer have to resample short Coinbase data.
    Binance.US is SPOT-only — no funding rates."""
    import ccxt, time as _t
    ex = ccxt.binanceus({"enableRateLimit": True})
    base = symbol.split("-")[0]
    fsym = f"{base}/USDT"
    now = int(_t.time() * 1000)
    since = now - int(years * 365 * 24 * 3600 * 1000)
    rows = []
    for _ in range(600):  # paginate; Binance limit up to 1000/req
        try:
            batch = ex.fetch_ohlcv(fsym, timeframe, since=since, limit=1000)
        except Exception as e:
            log.warning(f"{symbol} {timeframe}: binanceus fetch error {e}")
            break
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    return rows


def ensure_history(symbols=None, years_1h=3, years_1d=6):
    """Make sure each symbol has enough history: top up the 1h CSV (used for 1h +
    4h-resample) and fetch a long TRUE-daily {symbol}-1d.csv. Returns a per-symbol
    row-count report. Safe to run repeatedly (merges + dedupes)."""
    symbols = symbols or SYMBOLS
    report = {}
    for sym in symbols:
        rc = {}
        for tf, yrs in (("1h", years_1h), ("4h", max(years_1h, 5)), ("1d", years_1d)):
            path = os.path.join(DATA_DIR, f"{sym}-{tf}.csv")
            try:
                rows = _fetch_history(sym, tf, yrs)
                if not rows:
                    rc[tf] = "no data"
                    continue
                new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                new["timestamp"] = pd.to_datetime(new["timestamp"], unit="ms", utc=True)
                if os.path.exists(path):
                    old = _read_indexed(path)
                    if old is not None and len(old):
                        old = old.reset_index()
                        old = old.rename(columns={old.columns[0]: "timestamp"})
                        keep = ["timestamp", "open", "high", "low", "close", "volume"]
                        old = old[[c for c in keep if c in old.columns]]
                        new = pd.concat([old, new], ignore_index=True)
                new = new.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
                os.makedirs(DATA_DIR, exist_ok=True)
                new.to_csv(path, index=False)
                rc[tf] = len(new)
            except Exception as e:
                rc[tf] = f"error: {e}"
        report[sym] = rc
        log.info(f"ensure_history {sym}: {rc}")
    return report


# ============================================================
# 🔬 DEEP VALIDATION of a candidate (param-robustness walk-forward)
# ============================================================
def validate_config(symbol="SOL-USD", timeframe="1d", folds=8,
                    directions=("LONG", "BOTH"), macd_fast_grid=(8, 10, 12, 16, 20)):
    """Deep validation of the daily MACD candidate: long history, more folds, and a
    parameter-robustness sweep over MACD `fast`. A DURABLE edge holds across most
    params AND most folds — not one lucky setting."""
    raw = _load_tf(symbol, timeframe)
    if raw is None or len(raw) < MIN_ROWS:
        return {"error": f"insufficient data for {symbol} {timeframe}: {0 if raw is None else len(raw)} bars"}
    feat_df, _ = apply_mega_features(raw.copy())
    if feat_df is None or len(feat_df) < MIN_ROWS:
        return {"error": f"insufficient prepared data: {0 if feat_df is None else len(feat_df)}"}
    n = len(feat_df)
    edges = [int(n * k / folds) for k in range(folds + 1)]
    out = []
    for direction in directions:
        for fast in macd_fast_grid:
            strats = [{"code": "macd_crossover", "params": {"fast": int(fast)}}]
            cfg = {"symbol": symbol, "timeframe": timeframe, "initialBalance": 1000.0,
                   "risk_percentage": 1.0, "strategies": strats, "trade_direction": direction,
                   "mlModel": "off", "params": {"hybridMode": "OR"}, "comboConfig": {"combinationRule": "OR"}}
            bt = Backtester(cfg)
            rois, wrs, trades = [], [], 0
            for k in range(folds):
                sl = feat_df.iloc[edges[k]:edges[k + 1]]
                if len(sl) < 30:
                    continue
                try:
                    m = bt._simulate(sl, None).get("metrics", {})
                except Exception:
                    continue
                rois.append(float(m.get("roi", 0))); wrs.append(float(m.get("win_rate", 0))); trades += int(m.get("total_trades", 0) or 0)
            if not rois:
                continue
            prof = sum(1 for r in rois if r > 0)
            mean_roi = sum(rois) / len(rois)
            out.append({"direction": direction, "macd_fast": int(fast), "folds": len(rois),
                        "profitable_folds": prof, "mean_roi": round(mean_roi, 2),
                        "worst_roi": round(min(rois), 2), "best_roi": round(max(rois), 2),
                        "total_trades": trades, "mean_win_rate": round(sum(wrs) / len(wrs), 1),
                        "edge": prof >= int(0.6 * len(rois)) and mean_roi > 0})
    out.sort(key=lambda x: (x["edge"], x["profitable_folds"], x["mean_roi"]), reverse=True)
    robust = sum(1 for x in out if x["edge"])
    return {"symbol": symbol, "timeframe": timeframe, "bars": n, "folds": folds,
            "variants_tested": len(out), "variants_with_edge": robust,
            "robust": (robust >= max(2, int(0.5 * len(out))) if out else False),
            "results": out}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rep = discover()
    print(f"\nTested {rep['tested']} configs; {rep['edges_found']} cleared the edge bar.\n")
    for r in rep["top"][:15]:
        tag = "EDGE " if r["edge"] else "     "
        print(f"{tag}{r['symbol']:9} {r['timeframe']:3} {r['set']:10} {r['direction']:4} "
              f"| profit {r['profitable_folds']}/{r['folds']} | meanROI {r['mean_roi']:+.1f}% "
              f"| worst {r['worst_roi']:+.1f}% | {r['total_trades']}t | win {r['mean_win_rate']}%")
