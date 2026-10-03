"""
execution.py — order execution layer (STAGED, dry-run first).

Per LIMIT_EXECUTION_PLAN.md. The validated edge only survives at ~0% effective
cost, which means live trades MUST be post-only (maker) limit orders, not market
takers. This module is the execution abstraction that will eventually place those
orders — built one safe stage at a time.

STAGE 1 (this file): compute REAL post-only limit orders per exchange, but in
DRY-RUN (default) LOG the intended order and SEND NOTHING.

SAFETY — real order sending is TRIPLE-gated and OFF by default:
  1. the call must pass dry_run=False, AND
  2. env EXECUTION_LIVE must equal 'true', AND
  3. real API keys must be supplied.
If any is missing, place_limit returns the intended order WITHOUT touching the
exchange. Nothing in the fleet/live loop calls this yet — wiring it in (Stage 2+)
is a later, explicitly-authorized step. This module is import-safe and never
sends an order on its own.
"""
import os
import logging

logger = logging.getLogger("execution")

EXECUTION_LIVE = os.getenv("EXECUTION_LIVE", "false").lower() == "true"

# ccxt exchange id + our "BTC-USD" -> ccxt "BTC/USD"
_CCXT_ID = {
    "coinbase":  "coinbase",     # Coinbase Advanced (unified ccxt id)
    "binanceus": "binanceus",    # Binance.US
    "kraken":    "kraken",
}


def _ccxt_symbol(symbol: str) -> str:
    return symbol.replace("-", "/").upper()


class ExecutionClient:
    """Interface. Implementations must not send real orders unless explicitly and
    fully authorized (see module docstring)."""
    async def place_limit(self, symbol, side, usd_size, price=None, *, post_only=True, dry_run=True):
        raise NotImplementedError
    async def close(self):
        pass


class PaperExecution(ExecutionClient):
    """Models a fill exactly as the backtest/live-paper loop assumes — no network,
    no real order. Fee is supplied by the caller (0 for Coinbase One / Binance.US
    maker)."""
    def __init__(self, fee_rate=0.0):
        self.fee_rate = float(fee_rate)

    async def place_limit(self, symbol, side, usd_size, price=None, *, post_only=True, dry_run=True):
        px = float(price) if price else 0.0
        amount = (float(usd_size) / px) if px else 0.0
        return {
            "mode": "paper", "symbol": symbol, "side": side, "type": "limit",
            "post_only": post_only, "price": px, "amount": round(amount, 8),
            "usd": float(usd_size), "fee_rate": self.fee_rate, "filled": True,
            "note": "modeled maker fill (no network)",
        }


class CcxtExecution(ExecutionClient):
    """Builds a REAL post-only limit order for an exchange via ccxt. In dry-run it
    fetches public market data (no keys needed) to size + price the order, logs
    what it WOULD send, and returns it — WITHOUT creating any order."""
    def __init__(self, exchange_id="coinbase", api_key=None, secret=None, dry_run=True):
        if exchange_id not in _CCXT_ID:
            raise ValueError(f"unsupported exchange '{exchange_id}' (known: {list(_CCXT_ID)})")
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.dry_run = bool(dry_run)
        self._ex = None

    async def _client(self):
        if self._ex is None:
            import ccxt.async_support as ccxt
            klass = getattr(ccxt, _CCXT_ID[self.exchange_id])
            cfg = {"enableRateLimit": True}
            # keys only attached when we might actually send (never in dry-run)
            if not self.dry_run and self.api_key and self.secret:
                cfg["apiKey"] = self.api_key
                cfg["secret"] = self.secret
            self._ex = klass(cfg)
        return self._ex

    async def place_limit(self, symbol, side, usd_size, price=None, *, post_only=True, dry_run=None):
        dry = self.dry_run if dry_run is None else bool(dry_run)
        ex = await self._client()
        sym = _ccxt_symbol(symbol)
        # Price the order: a post-only MAKER rests inside the book — buy at best
        # bid, sell at best ask (illustrative; real wiring can offset further in).
        if price:
            limit_px = float(price)
        else:
            t = await ex.fetch_ticker(sym)
            limit_px = float(t.get("bid") if side == "buy" else t.get("ask") or t.get("last"))
        try:
            await ex.load_markets()
            amount = float(ex.amount_to_precision(sym, float(usd_size) / limit_px))
            limit_px = float(ex.price_to_precision(sym, limit_px))
        except Exception:
            amount = round(float(usd_size) / limit_px, 8)
        intended = {
            "exchange": self.exchange_id, "symbol": sym, "side": side, "type": "limit",
            "post_only": post_only, "price": limit_px, "amount": amount, "usd": float(usd_size),
            "params": {"postOnly": post_only},
        }

        # TRIPLE GATE — anything short of all three stays a dry run.
        can_send = (not dry) and EXECUTION_LIVE and bool(self.api_key and self.secret)
        if not can_send:
            reasons = []
            if dry: reasons.append("dry_run")
            if not EXECUTION_LIVE: reasons.append("EXECUTION_LIVE!=true")
            if not (self.api_key and self.secret): reasons.append("no keys")
            logger.info(f"[execution] DRY-RUN would place: {intended} (blocked by: {','.join(reasons)})")
            return {"sent": False, "dry_run": True, "blocked_by": reasons, "intended": intended}

        # ---- Stage 2+ (authorized real send). Unreachable unless all 3 gates pass. ----
        order = await ex.create_order(sym, "limit", side, amount, limit_px, {"postOnly": post_only})
        logger.warning(f"[execution] LIVE order sent: {order.get('id')} {sym} {side} {amount}@{limit_px}")
        return {"sent": True, "dry_run": False, "order": order}

    async def close(self):
        if self._ex is not None:
            try:
                await self._ex.close()
            except Exception:
                pass
            self._ex = None


async def dry_run_preview(exchange_id, symbol, side, usd_size, price=None):
    """Convenience for the admin dry-run endpoint: build the intended post-only
    limit order for one exchange and return it. Never sends (forced dry_run)."""
    client = CcxtExecution(exchange_id, dry_run=True)
    try:
        return await client.place_limit(symbol, side, float(usd_size), price=price, dry_run=True)
    finally:
        await client.close()
