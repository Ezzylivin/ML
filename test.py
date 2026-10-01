import time
import hmac
import hashlib
import requests
import json
import websocket
import threading
from typing import Dict, Any

# -----------------------------
# Global Config
# -----------------------------
EXCHANGES = ["coinbase", "kraken", "bitstamp", "gemini", "tradestation"]

API_KEYS = {
    "coinbase": {"key": "YOUR_COINBASE_KEY", "secret": "YOUR_COINBASE_SECRET", "passphrase": "YOUR_PASSPHRASE"},
    "kraken": {"key": "YOUR_KRAKEN_KEY", "secret": "YOUR_KRAKEN_SECRET"},
    "bitstamp": {"key": "YOUR_BITSTAMP_KEY", "secret": "YOUR_BITSTAMP_SECRET"},
    "gemini": {"key": "YOUR_GEMINI_KEY", "secret": "YOUR_GEMINI_SECRET"},
    "tradestation": {"key": "YOUR_TRADESTATION_KEY", "secret": "YOUR_TRADESTATION_SECRET"},
}

# Standardized symbol mapping across exchanges
SYMBOLS = {
    "BTCUSD": {
        "coinbase": "BTC-USD",
        "kraken": "XBTUSD",
        "bitstamp": "btcusd",
        "gemini": "BTCUSD",
        "tradestation": "BTC/USD"
    },
    "ETHUSD": {
        "coinbase": "ETH-USD",
        "kraken": "ETHUSD",
        "bitstamp": "ethusd",
        "gemini": "ETHUSD",
        "tradestation": "ETH/USD"
    }
}

# -----------------------------
# Retry Helper
# -----------------------------
def request_with_retry(method, url, headers=None, data=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if method.lower() == "get":
                r = requests.get(url, headers=headers, timeout=10)
            elif method.lower() == "post":
                r = requests.post(url, headers=headers, data=data, timeout=10)
            else:
                raise ValueError("Unsupported HTTP method")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Error: {e}")
            time.sleep(2 ** attempt)
    raise Exception(f"Failed request to {url} after {max_retries} retries")
# -----------------------------
# Base Exchange Class
# -----------------------------
class Exchange:
    def __init__(self, name, api_keys):
        self.name = name
        self.api_keys = api_keys
        self.ws = None

    def get_symbol(self, unified_symbol):
        return SYMBOLS[unified_symbol][self.name]

    # Override per exchange
    def get_price(self, symbol):
        raise NotImplementedError

    def place_order(self, symbol, side, qty, price=None, order_type="market"):
        raise NotImplementedError

    def start_ws(self, on_message):
        raise NotImplementedError

# -----------------------------
# Coinbase
# -----------------------------
class Coinbase(Exchange):
    BASE_URL = "https://api.pro.coinbase.com"

    def get_price(self, symbol):
        sym = self.get_symbol(symbol)
        url = f"{self.BASE_URL}/products/{sym}/ticker"
        data = request_with_retry("get", url)
        return float(data['price'])

    def place_order(self, symbol, side, qty, price=None, order_type="market"):
        url = f"{self.BASE_URL}/orders"
        payload = {
            "product_id": self.get_symbol(symbol),
            "side": side,
            "size": qty,
            "type": order_type
        }
        if order_type == "limit":
            payload["price"] = price
        headers = self._auth_headers(payload)
        return request_with_retry("post", url, headers=headers, data=json.dumps(payload))

    def _auth_headers(self, payload):
        import base64
        import time
        from requests.auth import AuthBase

        timestamp = str(time.time())
        message = timestamp + json.dumps(payload)
        hmac_key = base64.b64decode(self.api_keys["secret"])
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256).hexdigest()
        return {
            "CB-ACCESS-KEY": self.api_keys["key"],
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.api_keys["passphrase"],
            "Content-Type": "application/json"
        }

    def start_ws(self, on_message):
        url = "wss://ws-feed.pro.coinbase.com"
        self.ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: on_message(self.name, msg)
        )
        threading.Thread(target=self.ws.run_forever).start()


# -----------------------------
# Kraken
# -----------------------------
class Kraken(Exchange):
    BASE_URL = "https://api.kraken.com/0/public"

    def get_price(self, symbol):
        sym = self.get_symbol(symbol)
        url = f"{self.BASE_URL}/Ticker?pair={sym}"
        data = request_with_retry("get", url)
        pair = list(data['result'].keys())[0]
        return float(data['result'][pair]['c'][0])

    def place_order(self, symbol, side, qty, price=None, order_type="market"):
        # Kraken private endpoints require HMAC signature; skipping full implementation here for brevity
        print(f"[Kraken] Place order called: {symbol} {side} {qty} {price} {order_type}")
        return {"status": "mocked"}

    def start_ws(self, on_message):
        url = "wss://ws.kraken.com/"
        self.ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: on_message(self.name, msg)
        )
        threading.Thread(target=self.ws.run_forever).start()


# -----------------------------
# Bitstamp
# -----------------------------
class Bitstamp(Exchange):
    BASE_URL = "https://www.bitstamp.net/api/v2"

    def get_price(self, symbol):
        sym = self.get_symbol(symbol)
        url = f"{self.BASE_URL}/ticker/{sym}/"
        data = request_with_retry("get", url)
        return float(data['last'])

    def place_order(self, symbol, side, qty, price=None, order_type="market"):
        print(f"[Bitstamp] Place order called: {symbol} {side} {qty} {price} {order_type}")
        return {"status": "mocked"}

    def start_ws(self, on_message):
        print(f"[Bitstamp] WebSocket not implemented in this snippet")


# -----------------------------
# Gemini
# -----------------------------
class Gemini(Exchange):
    BASE_URL = "https://api.gemini.com/v1"

    def get_price(self, symbol):
        sym = self.get_symbol(symbol)
        url = f"{self.BASE_URL}/pubticker/{sym}"
        data = request_with_retry("get", url)
        return float(data['last'])

    def place_order(self, symbol, side, qty, price=None, order_type="market"):
        print(f"[Gemini] Place order called: {symbol} {side} {qty} {price} {order_type}")
        return {"status": "mocked"}

    def start_ws(self, on_message):
        print(f"[Gemini] WebSocket not implemented in this snippet")


# -----------------------------
# TradeStation
# -----------------------------
class TradeStation(Exchange):
    BASE_URL = "https://api.tradestation.com/v2"

    def get_price(self, symbol):
        sym = self.get_symbol(symbol)
        url = f"{self.BASE_URL}/marketdata/{sym}/quotes"
        data = request_with_retry("get", url)
        return float(data['Last'])  # Assuming 'Last' key exists

    def place_order(self, symbol, side, qty, price=None, order_type="market"):
        print(f"[TradeStation] Place order called: {symbol} {side} {qty} {price} {order_type}")
        return {"status": "mocked"}

    def start_ws(self, on_message):
        print(f"[TradeStation] WebSocket not implemented in this snippet")
# -----------------------------
# P&L Tracking
# -----------------------------
class PnLTracker:
    def __init__(self):
        # {symbol: {"long": qty, "short": qty, "realized": pnl}}
        self.positions = {}

    def update_position(self, symbol, side, qty, price):
        if symbol not in self.positions:
            self.positions[symbol] = {"long": 0, "short": 0, "realized": 0.0}

        pos = self.positions[symbol]

        if side == "buy":
            pos["long"] += qty
        elif side == "sell":
            pos["short"] += qty

        # Simple realized P&L calculation
        # (assuming market orders and immediate execution)
        # For real system, would need FIFO or weighted average
        pos["realized"] += 0  # placeholder

    def get_position(self, symbol):
        return self.positions.get(symbol, {"long": 0, "short": 0, "realized": 0.0})

    def print_positions(self):
        for sym, pos in self.positions.items():
            print(f"{sym}: Long {pos['long']}, Short {pos['short']}, Realized PnL {pos['realized']}")


# -----------------------------
# Order Manager
# -----------------------------
class OrderManager:
    def __init__(self, exchanges):
        self.exchanges = exchanges  # {name: Exchange instance}
        self.pnl_tracker = PnLTracker()

    def place_order(self, exchange_name, symbol, side, qty, price=None, order_type="market"):
        exchange = self.exchanges[exchange_name]
        result = exchange.place_order(symbol, side, qty, price, order_type)
        self.pnl_tracker.update_position(symbol, side, qty, price or 0)
        print(f"[{exchange_name}] Order executed: {side} {qty} {symbol} @ {price or 'market'}")
        return result

    def show_positions(self):
        self.pnl_tracker.print_positions()


# -----------------------------
# Main Execution Loop
# -----------------------------
def main():
    # Initialize exchanges with API keys
    exchanges = {
        "Coinbase": Coinbase("Coinbase", api_keys={"key":"YOUR_KEY","secret":"YOUR_SECRET","passphrase":"YOUR_PASS"}),
        "Kraken": Kraken("Kraken", api_keys={"key":"YOUR_KEY","secret":"YOUR_SECRET"}),
        "Bitstamp": Bitstamp("Bitstamp", api_keys={"key":"YOUR_KEY","secret":"YOUR_SECRET"}),
        "Gemini": Gemini("Gemini", api_keys={"key":"YOUR_KEY","secret":"YOUR_SECRET"}),
        "TradeStation": TradeStation("TradeStation", api_keys={"key":"YOUR_KEY","secret":"YOUR_SECRET"})
    }

    # Initialize Order Manager
    order_manager = OrderManager(exchanges)

    # Example: fetch prices every 5 seconds
    symbols = ["BTCUSD", "ETHUSD"]
    try:
        while True:
            for sym in symbols:
                print(f"\n--- Prices for {sym} ---")
                for name, exch in exchanges.items():
                    try:
                        price = exch.get_price(sym)
                        print(f"{name}: {price}")
                    except Exception as e:
                        print(f"{name}: Error fetching price: {e}")

            # Example order: buy 0.01 BTC on Coinbase at market
            # order_manager.place_order("Coinbase", "BTCUSD", "buy", 0.01)

            # Print positions
            order_manager.show_positions()

            time.sleep(5)

    except KeyboardInterrupt:
        print("Stopping main loop...")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
