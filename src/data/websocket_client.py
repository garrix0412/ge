"""
Real-time Binance WebSocket client for streaming kline / candlestick data.

Connects to the Binance public WebSocket API, parses incoming kline messages,
and forwards parsed OHLCV dicts to registered callback functions.

Usage::

    from src.data.websocket_client import BinanceWebSocket

    def on_candle(data: dict) -> None:
        print(data)

    ws = BinanceWebSocket(symbols=["btcusdt"], callbacks=[on_candle])
    ws.start()
    # ... later ...
    ws.stop()
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import websocket  # websocket-client

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type alias for the parsed candle dict
# ---------------------------------------------------------------------------
CandleDict = dict[str, Any]
CallbackFn = Callable[[CandleDict], None]

# Binance public WebSocket base URL
_BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

# Reconnection parameters
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0


class BinanceWebSocket:
    """Thread-safe Binance WebSocket client for live kline data.

    Parameters
    ----------
    symbols:
        List of lowercase trading-pair symbols (e.g. ``["btcusdt"]``).
    interval:
        Kline interval string recognised by Binance (e.g. ``"1m"``,
        ``"1h"``, ``"4h"``).
    callbacks:
        Optional list of callback functions.  Each receives a parsed
        :pydata:`CandleDict` whenever a kline message arrives.
    """

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        interval: str = "1m",
        callbacks: Optional[list[CallbackFn]] = None,
    ) -> None:
        self.symbols: list[str] = [s.lower() for s in (symbols or ["btcusdt"])]
        self.interval: str = interval
        self._callbacks: list[CallbackFn] = list(callbacks) if callbacks else []

        # Thread-safe shared state
        self._lock = threading.Lock()
        self._latest: dict[str, CandleDict] = {}
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._retry_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_callback(self, fn: CallbackFn) -> None:
        """Register a callback that receives parsed OHLCV candle dicts.

        Parameters
        ----------
        fn:
            Callable accepting a single ``dict`` argument.
        """
        with self._lock:
            self._callbacks.append(fn)
        logger.info("Callback registered: %s", fn.__name__)

    def start(self) -> None:
        """Connect to Binance WebSocket and begin receiving data.

        The connection runs in a daemon background thread so that it does
        not block the caller.
        """
        if self._running:
            logger.warning("WebSocket is already running.")
            return

        self._running = True
        self._retry_count = 0
        self._thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._thread.start()
        logger.info(
            "BinanceWebSocket started — symbols=%s, interval=%s",
            self.symbols,
            self.interval,
        )

    def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception as exc:
                logger.debug("Error closing WebSocket: %s", exc)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("BinanceWebSocket stopped.")

    def get_latest(self, symbol: Optional[str] = None) -> CandleDict:
        """Return the most recently received candle.

        Parameters
        ----------
        symbol:
            Optional symbol filter (lowercase, e.g. ``"btcusdt"``).  When
            *None*, the first symbol in the list is used.

        Returns
        -------
        dict
            The latest OHLCV candle dict, or an empty dict if no data has
            been received yet.
        """
        key = (symbol or self.symbols[0]).lower()
        with self._lock:
            return dict(self._latest.get(key, {}))

    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Parse an incoming kline JSON message and dispatch to callbacks."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse WebSocket message: %s", exc)
            return

        # Binance kline message structure: {"e": "kline", "k": {...}}
        if data.get("e") != "kline":
            return

        kline = data.get("k", {})
        candle: CandleDict = {
            "symbol": kline.get("s", "").upper(),
            "timestamp": datetime.fromtimestamp(
                kline.get("t", 0) / 1000.0, tz=timezone.utc
            ),
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": float(kline.get("c", 0)),
            "volume": float(kline.get("v", 0)),
            "is_closed": bool(kline.get("x", False)),
        }

        # Update latest candle (thread-safe)
        symbol_key = candle["symbol"].lower()
        with self._lock:
            self._latest[symbol_key] = candle

        # Dispatch to registered callbacks
        for cb in self._callbacks:
            try:
                cb(candle)
            except Exception as exc:
                logger.error(
                    "Callback %s raised an exception: %s",
                    cb.__name__,
                    exc,
                    exc_info=True,
                )

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Log the error; reconnection is handled by the connect loop."""
        logger.error("WebSocket error: %s", error)

    def on_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: Optional[int],
        close_msg: Optional[str],
    ) -> None:
        """Log connection closure."""
        logger.info(
            "WebSocket closed (code=%s, msg=%s).",
            close_status_code,
            close_msg,
        )

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Log successful connection and reset retry counter."""
        self._retry_count = 0
        logger.info("WebSocket connection established.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        """Construct the combined stream URL for all symbols."""
        if len(self.symbols) == 1:
            return f"{_BINANCE_WS_BASE}/{self.symbols[0]}@kline_{self.interval}"

        # Multiple symbols: use the combined-stream endpoint
        streams = "/".join(
            f"{sym}@kline_{self.interval}" for sym in self.symbols
        )
        return f"wss://stream.binance.com:9443/stream?streams={streams}"

    def _connect_loop(self) -> None:
        """Connection loop with exponential backoff on failure."""
        while self._running and self._retry_count < _MAX_RETRIES:
            url = self._build_url()
            logger.info(
                "Connecting to %s (attempt %d/%d) …",
                url,
                self._retry_count + 1,
                _MAX_RETRIES,
            )

            self._ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message_wrapper,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )

            try:
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                logger.error("WebSocket run_forever raised: %s", exc)

            if not self._running:
                break

            # Exponential backoff before reconnecting
            self._retry_count += 1
            backoff = min(
                _INITIAL_BACKOFF_S * (2 ** (self._retry_count - 1)),
                _MAX_BACKOFF_S,
            )
            logger.warning(
                "Reconnecting in %.1f s (retry %d/%d) …",
                backoff,
                self._retry_count,
                _MAX_RETRIES,
            )
            time.sleep(backoff)

        if self._retry_count >= _MAX_RETRIES:
            logger.error(
                "Maximum retries (%d) reached — giving up.", _MAX_RETRIES
            )
        self._running = False

    def _on_message_wrapper(
        self, ws: websocket.WebSocketApp, message: str
    ) -> None:
        """Handle combined-stream envelope before dispatching.

        When using the ``/stream?streams=`` endpoint, Binance wraps each
        message in ``{"stream": "...", "data": {...}}``.
        """
        try:
            raw = json.loads(message)
        except json.JSONDecodeError:
            self.on_message(ws, message)
            return

        # Unwrap combined-stream envelope if present
        if "data" in raw and "stream" in raw:
            self.on_message(ws, json.dumps(raw["data"]))
        else:
            self.on_message(ws, message)
