"""
Trade Journal & Adaptive Learning System
Persists trade history and derives confidence-adjustment weights
that improve signal quality over time.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

JOURNAL_PATH = ".crypto_cache/trade_journal.json"


# ──────────────────────────────────────────────────────────────
# Internal I/O
# ──────────────────────────────────────────────────────────────

def _load() -> dict:
    os.makedirs(".crypto_cache", exist_ok=True)
    if not os.path.exists(JOURNAL_PATH):
        return {"trades": [], "learning": {}}
    try:
        with open(JOURNAL_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"trades": [], "learning": {}}


def _save(data: dict):
    os.makedirs(".crypto_cache", exist_ok=True)
    with open(JOURNAL_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────
# Public API — Trade lifecycle
# ──────────────────────────────────────────────────────────────

def open_trade(signal: dict) -> str:
    """Register a new open trade from a signal dict. Returns trade_id."""
    data = _load()
    trade_id = str(uuid.uuid4())[:8]
    trade = {
        "id": trade_id,
        "coin_id": signal.get("coin_id", ""),
        "symbol": signal.get("symbol", ""),
        "name": signal.get("name", ""),
        "direction": signal.get("direction", "LONG"),
        "strategy": signal.get("strategy", ""),
        "entry_price": signal.get("entry_price", 0),
        "stop_loss": signal.get("stop_loss", 0),
        "tp1": signal.get("tp1", 0),
        "tp2": signal.get("tp2", 0),
        "tp3": signal.get("tp3", 0),
        "tp1_pct": signal.get("tp1_pct", 0),
        "tp2_pct": signal.get("tp2_pct", 0),
        "tp3_pct": signal.get("tp3_pct", 0),
        "sl_pct": signal.get("sl_pct", 0),
        "rr_ratio": signal.get("rr_ratio", 0),
        "confidence": signal.get("confidence", 0),
        "rsi_val": signal.get("rsi_val", 0),
        "macd_active": signal.get("macd_active", False),
        "rsi_active": signal.get("rsi_active", False),
        "cvd_active": signal.get("cvd_active", False),
        "signal_strength": signal.get("signal_strength", ""),
        "win_probability": signal.get("win_probability", 50),
        "reasons": signal.get("reasons", []),
        "notes": signal.get("notes", ""),
        "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "closed_at": None,
        "exit_price": None,
        "result": "OPEN",
        "pnl_pct": None,
        "hit_tp": None,
        "user_notes": "",
    }
    data["trades"].append(trade)
    _save(data)
    return trade_id


def close_trade(trade_id: str, exit_price: float, user_notes: str = "") -> dict:
    """Close an open trade and compute result. Updates learning weights."""
    data = _load()
    trade = next((t for t in data["trades"] if t["id"] == trade_id), None)
    if not trade:
        return {}

    entry = trade["entry_price"]
    direction = trade["direction"]

    if direction == "LONG":
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    # Determine result
    if pnl_pct > 0.15:
        result = "WIN"
    elif pnl_pct < -0.15:
        result = "LOSS"
    else:
        result = "BREAKEVEN"

    # Which TP was hit?
    hit_tp = None
    if result == "WIN":
        if direction == "LONG":
            if exit_price >= trade.get("tp3", float("inf")):
                hit_tp = 3
            elif exit_price >= trade.get("tp2", float("inf")):
                hit_tp = 2
            else:
                hit_tp = 1
        else:
            if exit_price <= trade.get("tp3", 0):
                hit_tp = 3
            elif exit_price <= trade.get("tp2", 0):
                hit_tp = 2
            else:
                hit_tp = 1

    trade.update({
        "exit_price": exit_price,
        "closed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "pnl_pct": round(pnl_pct, 3),
        "hit_tp": hit_tp,
        "user_notes": user_notes,
    })

    _save(data)
    _update_learning(data)
    return trade


def get_open_trades() -> List[dict]:
    return [t for t in _load()["trades"] if t["result"] == "OPEN"]


def get_closed_trades() -> List[dict]:
    closed = [t for t in _load()["trades"] if t["result"] != "OPEN"]
    return sorted(closed, key=lambda x: x.get("closed_at", ""), reverse=True)


def get_all_trades() -> List[dict]:
    return _load()["trades"]


def delete_trade(trade_id: str):
    data = _load()
    data["trades"] = [t for t in data["trades"] if t["id"] != trade_id]
    _save(data)
    _update_learning(data)


# ──────────────────────────────────────────────────────────────
# Learning engine
# ──────────────────────────────────────────────────────────────

def _update_learning(data: dict):
    """Recompute all learning stats from closed trades."""
    closed = [t for t in data["trades"] if t["result"] in ("WIN", "LOSS")]
    if not closed:
        data["learning"] = {}
        _save(data)
        return

    def _stats(subset):
        wins = sum(1 for t in subset if t["result"] == "WIN")
        losses = sum(1 for t in subset if t["result"] == "LOSS")
        total = wins + losses
        win_rate = round(wins / total * 100, 1) if total else 0.0
        pnls = [t["pnl_pct"] for t in subset if t.get("pnl_pct") is not None]
        avg_pnl = round(sum(pnls) / len(pnls), 2) if pnls else 0.0
        return {"wins": wins, "losses": losses, "total": total,
                "win_rate": win_rate, "avg_pnl": avg_pnl}

    # Strategy stats
    strategies = {}
    for t in closed:
        s = t.get("strategy", "Unknown")
        strategies.setdefault(s, []).append(t)
    strategy_stats = {s: _stats(trades) for s, trades in strategies.items()}

    # Indicator combination stats
    combos = {}
    for t in closed:
        key = "_".join(sorted(
            (["MACD"] if t.get("macd_active") else [])
            + (["RSI"] if t.get("rsi_active") else [])
            + (["CVD"] if t.get("cvd_active") else [])
        )) or "NONE"
        combos.setdefault(key, []).append(t)
    indicator_stats = {k: _stats(v) for k, v in combos.items()}

    # Direction stats
    direction_stats = {}
    for d in ("LONG", "SHORT"):
        subset = [t for t in closed if t.get("direction") == d]
        if subset:
            direction_stats[d] = _stats(subset)

    # RSI range stats
    rsi_buckets = {"<30 (Sobrevendido)": [], "30-50 (Recuperación)": [],
                   "50-70 (Tendencia)": [], ">70 (Sobrecomprado)": []}
    for t in closed:
        rsi = t.get("rsi_val", 50)
        if rsi < 30:
            rsi_buckets["<30 (Sobrevendido)"].append(t)
        elif rsi < 50:
            rsi_buckets["30-50 (Recuperación)"].append(t)
        elif rsi < 70:
            rsi_buckets["50-70 (Tendencia)"].append(t)
        else:
            rsi_buckets[">70 (Sobrecomprado)"].append(t)
    rsi_stats = {k: _stats(v) for k, v in rsi_buckets.items() if v}

    # Confidence bucket stats
    conf_buckets = {"40-54%": [], "55-69%": [], "70-84%": [], "85-100%": []}
    for t in closed:
        c = t.get("confidence", 0)
        if c < 55:
            conf_buckets["40-54%"].append(t)
        elif c < 70:
            conf_buckets["55-69%"].append(t)
        elif c < 85:
            conf_buckets["70-84%"].append(t)
        else:
            conf_buckets["85-100%"].append(t)
    confidence_stats = {k: _stats(v) for k, v in conf_buckets.items() if v}

    # Adaptive weights: confidence multipliers per strategy
    # win_rate > 65 → boost +10%, win_rate < 40 → penalty -15%
    adaptive_weights = {}
    for s, st in strategy_stats.items():
        wr = st["win_rate"]
        if st["total"] >= 3:
            if wr >= 75:
                adaptive_weights[s] = 1.15
            elif wr >= 65:
                adaptive_weights[s] = 1.08
            elif wr >= 50:
                adaptive_weights[s] = 1.0
            elif wr >= 40:
                adaptive_weights[s] = 0.92
            else:
                adaptive_weights[s] = 0.82

    # Best performing combo
    best_combo = max(indicator_stats.items(),
                     key=lambda x: x[1]["win_rate"] if x[1]["total"] >= 2 else 0,
                     default=(None, {}))

    data["learning"] = {
        "strategy_stats": strategy_stats,
        "indicator_stats": indicator_stats,
        "direction_stats": direction_stats,
        "rsi_stats": rsi_stats,
        "confidence_stats": confidence_stats,
        "adaptive_weights": adaptive_weights,
        "best_combo": best_combo[0] if best_combo[0] else "",
        "total_closed": len(closed),
        "overall_win_rate": round(
            sum(1 for t in closed if t["result"] == "WIN") / len(closed) * 100, 1
        ),
        "overall_avg_pnl": round(
            sum(t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None) / len(closed), 2
        ),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(data)


def get_learning() -> dict:
    return _load().get("learning", {})


def get_adaptive_weights() -> dict:
    """Returns {strategy_name: confidence_multiplier} to apply to live signals."""
    return _load().get("learning", {}).get("adaptive_weights", {})


def get_global_stats() -> dict:
    data = _load()
    trades = data["trades"]
    closed = [t for t in trades if t["result"] in ("WIN", "LOSS", "BREAKEVEN")]
    wins = [t for t in closed if t["result"] == "WIN"]
    losses = [t for t in closed if t["result"] == "LOSS"]
    open_t = [t for t in trades if t["result"] == "OPEN"]
    pnls = [t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None]
    return {
        "total": len(trades),
        "open": len(open_t),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "best_trade": round(max(pnls), 2) if pnls else 0.0,
        "worst_trade": round(min(pnls), 2) if pnls else 0.0,
        "total_pnl": round(sum(pnls), 2) if pnls else 0.0,
    }
