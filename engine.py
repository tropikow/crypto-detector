"""
Crypto Bull Detector - Analysis Engine
Fetches market data from CoinGecko and performs technical analysis
to score cryptocurrencies by their bull run proximity.
"""

import os
import time
import pickle
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

CACHE_DIR = ".crypto_cache"
BASE_URL = "https://api.coingecko.com/api/v3"

# ─────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────

def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_get(key: str, ttl: int = 300) -> Optional[any]:
    _ensure_cache()
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(path):
        if time.time() - os.path.getmtime(path) < ttl:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None

def _cache_set(key: str, data: any):
    _ensure_cache()
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)

# ─────────────────────────────────────────
# API client
# ─────────────────────────────────────────

def _api(endpoint: str, params: dict = None, retries: int = 3) -> Optional[any]:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15,
                             headers={"accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(65)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_markets(limit: int = 100) -> List[Dict]:
    """Get top coins by market cap with sparkline + price change data."""
    key = f"markets_{limit}"
    cached = _cache_get(key, 300)
    if cached:
        return cached

    all_coins: List[Dict] = []
    pages = (limit + 249) // 250
    for page in range(1, pages + 1):
        per_page = min(250, limit - len(all_coins))
        data = _api("coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "true",
            "price_change_percentage": "1h,24h,7d,30d",
        })
        if data:
            all_coins.extend(data)
        if len(all_coins) >= limit:
            break
        time.sleep(0.5)

    result = all_coins[:limit]
    _cache_set(key, result)
    return result


def fetch_ohlcv(coin_id: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candlestick data (4H candles for 30 days = ~180 candles)."""
    key = f"ohlcv_{coin_id}_{days}"
    cached = _cache_get(key, 600)
    if cached is not None:
        return cached

    data = _api(f"coins/{coin_id}/ohlc", {"vs_currency": "usd", "days": days})
    if not data:
        return None

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    _cache_set(key, df)
    time.sleep(0.35)
    return df


def fetch_volume_chart(coin_id: str, days: int = 30) -> Optional[pd.Series]:
    """Fetch daily volume data."""
    key = f"vol_{coin_id}_{days}"
    cached = _cache_get(key, 600)
    if cached is not None:
        return cached

    data = _api(f"coins/{coin_id}/market_chart",
                {"vs_currency": "usd", "days": days, "interval": "daily"})
    if not data:
        return None

    vols = pd.Series([v[1] for v in data.get("total_volumes", [])], name="volume")
    _cache_set(key, vols)
    time.sleep(0.35)
    return vols


def fetch_hourly_chart(coin_id: str, days: int = 14) -> Optional[pd.DataFrame]:
    """Fetch hourly price + volume data (used for whale/smart-money analysis)."""
    key = f"hourly_{coin_id}_{days}"
    cached = _cache_get(key, 600)
    if cached is not None:
        return cached

    data = _api(f"coins/{coin_id}/market_chart",
                {"vs_currency": "usd", "days": days})
    if not data:
        return None

    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    if not prices or not volumes:
        return None

    df_p = pd.DataFrame(prices, columns=["ts", "price"])
    df_v = pd.DataFrame(volumes, columns=["ts", "volume"])
    df = pd.merge(df_p, df_v, on="ts", how="inner")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    _cache_set(key, df)
    time.sleep(0.35)
    return df


# ─────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta).clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def calc_bb(close: pd.Series, period=20, std_dev=2) -> Tuple:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma.replace(0, np.nan)
    pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, sma, lower, width, pct


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def calc_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, d: int = 3) -> Tuple:
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    K = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    D = K.rolling(d).mean()
    return K, D


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — accumulation cuando OBV sube con precio lateral/bajo."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def calc_cmf(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow simplificado (solo close + volume).
    Proxy: MF multiplier = (close - rolling_low) / (rolling_high - rolling_low) * 2 - 1
    """
    rmin = close.rolling(period).min()
    rmax = close.rolling(period).max()
    mf_mult = (close - rmin) / (rmax - rmin + 1e-12) * 2 - 1
    mf_vol = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def calc_vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price."""
    tp = close  # without high/low, use close as typical price
    return (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ─────────────────────────────────────────
# Whale / Smart Money Analysis
# ─────────────────────────────────────────

def analyze_whale_activity(
    hourly_df: Optional[pd.DataFrame],
    daily_vols: Optional[pd.Series],
    daily_price_chg_7d: float = 0.0,
) -> Dict:
    """
    Detecta actividad de grandes traders (ballenas / institucionales) usando:
    - OBV divergence: OBV subiendo mientras precio cae = acumulación silenciosa
    - CMF: Chaikin Money Flow positivo = dinero entrando
    - VWAP position: precio bajo el VWAP = activo infravalorado vs smart money
    - Volume spikes: volumen 3x+ en corto periodo = entrada institucional
    - Quiet accumulation pattern: compresión volumen + breakout inminente
    """

    result = {
        "whale_score": 50,
        "whale_signal": "DATOS INSUFICIENTES",
        "whale_color": "#888888",
        "obv_trend": "neutral",
        "obv_divergence": False,
        "cmf": 0.0,
        "vwap_position": "unknown",
        "vol_spike": False,
        "vol_spike_ratio": 1.0,
        "quiet_accumulation": False,
        "distribution_warning": False,
        "smart_money_score": 0,
        "notes": [],
        "obv_series": [],
        "cmf_series": [],
        "vwap_series": [],
    }

    notes = []
    score = 0

    # ── 1. Daily volume analysis ──
    if daily_vols is not None and len(daily_vols) >= 7:
        vols_arr = daily_vols.dropna().values.astype(float)
        avg_20 = float(np.mean(vols_arr[-20:])) if len(vols_arr) >= 20 else float(np.mean(vols_arr))
        avg_3  = float(np.mean(vols_arr[-3:])) if len(vols_arr) >= 3 else vols_arr[-1]
        spike_ratio = avg_3 / avg_20 if avg_20 > 0 else 1.0
        result["vol_spike_ratio"] = round(spike_ratio, 2)

        if spike_ratio >= 3.0:
            result["vol_spike"] = True
            score += 20
            notes.append(f"🐳 SPIKE DE VOLUMEN {spike_ratio:.1f}× el promedio — entrada de ballenas detectada")
        elif spike_ratio >= 2.0:
            result["vol_spike"] = True
            score += 13
            notes.append(f"📈 Volumen elevado {spike_ratio:.1f}× la media — interés institucional creciente")
        elif spike_ratio >= 1.4:
            score += 7
            notes.append(f"📊 Volumen sobre media ({spike_ratio:.1f}×) — acumulación moderada")
        elif spike_ratio < 0.5:
            score -= 5
            notes.append("⚠️ Volumen muy bajo — smart money aún no se ha posicionado")
        else:
            notes.append(f"📊 Volumen en promedio ({spike_ratio:.1f}×) — sin señal de ballenas aún")

    # ── 2. OBV + CMF + VWAP desde datos horarios ──
    if hourly_df is not None and len(hourly_df) >= 50:
        close_h = hourly_df["price"]
        vol_h   = hourly_df["volume"]

        obv_s  = calc_obv(close_h, vol_h)
        cmf_s  = calc_cmf(close_h, vol_h, period=20)
        vwap_s = calc_vwap(close_h, vol_h)

        obv_now  = float(obv_s.iloc[-1])
        obv_24h  = float(obv_s.iloc[-24]) if len(obv_s) >= 24 else float(obv_s.iloc[0])
        obv_48h  = float(obv_s.iloc[-48]) if len(obv_s) >= 48 else float(obv_s.iloc[0])
        price_now = float(close_h.iloc[-1])
        price_24h = float(close_h.iloc[-24]) if len(close_h) >= 24 else float(close_h.iloc[0])

        cmf_val   = float(cmf_s.dropna().iloc[-1]) if not cmf_s.dropna().empty else 0.0
        vwap_now  = float(vwap_s.iloc[-1])

        result["cmf"]          = round(cmf_val, 4)
        result["obv_series"]   = obv_s.tail(168).round(0).tolist()
        result["cmf_series"]   = cmf_s.tail(168).round(4).tolist()
        result["vwap_series"]  = vwap_s.tail(168).round(8).tolist()
        result["vwap_value"]   = round(vwap_now, 8)

        # OBV trend
        obv_rising_24h = obv_now > obv_24h
        obv_rising_48h = obv_24h > obv_48h
        price_falling  = price_now < price_24h

        if obv_rising_24h and obv_rising_48h:
            result["obv_trend"] = "rising"
            score += 15
            notes.append("✅ OBV subiendo consistentemente — smart money acumulando")
        elif obv_rising_24h:
            result["obv_trend"] = "rising"
            score += 8
            notes.append("📈 OBV iniciando recuperación — posible acumulación temprana")
        elif not obv_rising_24h and not obv_rising_48h:
            result["obv_trend"] = "falling"
            score -= 8
            notes.append("⚠️ OBV en descenso — distribución o salida de capitales")
        else:
            result["obv_trend"] = "neutral"
            notes.append("➡️ OBV lateral — smart money en espera")

        # OBV divergence: precio baja pero OBV sube = SEÑAL FUERTE de acumulación
        if price_falling and obv_rising_24h:
            result["obv_divergence"] = True
            score += 20
            notes.append("🚨 DIVERGENCIA ALCISTA OBV — precio cae pero ballenas compran. Señal institucional fuerte.")

        # CMF
        if cmf_val > 0.15:
            score += 15
            notes.append(f"💰 CMF = {cmf_val:.3f} — flujo de dinero positivo fuerte (smart money comprando)")
        elif cmf_val > 0.05:
            score += 8
            notes.append(f"📊 CMF = {cmf_val:.3f} — flujo comprador moderado")
        elif cmf_val < -0.15:
            score -= 10
            notes.append(f"⛔ CMF = {cmf_val:.3f} — flujo vendedor intenso (distribución)")
            result["distribution_warning"] = True
        elif cmf_val < -0.05:
            score -= 5
            notes.append(f"⚠️ CMF = {cmf_val:.3f} — presión vendedora moderada")
        else:
            notes.append(f"➡️ CMF = {cmf_val:.3f} — flujo neutral")

        # VWAP position
        if price_now < vwap_now * 0.95:
            result["vwap_position"] = "far_below"
            score += 12
            notes.append(f"🎯 Precio {((price_now/vwap_now-1)*100):.1f}% bajo VWAP — zona de valor para institucionales")
        elif price_now < vwap_now:
            result["vwap_position"] = "below"
            score += 6
            notes.append(f"📉 Precio bajo VWAP ({((price_now/vwap_now-1)*100):.1f}%) — posible zona de acumulación")
        else:
            result["vwap_position"] = "above"
            notes.append(f"📈 Precio sobre VWAP (+{((price_now/vwap_now-1)*100):.1f}%) — momentum positivo")

        # Quiet accumulation: OBV rising + price compressing + CMF starting positive
        compression = abs(price_now / price_24h - 1) < 0.02  # precio casi igual en 24h
        if compression and obv_rising_24h and cmf_val > 0:
            result["quiet_accumulation"] = True
            score += 15
            notes.append("🤫 ACUMULACIÓN SILENCIOSA detectada — precio lateral con volumen entrando. Patrón pre-breakout.")

    # ── 3. Price change context for quick analysis ──
    if abs(daily_price_chg_7d) < 5 and result["vol_spike"]:
        score += 5
        notes.append("🔍 Consolidación + spike de volumen = setup de acumulación institucional")

    # ── Normalize score ──
    score = max(0, min(100, 50 + score))
    result["smart_money_score"] = score

    # ── Signal determination ──
    if score >= 75:
        result["whale_score"]  = score
        result["whale_signal"] = "🐳 ACUMULACIÓN INSTITUCIONAL"
        result["whale_color"]  = "#00ff88"
    elif score >= 62:
        result["whale_score"]  = score
        result["whale_signal"] = "📈 INTERÉS DE BALLENAS"
        result["whale_color"]  = "#66ff99"
    elif score >= 50:
        result["whale_score"]  = score
        result["whale_signal"] = "👀 OBSERVACIÓN ACTIVA"
        result["whale_color"]  = "#ffaa00"
    elif result["distribution_warning"]:
        result["whale_score"]  = score
        result["whale_signal"] = "⛔ SEÑAL DE DISTRIBUCIÓN"
        result["whale_color"]  = "#ff4444"
    else:
        result["whale_score"]  = score
        result["whale_signal"] = "😴 SIN ACTIVIDAD NOTABLE"
        result["whale_color"]  = "#888888"

    result["notes"] = notes
    return result


def _whale_quick_proxy(vol_ratio: float, chg7d: float, chg24h: float,
                       chg30d: float) -> Dict:
    """
    Proxy de actividad de ballenas solo con datos de mercado (sin OHLCV).
    Usado en analyze_coin_quick para dar un indicador rápido.
    """
    score = 50
    signal = "SIN DATOS OHLCV"
    notes = []

    # Volume ratio (24h vol / market cap normalized)
    if vol_ratio >= 3.0:
        score += 20
        notes.append(f"🐳 Volumen {vol_ratio:.1f}× el promedio — actividad institucional posible")
    elif vol_ratio >= 1.5:
        score += 10
        notes.append(f"📈 Volumen elevado {vol_ratio:.1f}× — interés comprador")
    elif vol_ratio < 0.5:
        score -= 10
        notes.append("⚠️ Volumen muy bajo — sin interés institucional")

    # Price action patterns that suggest accumulation
    if -30 <= chg30d <= -10 and abs(chg7d) < 5:
        score += 15
        notes.append("🤫 Corrección + estabilización — patrón de acumulación")
    if chg24h > 0 and chg7d > 0 and vol_ratio > 1.2:
        score += 8
        notes.append("✅ Precio subiendo con volumen — confirmación compradora")

    score = max(0, min(100, score))
    if score >= 70:
        signal = "🐳 POSIBLE INTERÉS INSTITUCIONAL"
        color = "#00ff88"
    elif score >= 55:
        signal = "👀 VIGILAR VOLUMEN"
        color = "#ffaa00"
    else:
        signal = "😴 SIN SEÑAL"
        color = "#888888"

    return {"whale_score": score, "whale_signal": signal, "whale_color": color,
            "notes": notes, "vol_spike": vol_ratio >= 2.0, "vol_spike_ratio": round(vol_ratio, 2)}


# ─────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────

def _score_rsi(val: float, prev: float) -> Tuple[int, str]:
    rising = val > prev
    if val < 25:
        return 10, "Capitulación extrema — puede rebotar"
    elif val <= 35:
        return (22 if rising else 17), "Sobrevendido" + (" en recuperación ✅" if rising else "")
    elif val <= 45:
        return (20 if rising else 13), "Zona de rebote" + (" ascendente ✅" if rising else "")
    elif val <= 55:
        return 11, "Zona neutral"
    elif val <= 65:
        return 6, "Momentum positivo"
    else:
        return 2, "Sobrecomprado ⚠️"


def _score_macd(macd_val: float, sig_val: float,
                hist: float, hist_prev: float) -> Tuple[int, str]:
    crossover = (macd_val > sig_val) and (hist_prev < 0) and (hist > 0)
    if crossover:
        return 25, "Cruce alcista MACD ✅"
    improving = hist > hist_prev
    if improving and hist < 0 and abs(hist) < abs(hist_prev) * 0.4:
        return 21, "Cruce alcista inminente ⚡"
    elif improving and hist < 0:
        return 15, "Histograma mejorando (aún negativo)"
    elif hist > 0 and improving:
        return 12, "MACD positivo y subiendo"
    elif hist > 0:
        return 8, "MACD positivo"
    else:
        return 3, "MACD bajista"


def _score_trend(price: float, e20: float, e50: float, e200: float,
                 pe20: float, pe50: float) -> Tuple[int, str]:
    golden = e20 > e50 and pe20 <= pe50
    if price > e20 > e50 > e200:
        return 25, "Tendencia alcista confirmada 🚀"
    elif price > e20 > e50:
        return 22, "Sobre EMA20 y EMA50 ✅"
    elif golden:
        return 22, "Golden Cross reciente ⭐"
    elif price > e20 and price < e50:
        return 16, "Sobre EMA20, bajo EMA50"
    elif price < e20 and price / e20 > 0.97:
        return 18, "Muy cerca de EMA20 — posible ruptura"
    elif price < e20 and price > e50:
        return 12, "Entre EMAs (consolidación)"
    elif price > e200:
        return 8, "Sobre EMA200 (soporte largo plazo)"
    else:
        return 4, "Bajo todas las EMAs"


def _score_bb(pct_b: float, width: float) -> Tuple[int, str]:
    squeeze = width < 0.05
    if squeeze and pct_b < 0.2:
        return 15, "Compresión + Precio bajo — Setup ideal ⭐"
    elif squeeze:
        return 11, "Compresión Bollinger — movimiento explosivo cerca"
    elif pct_b < 0.1:
        return 12, "Precio en banda inferior — rebote técnico"
    elif pct_b <= 0.3:
        return 8, "Zona baja de Bollinger"
    elif pct_b <= 0.5:
        return 5, "Zona media"
    else:
        return 2, "Zona alta — cuidado"


def _score_volume(ratio: float) -> Tuple[int, str]:
    if ratio >= 3.0:
        return 10, "Volumen muy alto — posible acumulación institucional"
    elif ratio >= 2.0:
        return 8, "Volumen elevado — interés comprador fuerte"
    elif ratio >= 1.4:
        return 6, "Volumen sobre promedio"
    elif ratio >= 0.8:
        return 4, "Volumen en promedio"
    else:
        return 1, "Volumen bajo — falta de interés"


def _score_ath_distance(ath_dist_pct: float) -> Tuple[int, str]:
    """ath_dist_pct is negative (price is below ATH)."""
    d = abs(ath_dist_pct)
    if d >= 80:
        return 25, f"Muy alejado del ATH ({d:.0f}% abajo) — máximo potencial"
    elif d >= 60:
        return 20, f"Significativamente bajo ATH ({d:.0f}%)"
    elif d >= 40:
        return 15, f"Moderadamente bajo ATH ({d:.0f}%)"
    elif d >= 20:
        return 8, f"Cercano al ATH ({d:.0f}% abajo)"
    else:
        return 3, f"Casi en ATH ({d:.0f}% abajo) — poco upside"


# ─────────────────────────────────────────
# Confirmation notes generator
# ─────────────────────────────────────────

def _confirmations_quick(rsi_val: float, chg7d: float, chg24h: float,
                         vol_ratio: float) -> List[str]:
    notes = []

    # RSI
    if rsi_val < 30:
        notes.append("⏳ RSI sobrevendido — esperar que supere 35–40 con vela verde")
    elif rsi_val <= 50:
        notes.append("📈 RSI en zona de recuperación — buscar divergencia alcista")
    elif rsi_val > 70:
        notes.append("⚠️ RSI sobrecomprado — esperar corrección antes de entrar")
    else:
        notes.append("✅ RSI en zona neutral-alcista")

    # Price action
    if chg7d < -10:
        notes.append("⏳ Esperar 2–3 sesiones con mínimos ascendentes")
    elif -5 <= chg7d <= 5:
        notes.append("🎯 Precio consolidando — vigilar ruptura del rango superior")
    elif chg7d > 10:
        notes.append("⚡ Momentum alcista activo — confirmar con volumen sostenido")

    # Volume
    if vol_ratio < 0.5:
        notes.append("⚠️ Volumen muy bajo — esperar entrada de volumen (>1.5× promedio)")
    elif vol_ratio >= 1.5:
        notes.append("✅ Volumen elevado — señal positiva")

    notes.append("🕯️ Patrón ideal: Vela envolvente alcista en soporte con volumen ×2")
    notes.append("📊 Confirmar: 2 cierres diarios sobre resistencia principal")
    return notes


def _confirmations_full(rsi_val: float, macd_val: float, sig_val: float,
                        hist: float, stoch_k: float, bb_pct: float,
                        bb_width: float) -> List[str]:
    notes = []

    notes.append("═══ 📋 SEÑALES DE ENTRADA ═══")
    if rsi_val < 30:
        notes.append("⏳ RSI: Esperar que supere 35 con cierre verde en diario")
    elif rsi_val <= 45:
        notes.append("✅ RSI: En zona de compra — confirmar con volumen")
    elif rsi_val <= 60:
        notes.append("⚠️ RSI: Zona neutral — entrada válida solo si resto confirma")
    else:
        notes.append("🚨 RSI: Zona alta — esperar corrección")

    if macd_val < sig_val and hist < 0:
        gap = abs(hist / (abs(macd_val) or 1)) * 100
        if gap < 15:
            notes.append("⚡ MACD: Cruce alcista muy próximo (<15% gap)")
        else:
            notes.append("⏳ MACD: Esperar cruce alcista (MACD sobre señal)")
    else:
        notes.append("✅ MACD: Cruce alcista activo o positivo")

    if stoch_k < 20:
        notes.append("⏳ Estocástico: Sobrevendido — esperar cruce %K > %D en zona baja")
    elif stoch_k > 80:
        notes.append("⚠️ Estocástico: Sobrecomprado — no entrar ahora")
    else:
        notes.append("✅ Estocástico: Zona neutral-bullish")

    if bb_width < 0.04:
        notes.append("🎯 Bollinger: Compresión extrema — ruptura explosiva inminente")
    elif bb_pct < 0.2:
        notes.append("✅ Bollinger: Precio en soporte de banda inferior")

    notes.append("")
    notes.append("═══ 🕯️ PATRONES DE VELA A ESPERAR ═══")
    notes.append("1. Vela Envolvente Alcista (Bullish Engulfing)")
    notes.append("2. Martillo (Hammer) en soporte clave")
    notes.append("3. Estrella de la Mañana (Morning Star) en 3 velas")
    notes.append("4. 3 Soldados Blancos con cierre en máximos")
    notes.append("5. Harami alcista tras consolidación larga")

    notes.append("")
    notes.append("═══ 📊 CONFIRMACIÓN DE VOLUMEN ═══")
    notes.append("• Volumen mínimo para confirmar: 150% del promedio 20 sesiones")
    notes.append("• Volumen ideal en ruptura: 200–300% del promedio")
    notes.append("• Señal falsa si ruptura ocurre con volumen < 80% promedio")

    notes.append("")
    notes.append("═══ ⏱️ MARCOS DE TIEMPO ═══")
    notes.append("• Señal inicial: gráfico de 4H")
    notes.append("• Validación: gráfico Diario (D1)")
    notes.append("• Entrada agresiva: ruptura 4H con volumen")
    notes.append("• Entrada conservadora: cierre diario sobre resistencia")
    return notes


# ─────────────────────────────────────────
# Main analysis functions
# ─────────────────────────────────────────

def analyze_coin_quick(coin: Dict) -> Dict:
    """
    Quick analysis from market data + sparkline only.
    No extra API calls needed.
    """
    coin_id = coin.get("id", "")
    symbol = coin.get("symbol", "").upper()
    current_price = coin.get("current_price") or 0
    market_cap = coin.get("market_cap") or 0
    volume_24h = coin.get("total_volume") or 0
    ath = coin.get("ath") or current_price
    ath_date = coin.get("ath_date", "")

    chg1h = coin.get("price_change_percentage_1h_in_currency") or 0
    chg24h = coin.get("price_change_percentage_24h") or 0
    chg7d = coin.get("price_change_percentage_7d_in_currency") or 0
    chg30d = coin.get("price_change_percentage_30d_in_currency") or 0

    sparkline = (coin.get("sparkline_in_7d") or {}).get("price", [])

    # Quick RSI from sparkline (7d hourly)
    quick_rsi = 50.0
    if len(sparkline) >= 20:
        sp = pd.Series(sparkline, dtype=float)
        rs = calc_rsi(sp, 14)
        v = rs.iloc[-1]
        if not np.isnan(v):
            quick_rsi = float(v)
    quick_rsi_prev = quick_rsi - chg7d * 0.1  # rough estimation of direction

    ath_distance = round((current_price / ath - 1) * 100, 2) if ath > 0 else 0

    # Volume ratio: normalized to market average daily turnover (~10% of mcap)
    vol_to_mcap = volume_24h / market_cap if market_cap > 0 else 0
    vol_ratio = min(vol_to_mcap / 0.10, 6.0)

    # ─── Scoring ───
    rsi_s, rsi_note = _score_rsi(quick_rsi, quick_rsi_prev)

    # Momentum from price changes
    if -50 <= chg30d <= -20 and -5 <= chg7d <= 10:
        mom_s, mom_note = 22, "Corrección sana + Estabilización (acumulación) ✅"
    elif -20 < chg30d <= -5 and chg7d >= 0:
        mom_s, mom_note = 18, "Corrección moderada + Recuperación"
    elif -5 < chg30d <= 5 and chg7d > 0:
        mom_s, mom_note = 15, "Consolidación + Momentum positivo"
    elif chg7d > 15:
        mom_s, mom_note = 9, "Movimiento alcista (puede estar tarde)"
    elif chg30d < -50:
        mom_s, mom_note = 8, "Caída fuerte (posible capitulación)"
    else:
        mom_s, mom_note = 8, "Sin patrón claro"

    ath_s, ath_note = _score_ath_distance(ath_distance)
    vol_s, vol_note = _score_volume(vol_ratio)
    vol_s = min(vol_s, 15)

    # Market cap — smaller = more upside potential
    if market_cap < 100_000_000:
        mc_s, mc_note = 10, "Micro-cap — máximo potencial"
    elif market_cap < 500_000_000:
        mc_s, mc_note = 8, "Small-cap — alto potencial"
    elif market_cap < 5_000_000_000:
        mc_s, mc_note = 5, "Mid-cap — potencial moderado"
    elif market_cap < 50_000_000_000:
        mc_s, mc_note = 3, "Large-cap — menor upside %"
    else:
        mc_s, mc_note = 1, "Mega-cap — muy bajo upside %"

    total_raw = rsi_s + mom_s + ath_s + vol_s + mc_s
    max_raw = 22 + 22 + 25 + 15 + 10  # approximate max
    bull_score = round((total_raw / max_raw) * 100)
    bull_score = max(0, min(100, bull_score))

    # Bull status
    if bull_score >= 72:
        status, label, signal = "hot", "🔥 ALTO POTENCIAL", "PRE-BULL"
    elif bull_score >= 55:
        status, label, signal = "warm", "⚡ CALENTANDO", "ACUMULACIÓN"
    elif bull_score >= 38:
        status, label, signal = "neutral", "⚠️ NEUTRAL", "OBSERVAR"
    else:
        status, label, signal = "cold", "❄️ BAJISTA", "EVITAR"

    # Quick price targets
    d = abs(ath_distance)
    if d >= 70:
        up_c, up_m, up_a = round(d * 0.35, 1), round(d * 0.65, 1), round(d * 1.0, 1)
    elif d >= 40:
        up_c, up_m, up_a = round(d * 0.3, 1), round(d * 0.55, 1), round(d * 0.85, 1)
    else:
        up_c, up_m, up_a = round(d * 0.25, 1), round(d * 0.45, 1), round(d * 0.70, 1)

    # Downside estimate
    if chg7d < -5:
        down_est = max(round(chg7d * 1.5, 1), -50)
    else:
        down_est = max(round(-d * 0.12, 1), -35)

    rr = round(up_m / abs(down_est), 2) if down_est != 0 else 0

    # Volume analysis
    avg_vol_estimate = volume_24h
    bull_volume_estimate = volume_24h * 2.5

    whale_data = _whale_quick_proxy(vol_ratio, chg7d, chg24h, chg30d)

    return {
        "id": coin_id,
        "symbol": symbol,
        "name": coin.get("name", ""),
        "image": coin.get("image", ""),
        "current_price": current_price,
        "market_cap": market_cap,
        "volume_24h": volume_24h,
        "avg_volume_est": avg_vol_estimate,
        "bull_volume_est": bull_volume_estimate,
        "ath": ath,
        "ath_date": ath_date,
        "ath_distance": ath_distance,
        "price_change": {"1h": chg1h, "24h": chg24h, "7d": chg7d, "30d": chg30d},
        "market_cap_rank": coin.get("market_cap_rank", 999),
        "bull_score": bull_score,
        "bull_status": status,
        "bull_label": label,
        "signal": signal,
        "whale": whale_data,
        "breakdown": {
            "rsi": {"score": rsi_s, "value": round(quick_rsi, 1), "note": rsi_note},
            "momentum": {"score": mom_s, "note": mom_note, "7d": chg7d, "30d": chg30d},
            "ath_dist": {"score": ath_s, "note": ath_note, "distance": ath_distance},
            "volume": {"score": vol_s, "note": vol_note, "ratio": round(vol_ratio, 2)},
            "marketcap": {"score": mc_s, "note": mc_note},
        },
        "targets": {
            "conservative_pct": up_c,
            "moderate_pct": up_m,
            "aggressive_pct": up_a,
            "downside_pct": down_est,
            "conservative_price": round(current_price * (1 + up_c / 100), 8),
            "moderate_price": round(current_price * (1 + up_m / 100), 8),
            "aggressive_price": round(current_price * (1 + up_a / 100), 8),
        },
        "rr_ratio": rr,
        "scenarios": {
            "bull_probability": min(max(bull_score, 5), 95),
            "error_probability": 100 - min(max(bull_score, 5), 95),
            "bull_upside_pct": up_m,
            "error_downside_pct": down_est,
        },
        "confirmations": _confirmations_quick(quick_rsi, chg7d, chg24h, vol_ratio),
        "has_full_analysis": False,
    }


def enrich_with_ohlcv(result: Dict) -> Dict:
    """
    Fetch OHLCV and volume chart for a coin and compute full TA.
    Replaces quick estimates with precise calculations.
    """
    coin_id = result["id"]
    df = fetch_ohlcv(coin_id, days=30)
    if df is None or len(df) < 40:
        return result

    vols = fetch_volume_chart(coin_id, days=30)
    hourly = fetch_hourly_chart(coin_id, days=14)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    current_price = float(close.iloc[-1])

    # ─── Indicators ───
    rsi_s = calc_rsi(close)
    rsi_val = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50.0
    rsi_prev = float(rsi_s.iloc[-3]) if len(rsi_s) > 3 else rsi_val

    macd_line, sig_line, hist = calc_macd(close)
    ml = float(macd_line.iloc[-1])
    sl = float(sig_line.iloc[-1])
    hl = float(hist.iloc[-1])
    hp = float(hist.iloc[-3]) if len(hist) > 3 else hl

    bb_u, bb_m, bb_l, bb_w, bb_p = calc_bb(close)
    bb_width = float(bb_w.iloc[-1]) if not np.isnan(bb_w.iloc[-1]) else 0.1
    bb_pct = float(bb_p.iloc[-1]) if not np.isnan(bb_p.iloc[-1]) else 0.5

    e20 = calc_ema(close, 20)
    e50 = calc_ema(close, 50)
    e200_period = min(180, len(close) - 1)
    e200 = calc_ema(close, e200_period)

    ema20 = float(e20.iloc[-1])
    ema50 = float(e50.iloc[-1])
    ema200 = float(e200.iloc[-1])
    prev_e20 = float(e20.iloc[-5]) if len(e20) > 5 else ema20
    prev_e50 = float(e50.iloc[-5]) if len(e50) > 5 else ema50

    stk, std_d = calc_stoch(high, low, close)
    stoch_k = float(stk.iloc[-1]) if not np.isnan(stk.iloc[-1]) else 50.0
    stoch_d = float(std_d.iloc[-1]) if not np.isnan(std_d.iloc[-1]) else 50.0

    atr_val = float(calc_atr(high, low, close).iloc[-1])

    # Volume analysis from series
    if vols is not None and len(vols) >= 7:
        avg_vol = float(vols.tail(20).mean())
        cur_vol = float(vols.iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        avg_vol = result["volume_24h"]
        cur_vol = avg_vol
        vol_ratio_mcap = result["volume_24h"] / result["market_cap"] if result["market_cap"] > 0 else 0
        vol_ratio = min(vol_ratio_mcap / 0.10, 6.0)

    # ─── Scoring ───
    rsi_score, rsi_note = _score_rsi(rsi_val, rsi_prev)
    macd_score, macd_note = _score_macd(ml, sl, hl, hp)
    trend_score, trend_note = _score_trend(current_price, ema20, ema50, ema200, prev_e20, prev_e50)
    bb_score, bb_note = _score_bb(bb_pct, bb_width)
    vol_score, vol_note = _score_volume(vol_ratio)
    vol_score = min(vol_score, 10)

    total_raw = rsi_score + macd_score + trend_score + bb_score + vol_score
    max_raw = 22 + 25 + 25 + 15 + 10
    bull_score = round((total_raw / max_raw) * 100)
    bull_score = max(0, min(100, bull_score))

    # Bull status
    if bull_score >= 72:
        status, label, signal = "hot", "🔥 ALTO POTENCIAL", "PRE-BULL"
    elif bull_score >= 55:
        status, label, signal = "warm", "⚡ CALENTANDO", "ACUMULACIÓN"
    elif bull_score >= 38:
        status, label, signal = "neutral", "⚠️ NEUTRAL", "OBSERVAR"
    else:
        status, label, signal = "cold", "❄️ BAJISTA", "EVITAR"

    # ─── Price targets (Fibonacci) ───
    recent_low = float(close.tail(30).min())
    recent_high = float(close.tail(30).max())
    fib_range = recent_high - recent_low

    t_conservative = current_price + fib_range * 0.618
    t_moderate = current_price + fib_range * 1.0
    t_aggressive = current_price + fib_range * 1.618

    up_c = round((t_conservative / current_price - 1) * 100, 1)
    up_m = round((t_moderate / current_price - 1) * 100, 1)
    up_a = round((t_aggressive / current_price - 1) * 100, 1)
    up_ath = round((result["ath"] / current_price - 1) * 100, 1) if result["ath"] > current_price else 0

    stop_loss = current_price - 2.0 * atr_val
    support = recent_low
    down_pct = round((stop_loss / current_price - 1) * 100, 1)
    rr = round(up_m / abs(down_pct), 2) if down_pct != 0 else 0

    # Phase detection
    if rsi_val < 35 and hl > hp:
        phase = "Fase de Acumulación"
    elif bull_score >= 70 and hl > 0:
        phase = "Fase Pre-Ruptura"
    elif bull_score >= 55:
        phase = "Fase de Recuperación"
    elif rsi_val > 65:
        phase = "Fase de Distribución"
    else:
        phase = "Sin Tendencia Clara"

    result.update({
        "current_price": current_price,
        "bull_score": bull_score,
        "bull_status": status,
        "bull_label": label,
        "signal": signal,
        "phase": phase,
        "breakdown": {
            "rsi": {"score": rsi_score, "max": 22, "value": round(rsi_val, 1), "note": rsi_note},
            "macd": {"score": macd_score, "max": 25, "note": macd_note,
                     "crossover": ml > sl and hp < 0 and hl > 0},
            "trend": {"score": trend_score, "max": 25, "note": trend_note,
                      "vs_ema20": round((current_price / ema20 - 1) * 100, 2),
                      "vs_ema50": round((current_price / ema50 - 1) * 100, 2)},
            "bollinger": {"score": bb_score, "max": 15, "note": bb_note,
                          "squeeze": bb_width < 0.05, "pct_b": round(bb_pct, 3)},
            "volume": {"score": vol_score, "max": 10, "note": vol_note,
                       "ratio": round(vol_ratio, 2)},
        },
        "targets": {
            "conservative_pct": up_c,
            "moderate_pct": up_m,
            "aggressive_pct": up_a,
            "to_ath_pct": up_ath,
            "downside_pct": down_pct,
            "conservative_price": round(t_conservative, 8),
            "moderate_price": round(t_moderate, 8),
            "aggressive_price": round(t_aggressive, 8),
            "support": round(support, 8),
            "stop_loss": round(stop_loss, 8),
        },
        "volume_24h": cur_vol,
        "avg_volume": avg_vol,
        "bull_volume_est": avg_vol * 2.5,
        "vol_ratio": round(vol_ratio, 2),
        "rr_ratio": rr,
        "scenarios": {
            "bull_probability": min(max(bull_score, 5), 95),
            "error_probability": 100 - min(max(bull_score, 5), 95),
            "bull_upside_pct": up_m,
            "error_downside_pct": down_pct,
        },
        "indicators_full": {
            "rsi": round(rsi_val, 2),
            "macd": round(ml, 8),
            "signal": round(sl, 8),
            "histogram": round(hl, 8),
            "ema20": round(ema20, 8),
            "ema50": round(ema50, 8),
            "ema200": round(ema200, 8),
            "stoch_k": round(stoch_k, 2),
            "stoch_d": round(stoch_d, 2),
            "atr": round(atr_val, 8),
            "bb_upper": round(float(bb_u.iloc[-1]), 8),
            "bb_lower": round(float(bb_l.iloc[-1]), 8),
            "bb_width": round(bb_width, 4),
            "bb_pct": round(bb_pct, 3),
        },
        "ohlcv": df.to_dict("records"),
        "volume_series": vols.tolist() if vols is not None else [],
        "confirmations": _confirmations_full(rsi_val, ml, sl, hl, stoch_k, bb_pct, bb_width),
        "has_full_analysis": True,
    })

    # ── Full whale analysis (with hourly OHLCV) ──
    whale_full = analyze_whale_activity(
        hourly_df=hourly,
        daily_vols=vols,
        daily_price_chg_7d=result["price_change"]["7d"],
    )
    result["whale"] = whale_full
    return result


def fmt_price(price: float) -> str:
    """Format price with appropriate decimal places."""
    if price == 0:
        return "$0"
    elif price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    elif price >= 0.001:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"


def fmt_large(n: float) -> str:
    """Format large numbers (volumes, market caps)."""
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    elif n >= 1e9:
        return f"${n/1e9:.2f}B"
    elif n >= 1e6:
        return f"${n/1e6:.2f}M"
    elif n >= 1e3:
        return f"${n/1e3:.2f}K"
    else:
        return f"${n:.2f}"


# ─────────────────────────────────────────
# SCALPING ENGINE — 15/30 min entries
# ─────────────────────────────────────────

def fetch_scalp_candles(coin_id: str) -> Optional[pd.DataFrame]:
    """
    Fetches 1-day OHLC data from CoinGecko → 30-min candles (48 per day).
    Used for 15-30 min scalping analysis.
    TTL = 3 min so data stays fresh for scalpers.
    """
    key = f"scalp_{coin_id}"
    cached = _cache_get(key, 180)   # 3-min cache
    if cached is not None:
        return cached

    data = _api(f"coins/{coin_id}/ohlc", {"vs_currency": "usd", "days": 1})
    if not data or len(data) < 8:
        return None

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)

    # Synthetic volume from price range (CoinGecko ohlc has no volume)
    df["range"] = df["high"] - df["low"]
    df["volume"] = df["range"] / df["close"].replace(0, np.nan)

    _cache_set(key, df)
    time.sleep(0.4)
    return df


def _roc(series: pd.Series, n: int = 3) -> float:
    """Rate of Change over n periods."""
    if len(series) < n + 1:
        return 0.0
    v_now = float(series.iloc[-1])
    v_prev = float(series.iloc[-1 - n])
    return ((v_now - v_prev) / abs(v_prev) * 100) if v_prev != 0 else 0.0


def analyze_scalp(coin_id: str, current_price: float, volume_24h: float,
                  market_data: Dict) -> Dict:
    """
    Analiza una moneda para entrada de scalping en 15-30 minutos.
    Usa candles de 30 min (1 día CoinGecko) y calcula:
      - RSI(7): momentum ultra-rápido
      - EMA5 / EMA10 cross: cruce de corto plazo
      - Bollinger (10,2): squeeze y posición del precio
      - Momentum (ROC 3 períodos)
      - Stochastic(5,3): señal de entrada/salida rápida
      - Volume proxy: rango de vela relativo
      - VWAP intradiario: precio vs precio justo del día
    """
    result = {
        "coin_id": coin_id,
        "scalp_score": 0,
        "scalp_signal": "SIN DATOS",
        "scalp_color": "#555",
        "entry_price": current_price,
        "target_15m": current_price,
        "target_30m": current_price,
        "stop_loss": current_price,
        "rsi7": None,
        "ema5": None,
        "ema10": None,
        "ema_cross": "neutral",
        "stoch_k": None,
        "stoch_d": None,
        "bb_pct": None,
        "bb_squeeze": False,
        "momentum_roc": 0.0,
        "vwap_intra": current_price,
        "vwap_signal": "neutral",
        "candles": [],
        "reasons": [],
        "urgency": "LOW",         # LOW / MEDIUM / HIGH / FIRE
        "window": "30m+",         # cuándo entraría el bot
    }

    df = fetch_scalp_candles(coin_id)
    if df is None or len(df) < 12:
        return result

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Indicadores scalping ──
    rsi7      = calc_rsi(close, 7)
    ema5      = calc_ema(close, 5)
    ema10     = calc_ema(close, 10)
    ema20     = calc_ema(close, 20)
    bb_u, bb_m, bb_l, bb_w, bb_p = calc_bb(close, 10, 2)
    stoch_k, stoch_d = calc_stoch(high, low, close, 5, 3)
    vwap_s    = calc_vwap(close, vol)
    atr_s     = calc_atr(high, low, close, 7)

    rsi_v   = float(rsi7.dropna().iloc[-1]) if not rsi7.dropna().empty else 50.0
    ema5_v  = float(ema5.iloc[-1])
    ema10_v = float(ema10.iloc[-1])
    ema20_v = float(ema20.iloc[-1])
    bb_p_v  = float(bb_p.dropna().iloc[-1]) if not bb_p.dropna().empty else 0.5
    bb_w_v  = float(bb_w.dropna().iloc[-1]) if not bb_w.dropna().empty else 0.05
    stk_v   = float(stoch_k.dropna().iloc[-1]) if not stoch_k.dropna().empty else 50.0
    std_v   = float(stoch_d.dropna().iloc[-1]) if not stoch_d.dropna().empty else 50.0
    vwap_v  = float(vwap_s.dropna().iloc[-1]) if not vwap_s.dropna().empty else current_price
    atr_v   = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else current_price * 0.005
    roc_v   = _roc(close, 3)

    # Previous EMA5/10 for cross detection
    ema5_prev  = float(ema5.iloc[-2]) if len(ema5) >= 2 else ema5_v
    ema10_prev = float(ema10.iloc[-2]) if len(ema10) >= 2 else ema10_v

    result.update({
        "rsi7": round(rsi_v, 1),
        "ema5": round(ema5_v, 8),
        "ema10": round(ema10_v, 8),
        "stoch_k": round(stk_v, 1),
        "stoch_d": round(std_v, 1),
        "bb_pct": round(bb_p_v, 3),
        "bb_squeeze": bb_w_v < 0.03,
        "momentum_roc": round(roc_v, 3),
        "vwap_intra": round(vwap_v, 8),
        "candles": df.tail(16).to_dict("records"),
        "atr": round(atr_v, 8),
    })

    # ── Scoring ──
    score = 0
    reasons = []

    # 1. RSI(7) — peso 25pts
    if rsi_v < 25:
        score += 25
        reasons.append("🔴 RSI(7) MUY SOBREVENDIDO (<25) — rebote inminente")
    elif rsi_v < 35:
        score += 20
        reasons.append(f"🟠 RSI(7) sobrevendido ({rsi_v:.1f}) — zona de compra")
    elif 35 <= rsi_v < 50:
        score += 12
        reasons.append(f"🟡 RSI(7) en recuperación ({rsi_v:.1f}) — momentum positivo")
    elif 50 <= rsi_v < 60:
        score += 8
        reasons.append(f"🟢 RSI(7) alcista ({rsi_v:.1f}) — tendencia intacta")
    elif rsi_v > 75:
        score -= 5
        reasons.append(f"⚠️ RSI(7) sobrecomprado ({rsi_v:.1f}) — no entrar, esperar corrección")

    # 2. EMA5 / EMA10 cross — peso 25pts
    bullish_cross = ema5_v > ema10_v and ema5_prev <= ema10_prev
    bearish_cross = ema5_v < ema10_v and ema5_prev >= ema10_prev
    if bullish_cross:
        score += 25
        result["ema_cross"] = "bullish_cross"
        reasons.append("✅ CRUCE ALCISTA EMA5 > EMA10 — señal de entrada NOW")
    elif ema5_v > ema10_v and ema10_v > ema20_v:
        score += 18
        result["ema_cross"] = "bullish_aligned"
        reasons.append("✅ EMAs alineadas al alza (5>10>20) — tendencia fuerte")
    elif ema5_v > ema10_v:
        score += 10
        result["ema_cross"] = "bullish"
        reasons.append("🟡 EMA5 sobre EMA10 — momentum a favor")
    elif bearish_cross:
        score -= 15
        result["ema_cross"] = "bearish_cross"
        reasons.append("❌ CRUCE BAJISTA EMA5 < EMA10 — evitar entrada long")
    else:
        result["ema_cross"] = "bearish"
        reasons.append("🔴 EMA5 bajo EMA10 — tendencia bajista a corto plazo")

    # 3. Stochastic(5,3) — peso 20pts
    if stk_v < 20 and std_v < 20:
        score += 20
        reasons.append(f"✅ Estocástico sobrevendido ({stk_v:.0f}/{std_v:.0f}) — zona de rebote")
    elif stk_v > std_v and stk_v < 50:
        score += 14
        reasons.append(f"🟡 Estocástico cruzando al alza ({stk_v:.0f}) — entrada temprana")
    elif stk_v > 80:
        score -= 8
        reasons.append(f"⚠️ Estocástico sobrecomprado ({stk_v:.0f}) — momentum agotado")

    # 4. Bollinger Bands — peso 15pts
    if bb_p_v < 0.15:
        score += 15
        reasons.append("✅ Precio en banda inferior BB — zona de reversal")
    elif bb_p_v < 0.30:
        score += 10
        reasons.append("🟡 Precio en zona baja BB — potencial rebote")
    elif bb_w_v < 0.02:
        score += 8
        reasons.append("🔵 BB SQUEEZE extremo — expansión violenta inminente")
    elif bb_w_v < 0.03:
        score += 5
        reasons.append("🔵 BB comprimido — breakout próximo")
    elif bb_p_v > 0.85:
        score -= 5
        reasons.append("⚠️ Precio en banda superior BB — sobreextendido")

    # 5. Momentum ROC — peso 15pts
    if roc_v > 1.5:
        score += 15
        reasons.append(f"🚀 Momentum fuerte +{roc_v:.2f}% en 3 velas — entrada en breakout")
    elif roc_v > 0.5:
        score += 10
        reasons.append(f"📈 Momentum positivo +{roc_v:.2f}% — tendencia acelerando")
    elif roc_v < -2.0:
        score -= 10
        reasons.append(f"📉 Caída fuerte {roc_v:.2f}% — esperar estabilización")
    elif -0.3 <= roc_v <= 0.3:
        score += 5
        reasons.append("⏸️ Precio consolidando — posible acumulación pre-breakout")

    # 6. VWAP intradiario — peso 10pts bonus
    vwap_gap = (current_price - vwap_v) / vwap_v * 100 if vwap_v > 0 else 0
    if vwap_gap < -1.0:
        score += 10
        result["vwap_signal"] = "under"
        reasons.append(f"💎 Precio {abs(vwap_gap):.1f}% BAJO VWAP — infravalorado intradiario")
    elif -1.0 <= vwap_gap <= 0.5:
        score += 5
        result["vwap_signal"] = "near"
        reasons.append("🟡 Precio cerca del VWAP — zona justa de entrada")
    elif vwap_gap > 2.5:
        score -= 5
        result["vwap_signal"] = "over"
        reasons.append(f"⚠️ Precio {vwap_gap:.1f}% SOBRE VWAP — caro intradiario, esperar pullback")
    else:
        result["vwap_signal"] = "neutral"

    score = max(0, min(100, score))
    result["scalp_score"] = score
    result["reasons"] = reasons

    # ── Target y Stop ──
    atr_mult_tp = 1.5   # 15-min target
    atr_mult_sl = 1.0   # stop loss
    if score >= 70:
        target_15 = current_price * (1 + atr_v / current_price * atr_mult_tp)
        target_30 = current_price * (1 + atr_v / current_price * atr_mult_tp * 2)
        stop      = current_price * (1 - atr_v / current_price * atr_mult_sl)
    else:
        target_15 = current_price * 1.005
        target_30 = current_price * 1.01
        stop      = current_price * 0.99

    result["target_15m"] = round(target_15, 8)
    result["target_30m"] = round(target_30, 8)
    result["stop_loss"]  = round(stop, 8)

    # ── Señal y urgencia ──
    if score >= 75:
        result["scalp_signal"] = "🔥 ENTRADA AHORA"
        result["scalp_color"]  = "#00ff88"
        result["urgency"]      = "FIRE"
        result["window"]       = "0-15 min"
    elif score >= 60:
        result["scalp_signal"] = "⚡ PRÓXIMA ENTRADA"
        result["scalp_color"]  = "#ffdd00"
        result["urgency"]      = "HIGH"
        result["window"]       = "15-30 min"
    elif score >= 45:
        result["scalp_signal"] = "👀 OBSERVAR"
        result["scalp_color"]  = "#ffaa00"
        result["urgency"]      = "MEDIUM"
        result["window"]       = "30-60 min"
    elif score >= 30:
        result["scalp_signal"] = "⏳ ESPERAR"
        result["scalp_color"]  = "#888"
        result["urgency"]      = "LOW"
        result["window"]       = "1h+"
    else:
        result["scalp_signal"] = "🚫 NO OPERAR"
        result["scalp_color"]  = "#ff4444"
        result["urgency"]      = "LOW"
        result["window"]       = "—"

    return result


def batch_scalp_scan(coins: List[Dict], top_n: int = 30) -> List[Dict]:
    """
    Escanea las top_n monedas más líquidas y retorna las mejores oportunidades
    de scalping ordenadas por scalp_score DESC.
    Solo analiza monedas con volume_24h > 10M USD para asegurar liquidez.
    """
    # Filtrar por liquidez mínima para scalping
    liquid = [c for c in coins if c.get("volume_24h", 0) >= 10_000_000]
    # Ordenar por volumen desc para tomar las más activas
    liquid.sort(key=lambda x: x.get("volume_24h", 0), reverse=True)
    targets = liquid[:top_n]

    results = []
    for i, c in enumerate(targets):
        scalp = analyze_scalp(
            c["id"], c["current_price"], c.get("volume_24h", 0), c
        )
        scalp["name"]    = c["name"]
        scalp["symbol"]  = c["symbol"]
        scalp["volume_24h"] = c.get("volume_24h", 0)
        scalp["market_cap"] = c.get("market_cap", 0)
        results.append(scalp)
        # Rate limiting: no más de 2 req/seg en free tier
        if i < len(targets) - 1:
            time.sleep(0.5)

    results.sort(key=lambda x: x["scalp_score"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRADING SIGNALS — Strategy Engine
# ─────────────────────────────────────────────────────────────────────────────

def macd_strategy_signal(df: pd.DataFrame) -> Dict:
    """
    MACD strategy with fast=3, slow=15, signal=3.
    Detects bullish crossovers and strong bullish momentum.
    """
    result = {
        "active": False,
        "direction": "NEUTRAL",
        "strength": 0,
        "note": "Sin señal MACD",
        "macd_val": 0.0,
        "signal_val": 0.0,
        "hist_val": 0.0,
    }

    if df is None or len(df) < 20:
        return result

    close = df["close"]
    macd_line, sig_line, hist = calc_macd(close, fast=3, slow=15, signal=3)

    if len(macd_line) < 3:
        return result

    macd_cur = float(macd_line.iloc[-1])
    macd_prev = float(macd_line.iloc[-2])
    sig_cur = float(sig_line.iloc[-1])
    sig_prev = float(sig_line.iloc[-2])
    hist_cur = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])
    hist_prev2 = float(hist.iloc[-3]) if len(hist) >= 3 else hist_prev

    result["macd_val"] = macd_cur
    result["signal_val"] = sig_cur
    result["hist_val"] = hist_cur

    ema20 = calc_ema(close, 20)
    price_above_ema20 = float(close.iloc[-1]) > float(ema20.iloc[-1])

    # Bullish crossover: MACD crosses above signal
    bullish_cross = (macd_prev < sig_prev) and (macd_cur > sig_cur)

    # Strong bullish: histogram growing positive for 2+ bars AND price above EMA20
    hist_growing = (hist_cur > 0) and (hist_cur > hist_prev) and (hist_prev > hist_prev2)
    strong_bullish = hist_growing and price_above_ema20

    if strong_bullish:
        result["active"] = True
        result["direction"] = "LONG"
        result["strength"] = 80
        result["note"] = "MACD fuerte: histograma positivo creciente 2+ velas y precio sobre EMA20"
    elif bullish_cross:
        result["active"] = True
        result["direction"] = "LONG"
        result["strength"] = 65
        result["note"] = f"MACD cruzó al alza la línea de señal (MACD: {macd_cur:.4f} > Señal: {sig_cur:.4f})"

    # Bearish check
    bearish_cross = (macd_prev > sig_prev) and (macd_cur < sig_cur)
    if bearish_cross and not result["active"]:
        result["active"] = True
        result["direction"] = "SHORT"
        result["strength"] = 55
        result["note"] = f"MACD cruzó a la baja la línea de señal"

    return result


def rsi_mean_reversion_signal(df: pd.DataFrame) -> Dict:
    """
    RSI(14) mean reversion with VWAP.
    Detects oversold bounces and RSI crossovers from below.
    """
    result = {
        "active": False,
        "direction": "NEUTRAL",
        "strength": 0,
        "note": "Sin señal RSI",
        "rsi_val": 50.0,
        "vwap_val": 0.0,
        "vwap_gap_pct": 0.0,
    }

    if df is None or len(df) < 15:
        return result

    close = df["close"]
    volume = df.get("volume", pd.Series(np.ones(len(df))))
    if "volume" not in df.columns:
        volume = pd.Series(np.ones(len(df)), index=df.index)

    rsi = calc_rsi(close, 14)
    vwap = calc_vwap(close, volume)

    if len(rsi) < 3:
        return result

    rsi_cur = float(rsi.iloc[-1])
    rsi_prev = float(rsi.iloc[-2])
    price_cur = float(close.iloc[-1])
    vwap_cur = float(vwap.iloc[-1]) if not np.isnan(float(vwap.iloc[-1])) else price_cur

    vwap_gap_pct = ((price_cur - vwap_cur) / vwap_cur * 100) if vwap_cur != 0 else 0.0
    price_below_vwap = price_cur < vwap_cur

    result["rsi_val"] = rsi_cur
    result["vwap_val"] = vwap_cur
    result["vwap_gap_pct"] = vwap_gap_pct

    # Bullish: RSI was oversold (prev < 30) and turning up
    oversold_bounce = (rsi_prev < 30) and (rsi_cur > rsi_prev)

    # Bullish: RSI crossing above 35 from below
    rsi_cross_35 = (rsi_prev < 35) and (rsi_cur >= 35)

    # Bearish: RSI was overbought (prev > 70) and turning down
    overbought_drop = (rsi_prev > 70) and (rsi_cur < rsi_prev)
    rsi_cross_65 = (rsi_prev > 65) and (rsi_cur <= 65)

    if oversold_bounce:
        strength = 75 if price_below_vwap else 60
        result["active"] = True
        result["direction"] = "LONG"
        result["strength"] = strength
        result["note"] = (
            f"RSI rebote desde sobreventa: {rsi_prev:.1f} → {rsi_cur:.1f}"
            + (" · Precio bajo VWAP = entrada óptima" if price_below_vwap else "")
        )
    elif rsi_cross_35:
        strength = 65 if price_below_vwap else 52
        result["active"] = True
        result["direction"] = "LONG"
        result["strength"] = strength
        result["note"] = (
            f"RSI cruzó 35 al alza ({rsi_prev:.1f} → {rsi_cur:.1f})"
            + (" · Precio bajo VWAP" if price_below_vwap else "")
        )
    elif overbought_drop or rsi_cross_65:
        result["active"] = True
        result["direction"] = "SHORT"
        result["strength"] = 60
        result["note"] = f"RSI girando desde sobrecompra: {rsi_prev:.1f} → {rsi_cur:.1f}"

    return result


def cvd_divergence_signal(df: pd.DataFrame) -> Dict:
    """
    OBV as CVD proxy. Detects bullish/bearish divergence over last 10 candles.
    """
    result = {
        "active": False,
        "direction": "NEUTRAL",
        "strength": 0,
        "note": "Sin divergencia CVD",
        "divergence_type": "none",
        "obv_slope": 0.0,
        "price_slope": 0.0,
    }

    if df is None or len(df) < 12:
        return result

    close = df["close"]
    volume = df.get("volume", pd.Series(np.ones(len(df))))
    if "volume" not in df.columns:
        volume = pd.Series(np.ones(len(df)), index=df.index)

    obv = calc_obv(close, volume)

    if len(obv) < 12:
        return result

    # Split last 10 candles into two halves of 5
    half_a_price = close.iloc[-10:-5]   # older 5
    half_b_price = close.iloc[-5:]       # recent 5
    half_a_obv = obv.iloc[-10:-5]
    half_b_obv = obv.iloc[-5:]

    price_low_a = float(half_a_price.min())
    price_low_b = float(half_b_price.min())
    obv_low_a = float(half_a_obv.min())
    obv_low_b = float(half_b_obv.min())

    price_high_a = float(half_a_price.max())
    price_high_b = float(half_b_price.max())
    obv_high_a = float(half_a_obv.max())
    obv_high_b = float(half_b_obv.max())

    # Slope calculation using numpy linregress proxy
    x = np.arange(10)
    price_arr = close.iloc[-10:].values.astype(float)
    obv_arr = obv.iloc[-10:].values.astype(float)

    if len(price_arr) == 10 and not np.any(np.isnan(price_arr)):
        price_slope = float(np.polyfit(x, price_arr, 1)[0])
        obv_slope = float(np.polyfit(x, obv_arr, 1)[0])
    else:
        price_slope = 0.0
        obv_slope = 0.0

    result["obv_slope"] = obv_slope
    result["price_slope"] = price_slope

    # Bullish divergence: price making lower low, OBV making higher low
    price_lower_low = price_low_b < price_low_a
    obv_higher_low = obv_low_b > obv_low_a

    # Bearish divergence: price making higher high, OBV making lower high
    price_higher_high = price_high_b > price_high_a
    obv_lower_high = obv_high_b < obv_high_a

    if price_lower_low and obv_higher_low:
        result["active"] = True
        result["direction"] = "LONG"
        result["divergence_type"] = "bullish"
        result["strength"] = 70
        result["note"] = (
            "Divergencia alcista CVD: precio marca mínimo más bajo pero OBV marca mínimo más alto"
            " — presión compradora oculta"
        )
    elif price_higher_high and obv_lower_high:
        result["active"] = True
        result["direction"] = "SHORT"
        result["divergence_type"] = "bearish"
        result["strength"] = 65
        result["note"] = (
            "Divergencia bajista CVD: precio marca máximo más alto pero OBV marca máximo más bajo"
            " — distribución institucional"
        )

    return result


def generate_trade_setup(coin: Dict, df: Optional[pd.DataFrame]) -> Dict:
    """
    Main function: generates a complete trade setup from coin market data and OHLCV df.
    Combines MACD, RSI and CVD signals, calculates entry/SL/TP levels via ATR(7).
    """
    coin_id = coin.get("id", "")
    symbol = coin.get("symbol", "").upper()
    name = coin.get("name", "")
    current_price = coin.get("current_price", 0.0)

    _empty = {
        "coin_id": coin_id,
        "symbol": symbol,
        "name": name,
        "has_signal": False,
        "confidence": 0,
        "direction": "NEUTRAL",
        "strategy": "—",
        "strategy_code": "none",
        "entry_price": current_price,
        "stop_loss": 0.0,
        "sl_pct": 0.0,
        "tp1": 0.0, "tp1_pct": 0.0,
        "tp2": 0.0, "tp2_pct": 0.0,
        "tp3": 0.0, "tp3_pct": 0.0,
        "rr_ratio": 0.0,
        "win_probability": 0.0,
        "loss_probability": 100.0,
        "signal_strength": "DÉBIL",
        "atr": 0.0,
        "rsi_val": 50.0,
        "macd_active": False,
        "rsi_active": False,
        "cvd_active": False,
        "reasons": [],
        "invalidation": "—",
        "notes": "—",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "status": "INACTIVE",
    }

    if df is None or len(df) < 15:
        return _empty

    # ── Step 1: Run all 3 strategies ────────────────────────────────────────
    macd_sig = macd_strategy_signal(df)
    rsi_sig = rsi_mean_reversion_signal(df)
    cvd_sig = cvd_divergence_signal(df)

    macd_active = macd_sig["active"]
    rsi_active = rsi_sig["active"]
    cvd_active = cvd_sig["active"]

    active_signals = [s for s in [macd_sig, rsi_sig, cvd_sig] if s["active"]]
    long_signals = [s for s in active_signals if s["direction"] == "LONG"]
    short_signals = [s for s in active_signals if s["direction"] == "SHORT"]

    # ── Step 2: Pick best / most confluent strategy ─────────────────────────
    if not active_signals:
        return _empty

    # Determine dominant direction
    if len(long_signals) >= len(short_signals):
        dominant_signals = long_signals if long_signals else short_signals
        direction = "LONG"
    else:
        dominant_signals = short_signals
        direction = "SHORT"

    if not dominant_signals:
        return _empty

    best_signal = max(dominant_signals, key=lambda s: s["strength"])
    best_strength = best_signal["strength"]

    # Name the strategy
    num_agree = len(dominant_signals)
    if num_agree >= 3:
        strategy = "MACD + RSI + CVD Confluencia Total"
        strategy_code = "confluence"
    elif num_agree == 2:
        names = []
        if macd_active and macd_sig["direction"] == direction:
            names.append("MACD")
        if rsi_active and rsi_sig["direction"] == direction:
            names.append("RSI")
        if cvd_active and cvd_sig["direction"] == direction:
            names.append("CVD")
        strategy = " + ".join(names) + " Confluencia"
        strategy_code = "confluence"
    elif best_signal is macd_sig:
        strategy = "MACD 3/15/3"
        strategy_code = "macd"
    elif best_signal is rsi_sig:
        strategy = "RSI Mean Reversion"
        strategy_code = "rsi"
    else:
        strategy = "CVD Divergencia"
        strategy_code = "cvd"

    # ── Step 3: ATR(7) for price levels ─────────────────────────────────────
    atr_series = calc_atr(df["high"], df["low"], df["close"], period=7)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else current_price * 0.02
    if np.isnan(atr) or atr == 0:
        atr = current_price * 0.015

    entry_price = current_price

    if direction == "LONG":
        stop_loss = entry_price - atr * 1.2
        tp1 = entry_price + atr * 1.5
        tp2 = entry_price + atr * 2.5
        tp3 = entry_price + atr * 4.0
    else:  # SHORT
        stop_loss = entry_price + atr * 1.2
        tp1 = entry_price - atr * 1.5
        tp2 = entry_price - atr * 2.5
        tp3 = entry_price - atr * 4.0

    sl_pct = (stop_loss - entry_price) / entry_price * 100
    tp1_pct = (tp1 - entry_price) / entry_price * 100
    tp2_pct = (tp2 - entry_price) / entry_price * 100
    tp3_pct = (tp3 - entry_price) / entry_price * 100

    # ── Step 4: R/R ratio ────────────────────────────────────────────────────
    rr_ratio = abs(tp2_pct) / abs(sl_pct) if sl_pct != 0 else 0.0

    # ── Step 5: Confidence ───────────────────────────────────────────────────
    rsi_val = float(rsi_sig["rsi_val"])

    confidence = best_strength
    if num_agree == 2:
        confidence += 10
    elif num_agree >= 3:
        confidence += 15

    if rsi_val < 40 and direction == "LONG":
        confidence += 5
    elif rsi_val > 60 and direction == "SHORT":
        confidence += 5

    # Check 4H trend with EMA50
    ema50 = calc_ema(df["close"], 50)
    if len(ema50) >= 2:
        ema50_slope = float(ema50.iloc[-1]) - float(ema50.iloc[-2])
        if direction == "LONG" and ema50_slope < 0:
            confidence -= 10
        elif direction == "SHORT" and ema50_slope > 0:
            confidence -= 10

    confidence = max(0, min(95, int(confidence)))

    # ── Step 6 & 7: Win/Loss probability ────────────────────────────────────
    win_probability = confidence * 0.85 + 10
    win_probability = max(20.0, min(90.0, win_probability))
    loss_probability = 100.0 - win_probability

    # ── Step 8: Signal strength label ───────────────────────────────────────
    if confidence >= 80:
        signal_strength = "MUY FUERTE"
    elif confidence >= 65:
        signal_strength = "FUERTE"
    elif confidence >= 50:
        signal_strength = "MODERADA"
    else:
        signal_strength = "DÉBIL"

    # ── Step 9: Reasons (Spanish) ────────────────────────────────────────────
    reasons = []
    dir_str = "alcista" if direction == "LONG" else "bajista"

    if macd_active and macd_sig["direction"] == direction:
        reasons.append(f"MACD señal {dir_str}: {macd_sig['note']}")
    if rsi_active and rsi_sig["direction"] == direction:
        reasons.append(f"RSI señal {dir_str}: {rsi_sig['note']}")
    if cvd_active and cvd_sig["direction"] == direction:
        reasons.append(f"CVD/OBV: {cvd_sig['note']}")
    if rr_ratio >= 2.0:
        reasons.append(f"Ratio riesgo/recompensa favorable: {rr_ratio:.2f}× (TP2)")
    if rsi_val < 35 and direction == "LONG":
        reasons.append(f"RSI bajo ({rsi_val:.1f}) sugiere zona de valor — activo no sobrecomprado")
    if num_agree >= 2:
        reasons.append(f"Confluencia de {num_agree} indicadores en la misma dirección aumenta probabilidad")

    vwap_gap = float(rsi_sig.get("vwap_gap_pct", 0.0))
    if vwap_gap < -1.0 and direction == "LONG":
        reasons.append(f"Precio {abs(vwap_gap):.1f}% bajo VWAP — zona de descuento respecto a precio justo")

    if not reasons:
        reasons.append(f"Señal técnica {dir_str} detectada por {strategy}")

    # ── Step 10 & 11: Invalidation and notes ────────────────────────────────
    invalidation = f"Cierre de vela por debajo de {fmt_price(stop_loss)}"
    if direction == "SHORT":
        invalidation = f"Cierre de vela por encima de {fmt_price(stop_loss)}"

    notes = (
        f"Esperar confirmación de vela de 30min cerrando en dirección {dir_str}. "
        f"Entrada LIMIT en {fmt_price(entry_price)} o mejor. "
        f"Gestión: mover SL a breakeven al alcanzar TP1 ({fmt_price(tp1)})."
    )

    # ── has_signal determination ─────────────────────────────────────────────
    has_signal = (confidence >= 45) and len(active_signals) >= 1

    return {
        "coin_id": coin_id,
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "strategy": strategy,
        "strategy_code": strategy_code,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "sl_pct": sl_pct,
        "tp1": tp1,
        "tp1_pct": tp1_pct,
        "tp2": tp2,
        "tp2_pct": tp2_pct,
        "tp3": tp3,
        "tp3_pct": tp3_pct,
        "rr_ratio": rr_ratio,
        "confidence": confidence,
        "win_probability": win_probability,
        "loss_probability": loss_probability,
        "signal_strength": signal_strength,
        "atr": atr,
        "rsi_val": rsi_val,
        "macd_active": macd_active,
        "rsi_active": rsi_active,
        "cvd_active": cvd_active,
        "reasons": reasons,
        "invalidation": invalidation,
        "notes": notes,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "status": "ACTIVE",
        "has_signal": has_signal,
    }


def scan_trade_signals(
    coins: List[Dict],
    min_volume: float = 5_000_000,
    top_n: int = 25,
) -> List[Dict]:
    """
    Scans top coins by volume for trade setups.
    Filters by min_volume, sorts by volume DESC, takes top_n.
    Returns list of trade setups sorted by confidence DESC.
    """
    # Filter by minimum volume
    liquid = [c for c in coins if c.get("total_volume", c.get("volume_24h", 0)) >= min_volume]
    # Sort by volume descending
    liquid.sort(
        key=lambda x: x.get("total_volume", x.get("volume_24h", 0)),
        reverse=True,
    )
    targets = liquid[:top_n]

    results = []
    for coin in targets:
        try:
            df = fetch_scalp_candles(coin["id"])
            setup = generate_trade_setup(coin, df)
            if setup.get("has_signal", False):
                results.append(setup)
        except Exception:
            pass
        time.sleep(0.4)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
