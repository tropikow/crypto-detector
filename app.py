"""
Crypto Bull Detector — Streamlit Dashboard
Evalúa el mercado cripto e identifica activos con mayor probabilidad de bull run.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from engine import (
    fetch_markets, analyze_coin_quick, enrich_with_ohlcv,
    calc_rsi, calc_macd, calc_bb, calc_ema, calc_stoch, calc_atr,
    calc_obv, calc_cmf, calc_vwap,
    fmt_price, fmt_large,
    batch_scalp_scan, analyze_scalp,
)

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Bull Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Header */
.main-header {
    background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #0f3460 100%);
    border: 1px solid #00ff8844;
    border-radius: 12px;
    padding: 22px 30px;
    margin-bottom: 18px;
    text-align: center;
}
.main-title { font-size: 2.2rem; font-weight: 800; color: #00ff88; margin: 0; }
.main-subtitle { color: #aaa; font-size: 0.95rem; margin-top: 4px; }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.85rem;
    min-width: 50px;
    text-align: center;
}
.score-hot { background: #00ff8822; color: #00ff88; border: 1px solid #00ff88; }
.score-warm { background: #ffaa0022; color: #ffaa00; border: 1px solid #ffaa00; }
.score-neutral { background: #5588ff22; color: #5588ff; border: 1px solid #5588ff; }
.score-cold { background: #ff444422; color: #ff4444; border: 1px solid #ff4444; }

/* Metric card */
.metric-card {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 800; color: #00ff88; }
.metric-label { font-size: 0.8rem; color: #888; margin-top: 2px; }

/* Section header */
.section-header {
    border-left: 4px solid #00ff88;
    padding-left: 12px;
    margin: 20px 0 10px 0;
    font-size: 1.1rem;
    font-weight: 700;
}

/* Indicator pill */
.pill {
    display: inline-block;
    background: #ffffff11;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    margin: 2px;
}
.pill-green { color: #00ff88; }
.pill-red { color: #ff4444; }
.pill-yellow { color: #ffaa00; }
.pill-blue { color: #5588ff; }

/* Scenario boxes */
.scenario-bull {
    background: #00ff8811;
    border: 1px solid #00ff8844;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.scenario-bear {
    background: #ff444411;
    border: 1px solid #ff444444;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.scenario-title { font-size: 0.9rem; color: #aaa; margin-bottom: 6px; }
.scenario-pct-bull { font-size: 2rem; font-weight: 800; color: #00ff88; }
.scenario-pct-bear { font-size: 2rem; font-weight: 800; color: #ff4444; }
.scenario-sub { font-size: 0.8rem; color: #777; margin-top: 4px; }

/* GitHub link — below Streamlit toolbar, top-right corner */
.gh-toolbar-link {
    position: fixed;
    top: 48px;
    right: 10px;
    z-index: 999999;
}
.gh-toolbar-link a {
    color: #555;
    text-decoration: none;
    display: flex;
    align-items: center;
    transition: color .15s;
}
.gh-toolbar-link a:hover { color: #fff; }

/* Confirmation notes */
.confirm-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.88rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    n_coins = st.selectbox(
        "Número de criptos a analizar",
        options=[50, 100, 150, 200],
        index=1,
        help="Más monedas = más tiempo de carga",
    )
    min_score = st.slider(
        "Bull Score mínimo a mostrar",
        min_value=0, max_value=100, value=0, step=5,
    )
    min_mcap = st.selectbox(
        "Market cap mínimo",
        options=["Sin filtro", "$1M+", "$10M+", "$100M+", "$1B+"],
        index=0,
    )
    sort_by = st.selectbox(
        "Ordenar por",
        options=["Bull Score ↓", "Market Cap ↓", "Upside % ↓", "R/R Ratio ↓", "RSI ↑"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 📡 Fuente de datos")
    st.markdown("**CoinGecko API** (gratuita)")
    st.markdown("Cache: 5 min (mercados) / 10 min (OHLCV)")
    st.markdown("---")
    refresh = st.button("🔄 Actualizar datos", use_container_width=True)
    if refresh:
        import os, glob
        for f in glob.glob(".crypto_cache/*.pkl"):
            os.remove(f)
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("⚠️ Esto no es consejo financiero. Solo análisis técnico automatizado.")


# ─────────────────────────────────────────
# Market cap filter mapping
# ─────────────────────────────────────────
MCAP_FILTERS = {
    "Sin filtro": 0,
    "$1M+": 1_000_000,
    "$10M+": 10_000_000,
    "$100M+": 100_000_000,
    "$1B+": 1_000_000_000,
}


# ─────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(limit: int) -> list:
    coins = fetch_markets(limit)
    results = []
    for coin in coins:
        try:
            a = analyze_coin_quick(coin)
            results.append(a)
        except Exception:
            pass
    return sorted(results, key=lambda x: x["bull_score"], reverse=True)


# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div class="gh-toolbar-link">
  <a href="https://github.com/tropikow/crypto-detector" target="_blank" title="Ver en GitHub">
    <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
               0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
               -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
               .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
               -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
               .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
               .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
               0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
    </svg>
  </a>
</div>

<div class="main-header">
  <div class="main-title">🚀 CRYPTO BULL DETECTOR</div>
  <div class="main-subtitle">Análisis técnico profesional · Detección de activos infravalorados pre-bull</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Load
# ─────────────────────────────────────────
with st.spinner("📡 Cargando datos de mercado..."):
    all_data = load_data(n_coins)

mcap_min = MCAP_FILTERS[min_mcap]
filtered = [
    c for c in all_data
    if c["bull_score"] >= min_score and c["market_cap"] >= mcap_min
]

hot = [c for c in filtered if c["bull_status"] == "hot"]
warm = [c for c in filtered if c["bull_status"] == "warm"]
neutral = [c for c in filtered if c["bull_status"] == "neutral"]
cold = [c for c in filtered if c["bull_status"] == "cold"]

# Sort
def sort_key(c):
    if "Bull Score" in sort_by:
        return -c["bull_score"]
    elif "Market Cap" in sort_by:
        return -c["market_cap"]
    elif "Upside" in sort_by:
        return -c["targets"]["moderate_pct"]
    elif "R/R" in sort_by:
        return -c["rr_ratio"]
    elif "RSI" in sort_by:
        return c["breakdown"]["rsi"]["value"]
    return -c["bull_score"]

filtered = sorted(filtered, key=sort_key)

# ─────────────────────────────────────────
# TABS: Scalping vs Bull Detector
# ─────────────────────────────────────────
tab_scalp, tab_bull = st.tabs(["⚡ SCALPING 15-30 min", "🚀 BULL DETECTOR"])

# ══════════════════════════════════════════
# TAB SCALPING
# ══════════════════════════════════════════
with tab_scalp:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a2a1a);
                border:1px solid #00ff8833; border-radius:12px; padding:16px; margin-bottom:16px;">
      <div style="font-size:1.3rem; font-weight:800; color:#00ff88;">⚡ MODO SCALPING — Entradas 15/30 minutos</div>
      <div style="color:#888; font-size:0.85rem; margin-top:4px;">
        Escaneo en tiempo real de las monedas más líquidas · Candles 30-min · RSI(7) · EMA5/10 · Stoch(5,3) · BB(10) · VWAP intradiario
      </div>
    </div>
    """, unsafe_allow_html=True)

    scalp_col1, scalp_col2 = st.columns([3, 1])
    with scalp_col1:
        n_scalp = st.slider("Monedas a escanear (por liquidez)", 10, 50, 20, 5,
                            key="scalp_n",
                            help="Solo monedas con Vol. 24h > $10M son elegibles para scalping")
    with scalp_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_scalp = st.button("🔍 Escanear ahora", use_container_width=True, key="run_scalp")

    if "scalp_results" not in st.session_state or run_scalp:
        with st.spinner(f"⚡ Escaneando {n_scalp} monedas para scalping de 15-30 min..."):
            @st.cache_data(ttl=180, show_spinner=False)
            def _run_scalp(coins_json: str, top_n: int):
                import json
                coins = json.loads(coins_json)
                return batch_scalp_scan(coins, top_n)

            import json as _json
            _coins_raw = [{"id": c["id"], "symbol": c["symbol"], "name": c["name"],
                           "current_price": c["current_price"],
                           "volume_24h": c["volume_24h"],
                           "market_cap": c["market_cap"]} for c in all_data]
            st.session_state["scalp_results"] = _run_scalp(
                _json.dumps(_coins_raw), n_scalp
            )

    scalp_data = st.session_state.get("scalp_results", [])

    if not scalp_data:
        st.warning("No se pudo obtener datos de scalping. Intenta de nuevo.")
    else:
        # ── KPIs ──
        fire_cnt = sum(1 for s in scalp_data if s["urgency"] == "FIRE")
        high_cnt = sum(1 for s in scalp_data if s["urgency"] == "HIGH")
        watch_cnt = sum(1 for s in scalp_data if s["urgency"] == "MEDIUM")

        kc1, kc2, kc3, kc4 = st.columns(4)
        for kc, val, label, clr in [
            (kc1, fire_cnt,  "🔥 ENTRADA AHORA",    "#00ff88"),
            (kc2, high_cnt,  "⚡ EN 15-30 min",      "#ffdd00"),
            (kc3, watch_cnt, "👀 Observar",          "#ffaa00"),
            (kc4, len(scalp_data), "📊 Escaneadas",  "#5588ff"),
        ]:
            kc.markdown(f"""
            <div style="background:#1a1a2e; border:1px solid {clr}44; border-radius:8px;
                        padding:12px; text-align:center; margin-bottom:10px;">
              <div style="font-size:1.8rem; font-weight:900; color:{clr}">{val}</div>
              <div style="font-size:0.75rem; color:#888">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top oportunidades ──
        top_scalp = [s for s in scalp_data if s["urgency"] in ("FIRE", "HIGH")][:6]
        if top_scalp:
            st.markdown("#### 🔥 Oportunidades Inmediatas")
            cols_s = st.columns(min(len(top_scalp), 3))
            for i, s in enumerate(top_scalp):
                col_s = cols_s[i % 3]
                sc_clr = s["scalp_color"]
                rsi_v = s.get("rsi7", 50) or 50
                with col_s:
                    st.markdown(f"""
                    <div style="background:#0d1b1d; border:2px solid {sc_clr}55;
                                border-top:3px solid {sc_clr};
                                border-radius:12px; padding:16px; margin-bottom:6px; position:relative;">
                      <div style="position:absolute; top:10px; right:12px;
                                  background:{sc_clr}22; color:{sc_clr};
                                  font-size:0.7rem; font-weight:800; padding:2px 8px;
                                  border-radius:20px; border:1px solid {sc_clr}55;">
                        {s['window']}
                      </div>

                      <div style="font-size:1.2rem; font-weight:900; color:#fff">{s['symbol']}</div>
                      <div style="font-size:0.75rem; color:#666; margin-bottom:8px">{s['name'][:20]}</div>

                      <div style="font-size:1.6rem; font-weight:700; color:{sc_clr}; margin-bottom:8px;">
                        {fmt_price(s['entry_price'])}
                      </div>

                      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:0.72rem; color:#888">Scalp Score</span>
                        <span style="font-size:0.88rem; font-weight:800; color:{sc_clr}">{s['scalp_score']}/100</span>
                      </div>
                      <div style="background:#333; border-radius:4px; height:6px; margin-bottom:10px;">
                        <div style="width:{s['scalp_score']}%; background:{sc_clr}; height:6px; border-radius:4px;"></div>
                      </div>

                      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-bottom:10px;">
                        <div style="background:#ffffff08; border-radius:5px; padding:6px; text-align:center;">
                          <div style="font-size:0.62rem; color:#888">RSI(7)</div>
                          <div style="font-size:0.95rem; font-weight:800;
                                      color:{'#ff6666' if rsi_v < 30 else '#00ff88' if rsi_v < 50 else '#ffaa00'}">
                            {rsi_v:.0f}
                          </div>
                        </div>
                        <div style="background:#ffffff08; border-radius:5px; padding:6px; text-align:center;">
                          <div style="font-size:0.62rem; color:#888">🎯 T15m</div>
                          <div style="font-size:0.88rem; font-weight:800; color:#00ff88">
                            {fmt_price(s['target_15m'])}
                          </div>
                        </div>
                        <div style="background:#ffffff08; border-radius:5px; padding:6px; text-align:center;">
                          <div style="font-size:0.62rem; color:#888">⛔ Stop</div>
                          <div style="font-size:0.88rem; font-weight:800; color:#ff4444">
                            {fmt_price(s['stop_loss'])}
                          </div>
                        </div>
                      </div>

                      <div style="font-size:0.8rem; font-weight:800;
                                  color:{sc_clr}; text-align:center;
                                  background:{sc_clr}11; border-radius:6px; padding:5px;">
                        {s['scalp_signal']}
                      </div>

                      <div style="margin-top:8px; font-size:0.72rem; color:#555; text-align:center;">
                        EMA: {s.get('ema_cross','—')} · VWAP: {s.get('vwap_signal','—')} ·
                        ROC: {s.get('momentum_roc',0):+.2f}%
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Botón de análisis detallado scalping
                    if st.button(f"📋 Detalle {s['symbol']}", key=f"scalp_detail_{s['coin_id']}",
                                 use_container_width=True):
                        st.session_state[f"show_scalp_{s['coin_id']}"] = True

                    if st.session_state.get(f"show_scalp_{s['coin_id']}"):
                        with st.expander(f"📊 Análisis Scalping · {s['symbol']}", expanded=True):
                            st.markdown(f"**Señal:** {s['scalp_signal']} · **Ventana:** {s['window']}")
                            st.markdown(f"**Entrada:** {fmt_price(s['entry_price'])} · "
                                        f"**T15m:** {fmt_price(s['target_15m'])} · "
                                        f"**T30m:** {fmt_price(s['target_30m'])} · "
                                        f"**Stop:** {fmt_price(s['stop_loss'])}")
                            st.markdown("**Razones del bot:**")
                            for r in s.get("reasons", []):
                                st.markdown(f"- {r}")
                            bb_sq = "✅ SÍ — BREAKOUT INMINENTE" if s.get("bb_squeeze") else "No"
                            st.markdown(f"""
                            | Indicador | Valor | Señal |
                            |---|---|---|
                            | RSI(7) | {s.get('rsi7','—')} | {'🟢 Sobrev.' if (s.get('rsi7') or 50) < 35 else '🟡 Neutro'} |
                            | EMA Cross | — | {s.get('ema_cross','—')} |
                            | Stoch %K/%D | {s.get('stoch_k','—')}/{s.get('stoch_d','—')} | {'🟢 Sobrev.' if (s.get('stoch_k') or 50) < 25 else '—'} |
                            | BB %B | {s.get('bb_pct','—')} | {'🔵 Zona baja' if (s.get('bb_pct') or 0.5) < 0.3 else '—'} |
                            | BB Squeeze | — | {bb_sq} |
                            | Momentum ROC | {s.get('momentum_roc',0):+.3f}% | {'🟢 Alcista' if (s.get('momentum_roc') or 0) > 0 else '🔴 Bajista'} |
                            | VWAP Intra | {fmt_price(s.get('vwap_intra', s['entry_price']))} | {s.get('vwap_signal','—')} |
                            """)

        # ── Tabla completa scalping ──
        st.markdown("---")
        st.markdown("#### 📊 Todas las monedas escaneadas")
        scalp_rows = []
        for s in scalp_data:
            scalp_rows.append({
                "Moneda": f"{s['symbol']} {s['name'][:12]}",
                "Precio": s["entry_price"],
                "Scalp Score": s["scalp_score"],
                "Señal": s["scalp_signal"],
                "Ventana": s["window"],
                "RSI(7)": s.get("rsi7") or 0,
                "EMA": s.get("ema_cross", "—"),
                "ROC%": s.get("momentum_roc", 0),
                "BB%B": s.get("bb_pct") or 0,
                "VWAP": s.get("vwap_signal", "—"),
                "T15m": s["target_15m"],
                "Stop": s["stop_loss"],
            })
        df_scalp = pd.DataFrame(scalp_rows)
        st.dataframe(
            df_scalp,
            use_container_width=True,
            height=min(60 + len(df_scalp) * 36, 500),
            column_config={
                "Moneda": st.column_config.TextColumn("🪙 Moneda", width="medium"),
                "Precio": st.column_config.NumberColumn("💲 Precio", format="$%.6g"),
                "Scalp Score": st.column_config.ProgressColumn(
                    "⚡ Scalp Score", min_value=0, max_value=100, format="%d"),
                "Señal": st.column_config.TextColumn("📶 Señal", width="medium"),
                "Ventana": st.column_config.TextColumn("⏱ Ventana"),
                "RSI(7)": st.column_config.NumberColumn("RSI(7)", format="%.1f"),
                "EMA": st.column_config.TextColumn("EMA Cross"),
                "ROC%": st.column_config.NumberColumn("ROC%", format="%.2f%%"),
                "BB%B": st.column_config.NumberColumn("BB %B", format="%.3f"),
                "VWAP": st.column_config.TextColumn("VWAP"),
                "T15m": st.column_config.NumberColumn("🎯 T15m", format="$%.6g"),
                "Stop": st.column_config.NumberColumn("⛔ Stop", format="$%.6g"),
            },
            hide_index=True,
        )

        st.markdown("""
        <div style="background:#1a1a1a; border:1px solid #333; border-radius:8px;
                    padding:12px; margin-top:12px; font-size:0.78rem; color:#666;">
          ⚠️ <b>Nota de scalping:</b> Los targets T15m y T30m se calculan usando ATR(7) sobre candles de 30 min.
          Siempre coloca stop loss. El scalping requiere atención constante al mercado.
          Esta señal tiene una ventana de validez de <b>15-30 minutos</b> — los datos se refrescan cada 3 minutos.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB BULL DETECTOR (contenido original)
# ══════════════════════════════════════════
with tab_bull:

# ─────────────────────────────────────────
# Summary cards
# ─────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#00ff88">{len(hot)}</div>
          <div class="metric-label">🔥 Alto Potencial</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#ffaa00">{len(warm)}</div>
          <div class="metric-label">⚡ Calentando</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#5588ff">{len(neutral)}</div>
          <div class="metric-label">⚠️ Neutral</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#ff4444">{len(cold)}</div>
          <div class="metric-label">❄️ Bajista</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#aaa">{len(filtered)}</div>
          <div class="metric-label">📊 Total Analizados</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # ─────────────────────────────────────────
    # Helper: score color  (defined before first use)
    # ─────────────────────────────────────────
    def score_class(s):
        if s >= 72: return "score-hot"
        elif s >= 55: return "score-warm"
        elif s >= 38: return "score-neutral"
        return "score-cold"

    def pct_color(v):
        if v > 0: return "#00ff88"
        elif v < 0: return "#ff4444"
        return "#aaa"

    def rsi_color(v):
        if v < 30: return "#ff6666"
        elif v < 50: return "#00ff88"
        elif v < 70: return "#ffaa00"
        return "#ff4444"


    # ─────────────────────────────────────────
    # Modal: análisis técnico detallado del pick
    # ─────────────────────────────────────────
    @st.dialog("🔬 Análisis Técnico Detallado", width="large")
    def show_pick_modal(coin: dict):
        with st.spinner(f"Cargando análisis completo de {coin['symbol']}..."):
            @st.cache_data(ttl=600, show_spinner=False)
            def _enrich(coin_id, data):
                return enrich_with_ohlcv(data.copy())
            full = _enrich(coin["id"], coin)

        sc = full["bull_score"]
        t = full["targets"]
        bd = full["breakdown"]
        ind = full.get("indicators_full", {})
        scenarios = full.get("scenarios", {})
        sc_color = "#00ff88" if sc >= 72 else ("#ffaa00" if sc >= 55 else "#ff4444")

        # ── Header ──
        h1, h2 = st.columns([3, 2])
        with h1:
            chg24 = full["price_change"]["24h"]
            st.markdown(f"""
            <div style="padding:4px 0 12px 0;">
              <span style="font-size:2rem; font-weight:900; color:#fff">{full['symbol']}</span>
              <span style="color:#666; margin-left:8px; font-size:0.9rem">{full['name']}</span><br>
              <span style="font-size:1.8rem; font-weight:700; color:{sc_color}">{fmt_price(full['current_price'])}</span>
              <span style="color:{'#00ff88' if chg24>=0 else '#ff4444'}; margin-left:10px; font-size:1rem">{chg24:+.2f}%</span><br>
              <span style="color:#555; font-size:0.82rem">ATH: {fmt_price(full['ath'])} · distancia: {full['ath_distance']:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        with h2:
            phase = full.get("phase", "")
            st.markdown(f"""
            <div style="text-align:center; background:#1a1a2e; border:2px solid {sc_color};
                        border-radius:10px; padding:14px;">
              <div style="font-size:0.72rem; color:#888; text-transform:uppercase">Bull Score</div>
              <div style="font-size:3rem; font-weight:900; color:{sc_color}; line-height:1">{sc}</div>
              <div style="font-size:0.82rem; color:{sc_color}; margin-top:2px">{full['bull_label']}</div>
              <div style="font-size:0.72rem; color:#666; margin-top:4px">{phase}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Por qué está aquí: razones del bot ──
        st.markdown("#### 🤖 ¿Por qué el bot seleccionó esta moneda?")
        reasons = []
        bd_rsi = bd.get("rsi", {}); bd_macd = bd.get("macd", {}); bd_trend = bd.get("trend", {})
        bd_bb = bd.get("bollinger", {}); bd_vol = bd.get("volume", {})
        bd_mom = bd.get("momentum", {}); bd_ath = bd.get("ath_dist", {})

        if bd_rsi.get("score", 0) >= 15:
            reasons.append(f"✅ **RSI {bd_rsi.get('value',0):.1f}** — {bd_rsi.get('note','')}")
        if bd_macd.get("score", 0) >= 15:
            reasons.append(f"✅ **MACD** — {bd_macd.get('note','')}")
        if bd_trend.get("score", 0) >= 15:
            reasons.append(f"✅ **Tendencia** — {bd_trend.get('note','')}")
        if bd_bb.get("score", 0) >= 10:
            reasons.append(f"✅ **Bollinger** — {bd_bb.get('note','')}")
        if bd_vol.get("score", 0) >= 6:
            reasons.append(f"✅ **Volumen** — {bd_vol.get('note','')}")
        if bd_mom.get("score", 0) >= 15:
            reasons.append(f"✅ **Momentum** — {bd_mom.get('note','')}")
        if bd_ath.get("score", 0) >= 15:
            reasons.append(f"✅ **Valuación** — {bd_ath.get('note','')}")
        if not reasons:
            reasons.append("⚠️ Señales moderadas — confluencia de múltiples indicadores en zona de posible rebote")

        for r in reasons:
            st.markdown(r)

        st.markdown("---")

        # ── Score breakdown visual ──
        st.markdown("#### 🧮 Desglose del Bull Score")
        comp_labels = {"rsi": "RSI (14)", "macd": "MACD", "trend": "Tendencia / EMAs",
                       "bollinger": "Bollinger Bands", "volume": "Volumen",
                       "momentum": "Momentum precio", "ath_dist": "Distancia al ATH", "marketcap": "Market Cap"}
        for key, data in bd.items():
            s = data.get("score", 0); mx = data.get("max", 25)
            note = data.get("note", ""); label = comp_labels.get(key, key.upper())
            pct = (s / mx * 100) if mx else 0
            bar_col = "#00ff88" if pct >= 70 else ("#ffaa00" if pct >= 40 else "#ff4444")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
              <div style="min-width:150px; font-size:0.82rem; color:#ccc">{label}</div>
              <div style="flex:1; background:#222; border-radius:4px; height:7px;">
                <div style="width:{pct:.0f}%; background:{bar_col}; height:7px; border-radius:4px;"></div>
              </div>
              <div style="min-width:50px; text-align:right; font-size:0.8rem; color:{bar_col}; font-weight:700">{s}/{mx}</div>
              <div style="min-width:180px; font-size:0.75rem; color:#777">{note}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Indicadores numéricos ──
        st.markdown("#### 📊 Indicadores Técnicos")
        if ind:
            ic1, ic2, ic3, ic4, ic5 = st.columns(5)
            def _ind_box(col, label, val, color, interp=""):
                col.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #333; border-radius:7px;
                            padding:9px; text-align:center; margin-bottom:6px;">
                  <div style="font-size:0.68rem; color:#888">{label}</div>
                  <div style="font-size:1rem; font-weight:800; color:{color}">{val}</div>
                  <div style="font-size:0.65rem; color:{color}; opacity:0.7">{interp}</div>
                </div>""", unsafe_allow_html=True)

            rsi_v2 = ind.get("rsi", 50)
            _ind_box(ic1, "RSI (14)", f"{rsi_v2:.1f}", rsi_color(rsi_v2),
                     "SOBREVENDIDO" if rsi_v2 < 30 else ("NEUTRO" if rsi_v2 < 70 else "SOBRECOMPRADO"))
            hl2 = ind.get("histogram", 0)
            _ind_box(ic2, "MACD Hist.", f"{hl2:.4f}", "#00ff88" if hl2 >= 0 else "#ff4444",
                     "ALCISTA" if hl2 >= 0 else "BAJISTA")
            sk = ind.get("stoch_k", 50)
            _ind_box(ic3, "Stoch %K", f"{sk:.1f}", "#ff6666" if sk < 20 else ("#00ff88" if sk < 80 else "#ff4444"),
                     "SOBREVENDIDO" if sk < 20 else ("NEUTRO" if sk < 80 else "SOBRECOMPRADO"))
            bw = ind.get("bb_width", 0)
            _ind_box(ic4, "BB Width", f"{bw:.4f}", "#9966ff", "SQUEEZE" if bw < 0.05 else "NORMAL")
            bp = ind.get("bb_pct", 0.5)
            _ind_box(ic5, "BB %B", f"{bp:.3f}", "#5588ff",
                     "ZONA BAJA" if bp < 0.3 else ("MEDIA" if bp < 0.7 else "ZONA ALTA"))

        st.markdown("---")

        # ── Gráfico de precio ──
        ohlcv = full.get("ohlcv", [])
        if ohlcv:
            st.markdown("#### 📈 Gráfico de Precio (30 días · 4H)")
            df_c = pd.DataFrame(ohlcv)
            close_c = df_c["close"]; high_c = df_c["high"]; low_c = df_c["low"]
            e20_c = calc_ema(close_c, 20); e50_c = calc_ema(close_c, 50)
            rsi_c = calc_rsi(close_c)
            ml_c, ms_c, mh_c = calc_macd(close_c)
            bb_u_c, _, bb_l_c, _, _ = calc_bb(close_c)

            fig_m = make_subplots(rows=3, cols=1, row_heights=[0.55, 0.22, 0.23],
                                  shared_xaxes=True, vertical_spacing=0.04,
                                  subplot_titles=["Precio + EMAs + Bollinger", "RSI (14)", "MACD"])
            fig_m.add_trace(go.Candlestick(x=df_c["ts"], open=df_c["open"], high=high_c,
                                           low=low_c, close=close_c, name="OHLC",
                                           increasing_line_color="#00ff88",
                                           decreasing_line_color="#ff4444"), row=1, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=bb_u_c, name="BB Up",
                                       line=dict(color="rgba(85,136,255,0.35)", width=1),
                                       showlegend=False), row=1, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=bb_l_c, name="BB Low",
                                       line=dict(color="rgba(85,136,255,0.35)", width=1),
                                       fill="tonexty", fillcolor="rgba(85,136,255,0.05)",
                                       showlegend=False), row=1, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=e20_c, name="EMA20",
                                       line=dict(color="#ffaa00", width=1.5)), row=1, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=e50_c, name="EMA50",
                                       line=dict(color="#ff6688", width=1.5)), row=1, col=1)
            if t.get("moderate_price"):
                fig_m.add_hline(y=t["moderate_price"], line_dash="dash", line_color="#00ff88",
                                annotation_text=f"🎯 {fmt_price(t['moderate_price'])}", row=1, col=1)
            if t.get("stop_loss"):
                fig_m.add_hline(y=t["stop_loss"], line_dash="dot", line_color="#ff4444",
                                annotation_text=f"⛔ {fmt_price(t['stop_loss'])}", row=1, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=rsi_c, name="RSI",
                                       line=dict(color="#9966ff", width=2)), row=2, col=1)
            fig_m.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.5)", row=2, col=1)
            fig_m.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=2, col=1)
            colors_h = ["#00ff88" if v >= 0 else "#ff4444" for v in mh_c]
            fig_m.add_trace(go.Bar(x=df_c["ts"], y=mh_c, marker_color=colors_h,
                                   opacity=0.8, name="Hist"), row=3, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=ml_c, name="MACD",
                                       line=dict(color="#5588ff", width=1.5)), row=3, col=1)
            fig_m.add_trace(go.Scatter(x=df_c["ts"], y=ms_c, name="Señal",
                                       line=dict(color="#ff6688", width=1.5)), row=3, col=1)
            fig_m.update_layout(height=520, template="plotly_dark",
                                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                xaxis_rangeslider_visible=False, margin=dict(l=10, r=80, t=30, b=10),
                                font=dict(color="#aaa"), legend=dict(orientation="h", y=1.05))
            fig_m.update_xaxes(showgrid=False); fig_m.update_yaxes(gridcolor="#1e1e1e")
            st.plotly_chart(fig_m, use_container_width=True)

        # ── Targets y Escenarios ──
        st.markdown("---")
        st.markdown("#### 🎯 Precios Objetivo y Escenarios")
        ta1, ta2, ta3 = st.columns(3)
        for col_t, lbl, pct_k, price_k, clr in [
            (ta1, "🟢 Conservador (Fib 0.618)", "conservative_pct", "conservative_price", "#4caf50"),
            (ta2, "🔵 Moderado (Fib 1.0)", "moderate_pct", "moderate_price", "#00ff88"),
            (ta3, "🔴 Agresivo (Fib 1.618)", "aggressive_pct", "aggressive_price", "#ff6688"),
        ]:
            with col_t:
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid {clr}44; border-radius:8px;
                            padding:12px; text-align:center;">
                  <div style="font-size:0.75rem; color:#888; margin-bottom:4px">{lbl}</div>
                  <div style="font-size:1.5rem; font-weight:800; color:{clr}">{fmt_price(t.get(price_k, 0))}</div>
                  <div style="font-size:1.1rem; font-weight:700; color:{clr}">+{t.get(pct_k, 0):.1f}%</div>
                </div>""", unsafe_allow_html=True)

        sb1, sb2 = st.columns(2)
        bull_p = scenarios.get("bull_probability", sc)
        err_p = scenarios.get("error_probability", 100 - sc)
        with sb1:
            st.markdown(f"""
            <div style="background:#00ff8811; border:1px solid #00ff8833; border-radius:8px;
                        padding:14px; text-align:center; margin-top:10px;">
              <div style="font-size:0.8rem; color:#aaa">🚀 ESCENARIO BULL</div>
              <div style="font-size:1.8rem; font-weight:900; color:#00ff88">+{t.get('moderate_pct',0):.1f}%</div>
              <div style="font-size:0.82rem; color:#00ff88">Probabilidad: {bull_p}%</div>
              <div style="font-size:0.75rem; color:#666; margin-top:4px">Target: {fmt_price(t.get('moderate_price',0))}</div>
            </div>""", unsafe_allow_html=True)
        with sb2:
            stop = t.get("stop_loss", full["current_price"] * 0.9)
            st.markdown(f"""
            <div style="background:#ff444411; border:1px solid #ff444433; border-radius:8px;
                        padding:14px; text-align:center; margin-top:10px;">
              <div style="font-size:0.8rem; color:#aaa">❌ ESCENARIO ERROR</div>
              <div style="font-size:1.8rem; font-weight:900; color:#ff4444">{t.get('downside_pct',0):.1f}%</div>
              <div style="font-size:0.82rem; color:#ff4444">Probabilidad: {err_p}%</div>
              <div style="font-size:0.75rem; color:#666; margin-top:4px">Stop: {fmt_price(stop)}</div>
            </div>""", unsafe_allow_html=True)

        # ── Whale / Smart Money Analysis ──
        st.markdown("---")
        whale = full.get("whale", {})
        w_score  = whale.get("whale_score", 50)
        w_signal = whale.get("whale_signal", "—")
        w_color  = whale.get("whale_color", "#888")
        w_notes  = whale.get("notes", [])
        w_cmf    = whale.get("cmf", 0.0)
        w_obv    = whale.get("obv_trend", "neutral")
        w_vwap   = whale.get("vwap_position", "unknown")
        w_spike  = whale.get("vol_spike", False)
        w_ratio  = whale.get("vol_spike_ratio", 1.0)
        w_div    = whale.get("obv_divergence", False)
        w_acc    = whale.get("quiet_accumulation", False)
        w_dist   = whale.get("distribution_warning", False)

        st.markdown("#### 🐳 Actividad de Ballenas / Smart Money")

        # Score banner
        pct_w = w_score
        st.markdown(f"""
        <div style="background:#0a0a1a; border:2px solid {w_color}55; border-radius:10px; padding:14px; margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:1.1rem; font-weight:800; color:{w_color}">{w_signal}</span>
            <span style="font-size:1.4rem; font-weight:900; color:{w_color}">{w_score}/100</span>
          </div>
          <div style="background:#222; border-radius:5px; height:8px;">
            <div style="width:{pct_w}%; background:linear-gradient(90deg, {w_color}66, {w_color}); height:8px; border-radius:5px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Indicators grid
        wc1, wc2, wc3, wc4 = st.columns(4)
        def _whale_box(col, label, val_str, color, sublabel=""):
            col.markdown(f"""
            <div style="background:#1a1a2e; border:1px solid {color}33; border-radius:7px;
                        padding:10px; text-align:center; margin-bottom:8px;">
              <div style="font-size:0.68rem; color:#888">{label}</div>
              <div style="font-size:1rem; font-weight:800; color:{color}">{val_str}</div>
              <div style="font-size:0.65rem; color:{color}; opacity:0.8">{sublabel}</div>
            </div>""", unsafe_allow_html=True)

        obv_col   = "#00ff88" if w_obv == "rising" else ("#ff4444" if w_obv == "falling" else "#888")
        obv_lbl   = "SUBIENDO" if w_obv == "rising" else ("BAJANDO" if w_obv == "falling" else "LATERAL")
        cmf_col   = "#00ff88" if w_cmf > 0.05 else ("#ff4444" if w_cmf < -0.05 else "#888")
        cmf_lbl   = "COMPRA" if w_cmf > 0.05 else ("VENTA" if w_cmf < -0.05 else "NEUTRAL")
        vwap_col  = "#00ff88" if "below" in w_vwap else "#ffaa00"
        vwap_lbl  = {"far_below": "ZONA VALOR", "below": "BAJO VWAP", "above": "SOBRE VWAP"}.get(w_vwap, "—")
        spike_col = "#00ff88" if w_spike else "#888"

        _whale_box(wc1, "OBV Trend", obv_lbl, obv_col, "acumulación" if w_obv == "rising" else "distribución")
        _whale_box(wc2, "CMF (20)", f"{w_cmf:+.3f}", cmf_col, cmf_lbl)
        _whale_box(wc3, "VWAP Pos.", vwap_lbl, vwap_col, "institucional" if "below" in w_vwap else "—")
        _whale_box(wc4, "Vol Spike", f"{w_ratio:.1f}×", spike_col, "ACTIVO" if w_spike else "normal")

        # Alert badges
        badges_html = ""
        if w_div:
            badges_html += f'<span style="background:#00ff8822; color:#00ff88; border:1px solid #00ff8855; border-radius:20px; padding:3px 10px; font-size:0.78rem; font-weight:700; margin-right:6px;">🚨 DIVERGENCIA OBV ALCISTA</span>'
        if w_acc:
            badges_html += f'<span style="background:#5588ff22; color:#5588ff; border:1px solid #5588ff55; border-radius:20px; padding:3px 10px; font-size:0.78rem; font-weight:700; margin-right:6px;">🤫 ACUMULACIÓN SILENCIOSA</span>'
        if w_dist:
            badges_html += f'<span style="background:#ff444422; color:#ff4444; border:1px solid #ff444455; border-radius:20px; padding:3px 10px; font-size:0.78rem; font-weight:700; margin-right:6px;">⛔ ALERTA DISTRIBUCIÓN</span>'
        if badges_html:
            st.markdown(f"<div style='margin-bottom:10px'>{badges_html}</div>", unsafe_allow_html=True)

        # OBV chart if available
        obv_series = whale.get("obv_series", [])
        cmf_series = whale.get("cmf_series", [])
        if obv_series and len(obv_series) >= 24:
            h_idx = list(range(len(obv_series)))
            fig_w = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True,
                                  vertical_spacing=0.05, subplot_titles=["OBV (On-Balance Volume)", "CMF (Chaikin Money Flow)"])
            obv_color_line = "#00ff88" if w_obv == "rising" else "#ff4444"
            obv_fill_color = "rgba(0,255,136,0.07)" if w_obv == "rising" else "rgba(255,68,68,0.07)"
            fig_w.add_trace(go.Scatter(x=h_idx, y=obv_series, name="OBV",
                                       line=dict(color=obv_color_line, width=2),
                                       fill="tozeroy", fillcolor=obv_fill_color),
                            row=1, col=1)
            if cmf_series:
                cmf_colors = ["#00ff88" if v >= 0 else "#ff4444" for v in cmf_series]
                fig_w.add_trace(go.Bar(x=h_idx, y=cmf_series, marker_color=cmf_colors,
                                       opacity=0.8, name="CMF"), row=2, col=1)
                fig_w.add_hline(y=0, line_color="#555", line_dash="dash", row=2, col=1)
                fig_w.add_hline(y=0.15, line_color="rgba(0,255,136,0.3)", line_dash="dot", row=2, col=1)
                fig_w.add_hline(y=-0.15, line_color="rgba(255,68,68,0.3)", line_dash="dot", row=2, col=1)
            fig_w.update_layout(height=280, template="plotly_dark",
                                 paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                 margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                                 font=dict(color="#aaa", size=10))
            fig_w.update_xaxes(showgrid=False, showticklabels=False)
            fig_w.update_yaxes(gridcolor="#1a1a1a")
            st.plotly_chart(fig_w, use_container_width=True)

        # Whale notes
        if w_notes:
            with st.expander("📋 Notas detalladas del análisis de ballenas", expanded=False):
                for n in w_notes:
                    st.markdown(n)

        # ── Confirmaciones ──
        st.markdown("---")
        st.markdown("#### 🕯️ Señales y Velas de Confirmación")
        confirms = full.get("confirmations", [])
        if confirms:
            box = "<br>".join(confirms)
            st.markdown(f'<div class="confirm-box">{box}</div>', unsafe_allow_html=True)

        # ── R/R y acción sugerida ──
        rr = full.get("rr_ratio", 0)
        rec = ("ACUMULAR con stops ajustados" if sc >= 72 else
               "OBSERVAR y esperar confirmación" if sc >= 55 else
               "WATCHLIST sin acción inmediata")
        st.markdown(f"""
        <br>
        <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:14px;
                    display:flex; justify-content:space-around; align-items:center; text-align:center;">
          <div>
            <div style="font-size:0.75rem; color:#888">R/R Ratio</div>
            <div style="font-size:1.6rem; font-weight:900; color:{'#00ff88' if rr>=2 else '#ffaa00'}">{rr:.2f}×</div>
          </div>
          <div style="width:1px; height:40px; background:#333;"></div>
          <div>
            <div style="font-size:0.75rem; color:#888">Acción Sugerida</div>
            <div style="font-size:1rem; font-weight:700; color:#ffaa00">{rec}</div>
          </div>
          <div style="width:1px; height:40px; background:#333;"></div>
          <div>
            <div style="font-size:0.75rem; color:#888">Vol. en Bull (est.)</div>
            <div style="font-size:1rem; font-weight:700; color:#5588ff">{fmt_large(full.get('bull_volume_est', full['volume_24h']*2.5))}</div>
          </div>
        </div>""", unsafe_allow_html=True)


    # ─────────────────────────────────────────
    # TOP PICKS — cards for highest potential
    # ─────────────────────────────────────────
    top_picks = [c for c in filtered if c["bull_status"] in ("hot", "warm")][:6]

    if top_picks:
        st.markdown('<div class="section-header">⭐ TOP PICKS — Mayor Potencial de Bull</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#888; font-size:0.85rem; margin-bottom:12px;'>Las criptomonedas con mayor probabilidad de movimiento alcista según el análisis técnico. <b style='color:#00ff88'>Click en cualquier tarjeta</b> para ver el análisis completo del bot.</p>", unsafe_allow_html=True)

        cols_tp = st.columns(min(len(top_picks), 3))
        for i, coin in enumerate(top_picks[:6]):
            col = cols_tp[i % 3]
            t_c = coin["targets"]
            bd_c = coin["breakdown"]
            rsi_v = bd_c["rsi"]["value"]
            sc = coin["bull_score"]
            up_m = t_c["moderate_pct"]
            down = t_c["downside_pct"]
            chg24 = coin["price_change"]["24h"]
            chg7 = coin["price_change"]["7d"]
            border_color = "#00ff88" if coin["bull_status"] == "hot" else "#ffaa00"
            sc_color = "#00ff88" if sc >= 72 else "#ffaa00"
            rsi_col = rsi_color(rsi_v) if rsi_v else "#aaa"

            with col:
                whale = coin.get("whale", {})
                w_signal = whale.get("whale_signal", "")
                w_color  = whale.get("whale_color", "#888")
                w_score  = whale.get("whale_score", 50)
                show_whale_badge = w_score >= 62
                st.markdown(f"""
                <div style="background:#1a1a2e; border:2px solid {border_color}44;
                            border-top: 3px solid {border_color};
                            border-radius:12px; padding:18px; margin-bottom:6px;
                            position:relative;">

                  <!-- Rank badge -->
                  <div style="position:absolute; top:12px; right:14px;
                              background:{border_color}22; color:{border_color};
                              font-size:0.72rem; font-weight:700; padding:2px 8px;
                              border-radius:20px; border:1px solid {border_color}55;">
                    #{i+1} PICK
                  </div>

                  <!-- Whale badge -->
                  {'<div style="position:absolute; top:38px; right:14px; background:' + w_color + '22; color:' + w_color + '; font-size:0.65rem; font-weight:700; padding:2px 7px; border-radius:20px; border:1px solid ' + w_color + '44;">' + w_signal[:22] + '</div>' if show_whale_badge else ''}

                  <!-- Coin name -->
                  <div style="font-size:1.3rem; font-weight:800; color:#fff; margin-bottom:2px;">
                    {coin['symbol']}
                    <span style="font-size:0.78rem; color:#666; font-weight:400; margin-left:6px;">{coin['name'][:16]}</span>
                  </div>

                  <!-- Price -->
                  <div style="font-size:1.5rem; font-weight:700; color:{border_color}; margin-bottom:10px;">
                    {fmt_price(coin['current_price'])}
                    <span style="font-size:0.85rem; color:{'#00ff88' if chg24 >= 0 else '#ff4444'}; margin-left:8px;">
                      {chg24:+.2f}%
                    </span>
                  </div>

                  <!-- Bull Score bar -->
                  <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                      <span style="font-size:0.75rem; color:#888;">Bull Score</span>
                      <span style="font-size:0.88rem; font-weight:800; color:{sc_color};">{sc}/100</span>
                    </div>
                    <div style="background:#333; border-radius:6px; height:8px;">
                      <div style="width:{sc}%; background:linear-gradient(90deg, {border_color}88, {border_color});
                                  height:8px; border-radius:6px;"></div>
                    </div>
                  </div>

                  <!-- Key stats grid -->
                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px;">
                    <div style="background:#ffffff08; border-radius:6px; padding:8px; text-align:center;">
                      <div style="font-size:0.68rem; color:#888;">🎯 UPSIDE EST.</div>
                      <div style="font-size:1.1rem; font-weight:800; color:#00ff88;">+{up_m:.0f}%</div>
                    </div>
                    <div style="background:#ffffff08; border-radius:6px; padding:8px; text-align:center;">
                      <div style="font-size:0.68rem; color:#888;">🛡️ STOP LOSS</div>
                      <div style="font-size:1.1rem; font-weight:800; color:#ff4444;">{down:.0f}%</div>
                    </div>
                    <div style="background:#ffffff08; border-radius:6px; padding:8px; text-align:center;">
                      <div style="font-size:0.68rem; color:#888;">📊 RSI</div>
                      <div style="font-size:1.1rem; font-weight:800; color:{rsi_col};">{rsi_v:.1f}</div>
                    </div>
                    <div style="background:#ffffff08; border-radius:6px; padding:8px; text-align:center;">
                      <div style="font-size:0.68rem; color:#888;">⚖️ R/R</div>
                      <div style="font-size:1.1rem; font-weight:800; color:{'#00ff88' if coin['rr_ratio'] >= 2 else '#ffaa00'};">
                        {coin['rr_ratio']:.1f}×
                      </div>
                    </div>
                  </div>

                  <!-- Signal + 7d change -->
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="background:{border_color}22; color:{border_color};
                                 font-size:0.78rem; font-weight:700; padding:3px 10px;
                                 border-radius:20px; border:1px solid {border_color}44;">
                      {coin['signal']}
                    </span>
                    <span style="font-size:0.8rem; color:#888;">
                      7d: <span style="color:{'#00ff88' if chg7 >= 0 else '#ff4444'}; font-weight:600;">{chg7:+.1f}%</span>
                    </span>
                  </div>

                  <!-- ATH distance -->
                  <div style="margin-top:8px; font-size:0.75rem; color:#666; text-align:center;">
                    {coin['ath_distance']:.1f}% vs ATH · {fmt_large(coin['market_cap'])} mcap
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Botón que abre el modal ──
                if st.button(
                    f"🔬 Ver Análisis · {coin['symbol']}",
                    key=f"pick_modal_{coin['id']}",
                    use_container_width=True,
                ):
                    show_pick_modal(coin)

        st.markdown("<br>", unsafe_allow_html=True)


    # ─────────────────────────────────────────
    # Main table
    # ─────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Dashboard de Criptomonedas</div>', unsafe_allow_html=True)

    if not filtered:
        st.warning("No hay monedas que cumplan los filtros actuales.")
    else:
        # ── Build DataFrame for native st.dataframe ──
        rows = []
        for c in filtered:
            t = c["targets"]
            bd = c["breakdown"]
            sc = c["bull_score"]
            rsi_v = bd["rsi"]["value"]
            rows.append({
                "Moneda": f"{c['symbol']} {c['name'][:14]}",
                "Precio": c["current_price"],
                "Bull Score": sc,
                "Señal": c["bull_label"],
                "RSI": round(rsi_v, 1),
                "24h %": round(c["price_change"]["24h"], 2),
                "7d %": round(c["price_change"]["7d"], 2),
                "Upside Est.": round(t["moderate_pct"], 1),
                "Riesgo %": round(t["downside_pct"], 1),
                "R/R": round(c["rr_ratio"], 2),
                "vs ATH %": round(c["ath_distance"], 1),
                "Mcap": c["market_cap"],
            })

        df_table = pd.DataFrame(rows)

        st.dataframe(
            df_table,
            use_container_width=True,
            height=min(50 + len(df_table) * 36, 600),
            column_config={
                "Moneda": st.column_config.TextColumn("🪙 Moneda", width="medium"),
                "Precio": st.column_config.NumberColumn(
                    "💲 Precio", format="$%.6g", width="small"
                ),
                "Bull Score": st.column_config.ProgressColumn(
                    "🔥 Bull Score", min_value=0, max_value=100, format="%d",
                    width="medium",
                ),
                "Señal": st.column_config.TextColumn("📶 Señal", width="medium"),
                "RSI": st.column_config.NumberColumn("📊 RSI", format="%.1f", width="small"),
                "24h %": st.column_config.NumberColumn("⏱ 24h %", format="%.2f%%", width="small"),
                "7d %": st.column_config.NumberColumn("📅 7d %", format="%.2f%%", width="small"),
                "Upside Est.": st.column_config.NumberColumn(
                    "🎯 Upside", format="+%.1f%%", width="small"
                ),
                "Riesgo %": st.column_config.NumberColumn(
                    "🛡 Riesgo", format="%.1f%%", width="small"
                ),
                "R/R": st.column_config.NumberColumn("⚖️ R/R", format="%.2f×", width="small"),
                "vs ATH %": st.column_config.NumberColumn("📉 vs ATH", format="%.1f%%", width="small"),
                "Mcap": st.column_config.NumberColumn(
                    "💰 Mcap", format="$%.3e", width="small"
                ),
            },
            hide_index=True,
        )


    # ─────────────────────────────────────────
    # Coin detail analysis
    # ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Análisis Detallado</div>', unsafe_allow_html=True)

    coin_names = [f"{c['symbol']} — {c['name']}" for c in filtered]
    selected_label = st.selectbox(
        "Selecciona una criptomoneda para análisis completo:",
        options=coin_names,
        index=0 if coin_names else None,
    )

    if selected_label:
        sel_idx = coin_names.index(selected_label)
        selected_coin = filtered[sel_idx]

        with st.spinner(f"📡 Cargando análisis completo de {selected_coin['symbol']}..."):
            @st.cache_data(ttl=600, show_spinner=False)
            def get_full_analysis(coin_id: str, data: dict) -> dict:
                return enrich_with_ohlcv(data.copy())

            full = get_full_analysis(selected_coin["id"], selected_coin)

        t = full["targets"]
        bd = full["breakdown"]
        sc = full["bull_score"]
        ind = full.get("indicators_full", {})
        scenarios = full.get("scenarios", {})

        # ── Coin header ──
        col_img, col_info, col_score = st.columns([1, 4, 2])
        with col_img:
            if full.get("image"):
                st.image(full["image"], width=80)
        with col_info:
            price_24h_color = pct_color(full["price_change"]["24h"])
            st.markdown(f"""
            <div style="padding-top:8px">
              <span style="font-size:1.8rem; font-weight:800; color:#fff">{full['symbol']}</span>
              <span style="color:#888; margin-left:8px">{full['name']}</span><br>
              <span style="font-size:1.5rem; font-weight:700; color:#00ff88">{fmt_price(full['current_price'])}</span>
              <span style="color:{price_24h_color}; margin-left:10px; font-size:1rem">
                {full['price_change']['24h']:+.2f}% (24h)
              </span><br>
              <span style="color:#888; font-size:0.85rem">
                Rank #{full['market_cap_rank']} · Mcap: {fmt_large(full['market_cap'])} ·
                ATH: {fmt_price(full['ath'])} ({full['ath_distance']:.1f}%)
              </span>
            </div>
            """, unsafe_allow_html=True)
        with col_score:
            sc_color = "#00ff88" if sc >= 72 else ("#ffaa00" if sc >= 55 else ("#5588ff" if sc >= 38 else "#ff4444"))
            phase = full.get("phase", "")
            st.markdown(f"""
            <div style="text-align:center; padding:12px; background:#1a1a2e; border-radius:10px; border:2px solid {sc_color};">
              <div style="font-size:0.8rem; color:#888">BULL SCORE</div>
              <div style="font-size:3rem; font-weight:900; color:{sc_color}; line-height:1.1">{sc}</div>
              <div style="font-size:0.85rem; color:{sc_color}; margin-top:2px">{full['bull_label']}</div>
              <div style="font-size:0.75rem; color:#888; margin-top:4px">{phase}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ──
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Gráfico", "🎯 Targets & Escenarios", "📊 Indicadores", "🕯️ Confirmaciones"])

        # ── TAB 1: Chart ──
        with tab1:
            ohlcv = full.get("ohlcv", [])
            if ohlcv:
                df_chart = pd.DataFrame(ohlcv)
                close = df_chart["close"]
                high_s = df_chart["high"]
                low_s = df_chart["low"]

                e20_s = calc_ema(close, 20)
                e50_s = calc_ema(close, 50)
                rsi_s = calc_rsi(close)
                macd_l, macd_sig, macd_hist = calc_macd(close)
                bb_u, bb_m, bb_l, _, _ = calc_bb(close)

                fig = make_subplots(
                    rows=3, cols=1,
                    row_heights=[0.55, 0.22, 0.23],
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=["Precio + EMAs + Bollinger", "RSI (14)", "MACD"],
                )

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df_chart["ts"],
                    open=df_chart["open"],
                    high=high_s,
                    low=low_s,
                    close=close,
                    name="OHLC",
                    increasing_line_color="#00ff88",
                    decreasing_line_color="#ff4444",
                ), row=1, col=1)

                # Bollinger Bands
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=bb_u, name="BB Up",
                                         line=dict(color="rgba(85,136,255,0.35)", width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=bb_l, name="BB Low",
                                         line=dict(color="rgba(85,136,255,0.35)", width=1),
                                         fill="tonexty", fillcolor="rgba(85,136,255,0.05)",
                                         showlegend=False), row=1, col=1)

                # EMAs
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=e20_s, name="EMA20",
                                         line=dict(color="#ffaa00", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=e50_s, name="EMA50",
                                         line=dict(color="#ff6688", width=1.5)), row=1, col=1)

                # Target lines
                if t.get("moderate_price"):
                    fig.add_hline(y=t["moderate_price"], line_dash="dash",
                                  line_color="#00ff88", line_width=1.5,
                                  annotation_text=f"Target: {fmt_price(t['moderate_price'])}",
                                  annotation_position="right", row=1, col=1)
                if t.get("stop_loss"):
                    fig.add_hline(y=t["stop_loss"], line_dash="dot",
                                  line_color="#ff4444", line_width=1.2,
                                  annotation_text=f"Stop: {fmt_price(t['stop_loss'])}",
                                  annotation_position="right", row=1, col=1)

                # RSI
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=rsi_s, name="RSI",
                                         line=dict(color="#9966ff", width=2)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.5)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=2, col=1)
                fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)",
                              line_width=0, row=2, col=1)

                # MACD
                colors_hist = ["#00ff88" if v >= 0 else "#ff4444" for v in macd_hist]
                fig.add_trace(go.Bar(x=df_chart["ts"], y=macd_hist, name="Histograma",
                                     marker_color=colors_hist, opacity=0.8), row=3, col=1)
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=macd_l, name="MACD",
                                         line=dict(color="#5588ff", width=1.5)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=macd_sig, name="Señal",
                                         line=dict(color="#ff6688", width=1.5)), row=3, col=1)

                fig.update_layout(
                    height=650,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=80, t=30, b=10),
                    font=dict(color="#aaa"),
                )
                fig.update_xaxes(showgrid=False, gridcolor="#222")
                fig.update_yaxes(showgrid=True, gridcolor="#1e1e1e")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos OHLCV disponibles. Intenta refrescar.")

        # ── TAB 2: Targets & Scenarios ──
        with tab2:
            st.markdown('<div class="section-header">🎯 Estimación de Precio</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            for col, label, pct, price_key, color in [
                (c1, "🟢 Conservador", t.get("conservative_pct", 0), "conservative_price", "#4caf50"),
                (c2, "🔵 Moderado", t.get("moderate_pct", 0), "moderate_price", "#00ff88"),
                (c3, "🔴 Agresivo", t.get("aggressive_pct", 0), "aggressive_price", "#ff6688"),
            ]:
                with col:
                    est_price = t.get(price_key, full["current_price"])
                    st.markdown(f"""
                    <div style="background:#1a1a2e; border:1px solid {color}44; border-radius:10px;
                                padding:16px; text-align:center; margin-bottom:8px;">
                      <div style="font-size:0.85rem; color:#888">{label}</div>
                      <div style="font-size:1.6rem; font-weight:800; color:{color}">{fmt_price(est_price)}</div>
                      <div style="font-size:1.2rem; font-weight:700; color:{color}">+{pct:.1f}%</div>
                      <div style="font-size:0.75rem; color:#666; margin-top:4px">
                        Fib {'0.618' if 'conserv' in price_key else ('1.000' if 'moderate' in price_key else '1.618')} extension
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            if t.get("to_ath_pct"):
                st.info(f"🏆 **Recuperación a ATH** ({fmt_price(full['ath'])}): potencial de **+{t['to_ath_pct']:.1f}%** desde precio actual")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📦 Estimación de Volumen</div>', unsafe_allow_html=True)

            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:12px; text-align:center;">
                  <div style="font-size:0.78rem; color:#888">Vol. 24h Actual</div>
                  <div style="font-size:1.2rem; font-weight:700; color:#fff">{fmt_large(full['volume_24h'])}</div>
                </div>""", unsafe_allow_html=True)
            with vc2:
                avg_v = full.get("avg_volume", full["volume_24h"])
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:12px; text-align:center;">
                  <div style="font-size:0.78rem; color:#888">Vol. Promedio Est.</div>
                  <div style="font-size:1.2rem; font-weight:700; color:#ffaa00">{fmt_large(avg_v)}</div>
                </div>""", unsafe_allow_html=True)
            with vc3:
                bull_v = full.get("bull_volume_est", full["volume_24h"] * 2.5)
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #00ff8844; border-radius:8px; padding:12px; text-align:center;">
                  <div style="font-size:0.78rem; color:#888">Vol. en Bull (est.)</div>
                  <div style="font-size:1.2rem; font-weight:700; color:#00ff88">{fmt_large(bull_v)}</div>
                  <div style="font-size:0.72rem; color:#666">~2.5× promedio</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚖️ Escenarios</div>', unsafe_allow_html=True)

            sa, sb = st.columns(2)
            bull_prob = scenarios.get("bull_probability", sc)
            err_prob = scenarios.get("error_probability", 100 - sc)
            bull_up = scenarios.get("bull_upside_pct", t.get("moderate_pct", 0))
            err_down = scenarios.get("error_downside_pct", t.get("downside_pct", 0))

            with sa:
                st.markdown(f"""
                <div class="scenario-bull">
                  <div class="scenario-title">🚀 ESCENARIO BULL CONFIRMADO</div>
                  <div class="scenario-pct-bull">+{bull_up:.1f}%</div>
                  <div style="font-size:1.1rem; color:#00ff88; margin-top:4px">
                    Precio objetivo: {fmt_price(t.get('moderate_price', full['current_price']))}
                  </div>
                  <div class="scenario-sub">Probabilidad estimada: <b style="color:#00ff88">{bull_prob}%</b></div>
                  <div class="scenario-sub">Basado en score técnico + momentum</div>
                </div>""", unsafe_allow_html=True)

            with sb:
                stop = t.get("stop_loss", full["current_price"] * 0.9)
                st.markdown(f"""
                <div class="scenario-bear">
                  <div class="scenario-title">❌ ESCENARIO DE ERROR (Señal Falsa)</div>
                  <div class="scenario-pct-bear">{err_down:.1f}%</div>
                  <div style="font-size:1.1rem; color:#ff4444; margin-top:4px">
                    Stop loss: {fmt_price(stop)}
                  </div>
                  <div class="scenario-sub">Probabilidad de error: <b style="color:#ff4444">{err_prob}%</b></div>
                  <div class="scenario-sub">Stop recomendado: 2× ATR por debajo entrada</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <br>
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:14px; text-align:center;">
              <span style="color:#888">Ratio Riesgo/Recompensa: </span>
              <span style="font-size:1.4rem; font-weight:800; color:{'#00ff88' if full['rr_ratio'] >= 2 else '#ffaa00' if full['rr_ratio'] >= 1 else '#ff4444'}">
                {full['rr_ratio']:.2f}×
              </span>
              <span style="color:#666; font-size:0.82rem; margin-left:8px">
                (≥ 2× = favorable · ≥ 3× = excelente)
              </span>
            </div>""", unsafe_allow_html=True)

        # ── TAB 3: Indicators ──
        with tab3:
            st.markdown('<div class="section-header">📊 Indicadores Técnicos</div>', unsafe_allow_html=True)

            if ind:
                cols = st.columns(4)
                indicators_display = [
                    ("RSI (14)", ind.get("rsi", 50), "rsi"),
                    ("MACD", ind.get("macd", 0), "macd"),
                    ("Histograma", ind.get("histogram", 0), "hist"),
                    ("Stoch %K", ind.get("stoch_k", 50), "stoch"),
                    ("EMA20", ind.get("ema20", 0), "ema"),
                    ("EMA50", ind.get("ema50", 0), "ema"),
                    ("EMA200", ind.get("ema200", 0), "ema"),
                    ("ATR", ind.get("atr", 0), "atr"),
                    ("BB Width", ind.get("bb_width", 0), "bb"),
                    ("BB %B", ind.get("bb_pct", 0.5), "bb"),
                    ("BB Upper", ind.get("bb_upper", 0), "ema"),
                    ("BB Lower", ind.get("bb_lower", 0), "ema"),
                ]
                for idx, (label, val, kind) in enumerate(indicators_display):
                    with cols[idx % 4]:
                        if kind == "rsi":
                            color = rsi_color(val)
                            interp = "SOBREVENDIDO" if val < 30 else ("NEUTRAL" if val < 70 else "SOBRECOMPRADO")
                        elif kind == "macd" or kind == "hist":
                            color = "#00ff88" if val > 0 else "#ff4444"
                            interp = "ALCISTA" if val > 0 else "BAJISTA"
                        elif kind == "stoch":
                            color = "#ff6666" if val < 20 else ("#00ff88" if val < 80 else "#ff4444")
                            interp = "SOBREVENDIDO" if val < 20 else ("NEUTRAL" if val < 80 else "SOBRECOMPRADO")
                        elif kind == "bb":
                            color = "#5588ff"
                            interp = "SQUEEZE" if (kind == "bb" and label == "BB Width" and val < 0.05) else ""
                        else:
                            color = "#aaa"
                            interp = ""

                        # Format display value
                        if kind in ["ema", "atr"] and val > 0.001:
                            disp = fmt_price(val)
                        elif kind == "bb" and label == "BB Width":
                            disp = f"{val:.4f}"
                        elif kind == "bb" and label == "BB %B":
                            disp = f"{val:.3f}"
                        else:
                            disp = f"{val:.4f}" if abs(val) < 0.1 else f"{val:.2f}"

                        st.markdown(f"""
                        <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px;
                                    padding:10px; text-align:center; margin-bottom:8px;">
                          <div style="font-size:0.72rem; color:#888; text-transform:uppercase">{label}</div>
                          <div style="font-size:1.1rem; font-weight:700; color:{color}">{disp}</div>
                          <div style="font-size:0.68rem; color:{color}; opacity:0.7">{interp}</div>
                        </div>""", unsafe_allow_html=True)

            # Score breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🧮 Desglose del Bull Score</div>', unsafe_allow_html=True)

            for comp_name, comp_data in bd.items():
                comp_score = comp_data.get("score", 0)
                comp_max = comp_data.get("max", 25)
                comp_note = comp_data.get("note", "")
                pct_fill = (comp_score / comp_max * 100) if comp_max > 0 else 0
                bar_color = "#00ff88" if pct_fill >= 70 else ("#ffaa00" if pct_fill >= 40 else "#ff4444")
                labels = {
                    "rsi": "RSI", "macd": "MACD", "trend": "Tendencia/EMAs",
                    "bollinger": "Bollinger Bands", "volume": "Volumen",
                    "momentum": "Momentum", "ath_dist": "Distancia ATH", "marketcap": "Market Cap",
                }
                label = labels.get(comp_name, comp_name.upper())
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                  <div style="min-width:140px; font-size:0.85rem; color:#ccc">{label}</div>
                  <div style="flex:1; background:#222; border-radius:4px; height:8px;">
                    <div style="width:{pct_fill:.0f}%; background:{bar_color}; height:8px; border-radius:4px;
                                transition:width 0.3s;"></div>
                  </div>
                  <div style="min-width:60px; text-align:right; font-size:0.82rem; color:{bar_color}">
                    {comp_score}/{comp_max}
                  </div>
                  <div style="min-width:200px; font-size:0.78rem; color:#888">{comp_note}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── TAB 4: Confirmations ──
        with tab4:
            st.markdown('<div class="section-header">🕯️ Guía de Confirmaciones y Velas</div>', unsafe_allow_html=True)
            confirmations = full.get("confirmations", [])

            if confirmations:
                notes_md = "\n".join(confirmations)
                st.markdown(f'<div class="confirm-box">{notes_md.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Timeframe guide
            st.markdown('<div class="section-header">⏱️ Guía de Marcos de Tiempo</div>', unsafe_allow_html=True)
            tf_c1, tf_c2 = st.columns(2)
            with tf_c1:
                st.markdown("""
                **📌 Entrada Agresiva (4H)**
                - Esperar ruptura de resistencia en gráfico 4H
                - Confirmar con volumen ≥ 150% del promedio
                - RSI 4H entre 45-65 al momento de ruptura
                - Stocástico %K cruzando hacia arriba en zona neutral

                **✅ Señales de entrada válidas:**
                - MACD cruza al alza con histograma positivo
                - Precio cierra sobre EMA20 con volumen
                - Vela envolvente alcista sobre soporte
                """)
            with tf_c2:
                st.markdown("""
                **📌 Entrada Conservadora (Diario)**
                - Confirmar cierre diario sobre nivel de resistencia clave
                - RSI diario debe estar por debajo de 65
                - 2 cierres consecutivos sobre la zona de ruptura
                - Volumen del día de ruptura ≥ 200% del promedio 20 días

                **⛔ Señales de salida (stop loss):**
                - Precio cierra por debajo de 2× ATR
                - RSI rompe hacia abajo desde zona neutral
                - Volumen en caída con velas bajistas consecutivas
                """)

            st.markdown("<br>", unsafe_allow_html=True)
            rsi_v_tab4 = bd["rsi"].get("value", 50)
            chg7d_tab4 = full["price_change"]["7d"]
            chg24h_tab4 = full["price_change"]["24h"]

            # Quick summary
            st.markdown('<div class="section-header">📋 Resumen Ejecutivo</div>', unsafe_allow_html=True)
            score_status = ("ALTA" if sc >= 72 else ("MODERADA" if sc >= 55 else ("BAJA" if sc >= 38 else "MUY BAJA")))
            rec_action = (
                "ACUMULAR en soporte con stops ajustados" if sc >= 72 else
                "OBSERVAR y esperar confirmación de volumen" if sc >= 55 else
                "WATCHLIST — sin acción inmediata" if sc >= 38 else
                "EVITAR por ahora — tendencia bajista activa"
            )
            st.markdown(f"""
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:10px; padding:20px;">
              <table style="width:100%; font-size:0.88rem;">
                <tr>
                  <td style="color:#888; padding:6px 12px">Activo:</td>
                  <td style="color:#fff; font-weight:700; padding:6px 12px">{full['symbol']} — {full['name']}</td>
                  <td style="color:#888; padding:6px 12px">Precio actual:</td>
                  <td style="color:#00ff88; font-weight:700; padding:6px 12px">{fmt_price(full['current_price'])}</td>
                </tr>
                <tr>
                  <td style="color:#888; padding:6px 12px">Bull Score:</td>
                  <td style="color:#00ff88; font-weight:700; padding:6px 12px">{sc}/100 — Probabilidad {score_status}</td>
                  <td style="color:#888; padding:6px 12px">Señal:</td>
                  <td style="color:#ffaa00; font-weight:700; padding:6px 12px">{full['signal']}</td>
                </tr>
                <tr>
                  <td style="color:#888; padding:6px 12px">Target moderado:</td>
                  <td style="color:#00ff88; font-weight:700; padding:6px 12px">
                    {fmt_price(t.get('moderate_price', 0))} (+{t.get('moderate_pct',0):.1f}%)
                  </td>
                  <td style="color:#888; padding:6px 12px">Stop loss:</td>
                  <td style="color:#ff4444; font-weight:700; padding:6px 12px">
                    {fmt_price(t.get('stop_loss', full['current_price'] * 0.9))} ({t.get('downside_pct', 0):.1f}%)
                  </td>
                </tr>
                <tr>
                  <td style="color:#888; padding:6px 12px">R/R Ratio:</td>
                  <td style="color:{'#00ff88' if full['rr_ratio'] >= 2 else '#ffaa00'}; font-weight:700; padding:6px 12px">
                    {full['rr_ratio']:.2f}×
                  </td>
                  <td style="color:#888; padding:6px 12px">Acción sugerida:</td>
                  <td style="color:#ffaa00; font-weight:700; padding:6px 12px">{rec_action}</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)


    # ─────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#444; font-size:0.78rem; border-top:1px solid #222; padding-top:16px;">
      ⚠️ Este dashboard es solo para fines informativos y educativos. No constituye consejo financiero ni de inversión.<br>
      Los mercados de criptomonedas son altamente volátiles. Siempre realiza tu propia investigación (DYOR).<br>
      Datos: CoinGecko API · Indicadores: RSI, MACD, Bollinger Bands, EMAs, Estocástico, ATR
    </div>
    """, unsafe_allow_html=True)
