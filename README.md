# 🚀 Crypto Bull Detector

> Herramienta de análisis técnico automatizado para detección de criptomonedas con alto potencial alcista y oportunidades de scalping en tiempo real.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![API](https://img.shields.io/badge/API-CoinGecko%20Free-orange)

---

## Índice

- [Descripción](#descripción)
- [Modos de operación](#modos-de-operación)
- [Arquitectura](#arquitectura)
- [Stack técnico](#stack-técnico)
- [Instalación](#instalación)
- [Uso](#uso)
- [Motor de análisis](#motor-de-análisis)
- [Indicadores técnicos](#indicadores-técnicos)
- [Sistema de scoring](#sistema-de-scoring)
- [Caché y rate limiting](#caché-y-rate-limiting)
- [Seguridad](#seguridad)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Aviso legal](#aviso-legal)

---

## Descripción

**Crypto Bull Detector** es un dashboard interactivo construido con Streamlit que conecta a la API pública de CoinGecko para analizar el mercado de criptomonedas en tiempo real. El bot aplica análisis técnico profesional para:

1. Identificar activos con mercados mal valorados y configuraciones pre-bull
2. Estimar precios objetivo usando niveles de Fibonacci
3. Calcular probabilidades de éxito/error por operación
4. Detectar actividad de grandes traders (ballenas / smart money)
5. Generar señales de scalping de 15-30 minutos

---

## Modos de operación

### ⚡ Modo Scalping (15-30 min)

Escanea las monedas más líquidas (Vol. > $10M/día) usando **candles de 30 minutos** — la granularidad mínima disponible en la API gratuita de CoinGecko.

Calcula una puntuación de scalp (0-100) basada en:

| Indicador | Peso | Lógica |
|---|---|---|
| RSI(7) | 25 pts | < 35 = sobrevendido, rebote inmediato |
| EMA5 / EMA10 cross | 25 pts | Cruce alcista = entrada NOW |
| Stochastic(5,3) | 20 pts | K < 20 + D < 20 = zona de rebote |
| Bollinger Bands(10,2) | 15 pts | %B < 0.15 o squeeze extremo |
| Momentum ROC(3) | 15 pts | Aceleración positiva de precio |
| VWAP intradiario | bonus 10 pts | Precio bajo VWAP = infravalorado |

**Señales de urgencia:**

| Score | Señal | Ventana |
|---|---|---|
| ≥ 75 | 🔥 ENTRADA AHORA | 0–15 min |
| ≥ 60 | ⚡ PRÓXIMA ENTRADA | 15–30 min |
| ≥ 45 | 👀 OBSERVAR | 30–60 min |
| ≥ 30 | ⏳ ESPERAR | 1h+ |
| < 30 | 🚫 NO OPERAR | — |

### 🚀 Modo Bull Detector

Analiza el mercado amplio (top 100 por market cap) buscando configuraciones de previo al bull run usando datos de largo plazo (OHLCV de 4H, 30 días).

**Bull Score** (0-100) ponderado por:

| Componente | Peso máx | Descripción |
|---|---|---|
| RSI(14) | 25 pts | Zona de acumulación: RSI 30-50 |
| MACD | 25 pts | Histograma virando positivo |
| Tendencia / EMAs | 25 pts | EMA20 > EMA50 con precio sobre EMA20 |
| Bollinger Bands | 15 pts | Squeeze + posición en banda baja |
| Volumen | 15 pts | Incremento de volumen confirmando movimiento |
| Momentum (30d) | 20 pts | Price change positivo en marco temporal amplio |
| Distancia al ATH | 20 pts | Activos >50% bajo ATH = mayor potencial |
| Market Cap | 10 pts | Small/mid caps priorizadas |

**Actividad de ballenas** (whale score 0-100) detectada vía:
- OBV divergence (OBV sube mientras precio cae = acumulación silenciosa)
- Chaikin Money Flow (CMF > 0.1 = dinero institucional entrando)
- VWAP position (precio bajo VWAP = infravalorado vs smart money)
- Volume spikes (volumen 3× el promedio = entrada institucional)

---

## Arquitectura

```
┌────────────────────────────────────────────────────────┐
│                     app.py (UI)                        │
│   Streamlit dashboard  ·  2 tabs  ·  Plotly charts     │
│   Modals con análisis completo  ·  st.dataframe tables │
└─────────────────────────┬──────────────────────────────┘
                          │ imports
┌─────────────────────────▼──────────────────────────────┐
│                   engine.py (Core)                     │
│                                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  API Layer  │  │  Indicators  │  │   Scoring    │  │
│  │  CoinGecko  │  │  RSI/MACD/   │  │  Bull Score  │  │
│  │  fetch_*    │  │  BB/EMA/etc  │  │  Scalp Score │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                │                 │           │
│  ┌──────▼──────────────────────────────────▼───────┐   │
│  │              Cache Layer (.crypto_cache/)        │   │
│  │      Pickle TTL: 3min (scalp) / 10min (OHLCV)   │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼──────────────────────────────┐
│              CoinGecko API v3 (Free Tier)              │
│  /coins/markets  ·  /coins/{id}/ohlc                   │
│  /coins/{id}/market_chart                              │
└────────────────────────────────────────────────────────┘
```

---

## Stack técnico

| Componente | Tecnología | Versión mínima |
|---|---|---|
| UI / Dashboard | Streamlit | 1.32.0 |
| Visualización | Plotly | 5.18.0 |
| Manipulación de datos | Pandas | 2.0.0 |
| Cálculo numérico | NumPy | 1.24.0 |
| HTTP client | Requests | 2.31.0 |
| Runtime | Python | 3.9+ |
| Datos de mercado | CoinGecko API | Free / v3 |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/crypto-bull-detector.git
cd crypto-bull-detector

# 2. Crear entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

---

## Uso

### Sidebar — Configuración

| Parámetro | Descripción | Default |
|---|---|---|
| Nº de criptos a analizar | Cuántas monedas del top global analizar | 100 |
| Bull Score mínimo | Filtro de puntuación mínima | 0 |
| Market cap mínimo | Filtro por capitalización | Sin filtro |
| Ordenar por | Criterio de ordenación de la tabla | Bull Score |

### Tab Scalping

1. Selecciona cuántas monedas escanear (10–50)
2. Haz click en **"🔍 Escanear ahora"**
3. Las tarjetas de urgencia inmediata aparecen en la parte superior
4. Haz click en **"📋 Detalle"** para ver todos los indicadores

> Los datos de scalping se cachean 3 minutos. Para forzar refresco, usa el botón de actualizar en la sidebar.

### Tab Bull Detector

1. Las tarjetas **TOP PICKS** muestran las 6 con mayor puntuación
2. Haz click en **"🔬 Ver Análisis · SYMBOL"** para abrir el modal con el análisis completo
3. El modal muestra:
   - Desglose del Bull Score por indicador
   - Gráfico de precio 30 días con EMAs, Bollinger, RSI y MACD
   - Precios objetivo en 3 escenarios (Fibonacci 0.618 / 1.0 / 1.618)
   - Probabilidad de bull vs probabilidad de error
   - Señales de confirmación: qué velas esperar

---

## Motor de análisis

### Flujo de datos

```
fetch_markets(n)
    └─► analyze_coin_quick(coin)        ← análisis rápido sin OHLCV
            └─► [RSI desde sparkline, MACD estimado, scoring rápido]

[al abrir modal o análisis detallado]
    └─► enrich_with_ohlcv(coin)        ← descarga candles 4H / 30 días
            ├─► fetch_ohlcv(id, 30)
            ├─► fetch_volume_chart(id)
            ├─► fetch_hourly_chart(id)
            ├─► analyze_whale_activity(hourly, vols)
            └─► [targets Fibonacci, confirmaciones, indicadores completos]

[modo scalping]
    └─► batch_scalp_scan(coins, n)
            └─► analyze_scalp(id, price, vol, data)
                    └─► fetch_scalp_candles(id)   ← OHLC 1 día = candles 30min
```

### Cálculo de precios objetivo

Los targets se calculan usando extensiones de Fibonacci sobre el ATH y el soporte identificado:

```python
# Conservador: Fibonacci 0.618
target_conservative = support + (ath - support) * 0.618

# Moderado: Fibonacci 1.0 (retroceso completo)
target_moderate = ath * 0.95

# Agresivo: Fibonacci 1.618
target_aggressive = support + (ath - support) * 1.618

# Stop loss: precio - (ATR × 1.5)
stop_loss = current_price - atr * 1.5
```

### R/R Ratio

```python
rr_ratio = upside_pct / abs(downside_pct)
# R/R ≥ 2 = favorable (verde)
# R/R 1-2 = aceptable (naranja)
# R/R < 1 = desfavorable (rojo)
```

---

## Indicadores técnicos

Todos los indicadores están implementados en `engine.py` sin dependencias externas (TA-Lib, ta, etc.):

| Función | Descripción | Parámetros |
|---|---|---|
| `calc_rsi(close, period)` | Relative Strength Index con EWM | 14 (bull) / 7 (scalp) |
| `calc_macd(close, fast, slow, signal)` | MACD line + signal + histogram | 12, 26, 9 |
| `calc_bb(close, period, std_dev)` | Bollinger Bands + width + %B | 20, 2 (bull) / 10, 2 (scalp) |
| `calc_ema(close, period)` | Exponential Moving Average | variable |
| `calc_stoch(high, low, close, k, d)` | Stochastic Oscillator | 14, 3 (bull) / 5, 3 (scalp) |
| `calc_atr(high, low, close, period)` | Average True Range (EWM) | 14 |
| `calc_obv(close, volume)` | On-Balance Volume acumulado | — |
| `calc_cmf(close, volume, period)` | Chaikin Money Flow simplificado | 20 |
| `calc_vwap(close, volume)` | Volume Weighted Average Price | — |

---

## Sistema de scoring

### Bull Score (0-100)

Implementado en `analyze_coin_quick()` y `enrich_with_ohlcv()`:

```
bull_score = sum(component_scores)

donde:
  rsi_score      ∈ [0, 25]
  macd_score     ∈ [0, 25]
  trend_score    ∈ [0, 25]
  bb_score       ∈ [0, 15]
  volume_score   ∈ [0, 15]
  momentum_score ∈ [0, 20]
  ath_score      ∈ [0, 20]
  mcap_score     ∈ [0, 10]
```

**Etiquetas:**

| Score | Status | Label |
|---|---|---|
| ≥ 72 | `hot` | 🔥 ALTO POTENCIAL |
| ≥ 55 | `warm` | ⚡ PRE-BULL |
| ≥ 38 | `neutral` | 🎯 ACUMULACIÓN |
| < 38 | `cold` | ❄️ SIN SEÑAL |

### Scalp Score (0-100)

Implementado en `analyze_scalp()`:

```
scalp_score = rsi7_score + ema_cross_score + stoch_score + bb_score + roc_score + vwap_bonus
```

---

## Caché y rate limiting

### Estrategia de caché

El sistema usa archivos Pickle con TTL para minimizar llamadas a la API:

| Tipo de dato | Función | TTL | Archivo |
|---|---|---|---|
| Mercados (top N) | `fetch_markets()` | 5 min | `markets_{n}.pkl` |
| OHLCV 4H (30d) | `fetch_ohlcv()` | 10 min | `ohlcv_{id}_{d}.pkl` |
| Volumen diario | `fetch_volume_chart()` | 10 min | `vol_{id}_{d}.pkl` |
| Horario (14d) | `fetch_hourly_chart()` | 10 min | `hourly_{id}_{d}.pkl` |
| Scalp (30min) | `fetch_scalp_candles()` | 3 min | `scalp_{id}.pkl` |

El directorio `.crypto_cache/` se crea automáticamente en el directorio de trabajo y se lista en `.gitignore`.

### Rate limiting

La API gratuita de CoinGecko permite ~30 requests/minuto:

```python
time.sleep(0.35)   # entre requests de OHLCV
time.sleep(0.5)    # entre monedas en batch_scalp_scan
time.sleep(65)     # al recibir HTTP 429 (rate limit)
```

---

## Seguridad

- **Sin credenciales hardcodeadas**: La app no requiere API keys (CoinGecko free tier es público)
- **Sin base de datos**: No persiste datos de usuario ni información sensible
- **Sin eval/exec**: No ejecuta código dinámico
- **Pickle cache**: Solo almacena datos de mercado públicos, sin información sensible
- **HTTPS only**: Todas las llamadas a CoinGecko usan HTTPS

### Para producción

Si se despliega en un servidor público, se recomienda:

```toml
# .streamlit/config.toml
[server]
headless = true
address = "0.0.0.0"
port = 8501

[browser]
gatherUsageStats = false
```

Para limitar acceso, usar autenticación a nivel de proxy (nginx/Caddy) o el módulo `streamlit-authenticator`.

---

## Estructura del proyecto

```
crypto-bull-detector/
├── app.py                    # Dashboard Streamlit (UI, tabs, modales)
├── engine.py                 # Motor de análisis técnico
├── requirements.txt          # Dependencias Python
├── .gitignore                # Archivos ignorados por Git
├── README.md                 # Este archivo
└── .streamlit/
    └── config.toml           # Tema y configuración del servidor
```

**Archivos generados en runtime (no versionados):**

```
.crypto_cache/                # Caché de datos de mercado (auto-limpia)
__pycache__/                  # Bytecode Python compilado
```

---

## Limitaciones conocidas

| Limitación | Detalle |
|---|---|
| Granularidad mínima | CoinGecko free tier: 30 min (OHLC 1 día). Sin acceso a candles de 1m o 5m. |
| Sin order book | No hay datos de profundidad de mercado (bid/ask) |
| Sin datos on-chain | El análisis de ballenas es proxy (OBV/CMF/VWAP), no transacciones reales |
| Rate limit | ~30 req/min en free tier. El escaneo de 50 monedas puede tardar ~30 segundos |
| Mercados soportados | Solo pares USD (CoinGecko `vs_currency=usd`) |
| API premium | Para datos de 1-5 min o historical tick data, se requiere CoinGecko Pro |

---

## Aviso legal

> **Este software es solo para fines educativos e informativos.**
> No constituye asesoramiento financiero ni recomendación de inversión.
> El trading de criptomonedas conlleva riesgo significativo de pérdida.
> Usa siempre stop-loss y gestión de riesgo adecuada.
> El autor no se hace responsable de pérdidas derivadas del uso de esta herramienta.

---

## Licencia

MIT License — libre para uso, modificación y distribución con atribución.
