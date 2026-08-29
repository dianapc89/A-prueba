"""
app.py
------
App de Streamlit: Indicadores de desempeño / valuación de activos
financieros (Renta Variable), a partir de precios de cierre de Yahoo
Finance. Todo el código (UI + cálculos) vive en este único archivo para
evitar errores de módulos faltantes al desplegar.

Ejecutar localmente:
    streamlit run app.py

Desplegar en Streamlit Community Cloud:
    1. Sube app.py y requirements.txt a la raíz de tu repo de GitHub.
    2. En share.streamlit.io, conecta el repo y selecciona app.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Catálogos / mapeos de los inputs del usuario a parámetros técnicos
# ---------------------------------------------------------------------------

# "Plazo a calcular" -> string de periodo que entiende yfinance
PLAZO_MAP = {
    "5 días": "5d",
    "3 Meses": "3mo",
    "6 meses": "6mo",
    "YTD": "ytd",
    "12 meses": "1y",
    "1 año": "1y",
    "5 años": "5y",
}

# "Periodicidad de los precios" -> intervalo de yfinance
PERIODICIDAD_MAP = {
    "Diaria": "1d",
    "Semanal": "1wk",
    "Mensual": "1mo",
}

# Número de periodos por año, usado para anualizar volatilidad (sigma * sqrt(n))
PERIODOS_POR_ANIO = {
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}

# Confianza -> valor crítico z (un extremo / one-tailed), usado para VaR
NIVELES_CONFIANZA = {
    "90%": 0.90,
    "95%": 0.95,
    "97.5%": 0.975,
    "99%": 0.99,
}

# Ticker del índice del Tesoro de EUA (10 años) usado como proxy de tasa
# libre de riesgo para activos de EUA. Yahoo Finance reporta ^TNX como el
# rendimiento anual * 10 (p. ej. 44.50 = 4.45%), por lo que se divide entre
# 1000 para obtener la tasa decimal anual.
US_RISK_FREE_TICKER = "^TNX"


# ---------------------------------------------------------------------------
# Utilidades de tickers / origen del activo
# ---------------------------------------------------------------------------

def is_us_ticker(ticker: str) -> bool:
    """
    Heurística simple: en Yahoo Finance los tickers de EUA (NYSE/NASDAQ) no
    llevan sufijo de bolsa (ej. "AAPL", "MSFT"), mientras que los de otros
    países sí (ej. "WALMEX.MX", "SHOP.TO", "VALE3.SA", "SAN.MC").
    """
    ticker = ticker.strip().upper()
    return "." not in ticker and not ticker.startswith("^")


# ---------------------------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------------------------

def fetch_close_prices(tickers: list, plazo_label: str, periodicidad_label: str) -> pd.DataFrame:
    """
    Descarga precios de cierre (ajustados) para una lista de tickers.
    Regresa un DataFrame con una columna por ticker, indexado por fecha.
    """
    period = PLAZO_MAP[plazo_label]
    interval = PERIODICIDAD_MAP[periodicidad_label]

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if data.empty:
        raise ValueError(
            "Yahoo Finance no devolvió datos para los tickers/periodo solicitados. "
            "Verifica los símbolos y el plazo seleccionado."
        )

    closes = {}
    if len(tickers) == 1:
        # yfinance no agrupa por ticker cuando solo se pide uno
        closes[tickers[0]] = data["Close"]
    else:
        for t in tickers:
            try:
                closes[t] = data[t]["Close"]
            except (KeyError, TypeError):
                # Fallback por si yfinance regresó columnas planas
                if "Close" in data.columns:
                    closes[t] = data["Close"]

    df = pd.DataFrame(closes).dropna(how="all")
    return df


def fetch_us_risk_free_rate():
    """
    Trae la tasa libre de riesgo anual (decimal, ej. 0.045 = 4.5%) para
    activos de EUA a partir del rendimiento del bono del Tesoro a 10 años
    (^TNX). Si falla la descarga, regresa None para que la UI pida el dato
    manualmente.
    """
    try:
        hist = yf.Ticker(US_RISK_FREE_TICKER).history(period="5d")
        if hist.empty:
            return None
        last_value = hist["Close"].dropna().iloc[-1]
        return float(last_value) / 1000.0  # ^TNX cotiza el rendimiento *10
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cálculo de retornos y métricas (fórmulas del documento anexo)
# ---------------------------------------------------------------------------

def compute_period_returns(prices: pd.Series) -> pd.Series:
    """Rendimientos simples periodo a periodo."""
    return prices.pct_change().dropna()


def n_years_span(prices: pd.Series) -> float:
    """Número de años (fracción) que abarca la serie de precios."""
    delta_days = (prices.index[-1] - prices.index[0]).days
    return max(delta_days / 365.25, 1e-6)


def retorno_anualizado(prices: pd.Series) -> float:
    """
    Retorno Anual = (Valor Final / Valor Inicial)^(1/n) - 1
    n = número de años que cubre la serie de precios.
    """
    n = n_years_span(prices)
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1.0 / n) - 1.0)


def volatilidad_anualizada(returns: pd.Series, interval: str) -> float:
    """
    Volatilidad Anual = sigma * sqrt(n)
    sigma = desviación estándar de los retornos del periodo elegido
    n = número de periodos en un año (252 días, 52 semanas, 12 meses)
    """
    n = PERIODOS_POR_ANIO[interval]
    return float(returns.std(ddof=1) * np.sqrt(n))


def coeficiente_pearson(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def beta_activo(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Beta = Cov(Ri, Rm) / Var(Rm)
    """
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    cov_im = cov_matrix[0, 1]
    var_m = cov_matrix[1, 1]
    if var_m == 0:
        return float("nan")
    return float(cov_im / var_m)


def indice_sharpe(Rp: float, Rf: float, sigma_p: float) -> float:
    """Índice Sharpe = (Rp - Rf) / sigma_p"""
    if sigma_p == 0:
        return float("nan")
    return (Rp - Rf) / sigma_p


def indice_traynor(Ra: float, Rf: float, beta_a: float) -> float:
    """Índice Traynor = (Ra - Rf) / beta_a"""
    if beta_a == 0 or np.isnan(beta_a):
        return float("nan")
    return (Ra - Rf) / beta_a


def capm_retorno_esperado(Rf: float, beta_i: float, Rm: float) -> float:
    """CAPM: Ri = Rf + beta_i * (Rm - Rf)"""
    return Rf + beta_i * (Rm - Rf)


def alpha_activo(Ri: float, Rf: float, beta_i: float, Rm: float) -> float:
    """Alpha = Ri - [Rf + beta_i * (Rm - Rf)]"""
    return Ri - capm_retorno_esperado(Rf, beta_i, Rm)


def valor_z(nivel_confianza: float) -> float:
    """
    Valor crítico z de la distribución normal estándar para un VaR a un
    extremo (one-tailed). Para 95% de confianza, z ~ -1.645 (percentil 5).
    Se regresa como valor negativo porque así se usa directamente en la
    fórmula VaR = mu + z * sigma.
    """
    return float(norm.ppf(1.0 - nivel_confianza))


def value_at_risk(mu_period, sigma_period, z, capital, horizonte_dias, interval):
    """
    VaR_alpha = mu + z_alpha * sigma   (fórmula del documento anexo)

    mu_period / sigma_period: media y desviación estándar de los retornos
        en la periodicidad elegida por el usuario (diaria/semanal/mensual).
    Se escalan al horizonte de VaR solicitado (en días) usando la regla
    de la raíz del tiempo.

    Regresa (VaR_pct, VaR_dolares), ambos como pérdida esperada positiva.
    """
    dias_por_periodo = 365.25 / PERIODOS_POR_ANIO[interval]
    n_periodos_horizonte = max(horizonte_dias / dias_por_periodo, 1e-6)

    mu_h = mu_period * n_periodos_horizonte
    sigma_h = sigma_period * np.sqrt(n_periodos_horizonte)

    var_retorno = mu_h + z * sigma_h  # típicamente negativo (pérdida)
    var_pct = -var_retorno * 100.0
    var_dolares = -var_retorno * capital

    return float(var_pct), float(var_dolares)


def calcular_metricas_activo(
    prices_asset, prices_market, rf_anual, interval,
    nivel_confianza_label, capital, horizonte_var_dias,
) -> dict:
    """Calcula todas las métricas de salida para un solo activo."""

    ret_asset = compute_period_returns(prices_asset)
    ret_market = compute_period_returns(prices_market)

    rent_anual_activo = retorno_anualizado(prices_asset)
    rent_anual_mercado = retorno_anualizado(prices_market)

    vol_anual = volatilidad_anualizada(ret_asset, interval)
    pearson = coeficiente_pearson(ret_asset, ret_market)
    beta = beta_activo(ret_asset, ret_market)

    sharpe = indice_sharpe(rent_anual_activo, rf_anual, vol_anual)
    traynor = indice_traynor(rent_anual_activo, rf_anual, beta)
    capm = capm_retorno_esperado(rf_anual, beta, rent_anual_mercado)
    alpha = alpha_activo(rent_anual_activo, rf_anual, beta, rent_anual_mercado)

    nivel_confianza = NIVELES_CONFIANZA[nivel_confianza_label]
    z = valor_z(nivel_confianza)

    var_pct, var_dolares = value_at_risk(
        mu_period=ret_asset.mean(),
        sigma_period=ret_asset.std(ddof=1),
        z=z,
        capital=capital,
        horizonte_dias=horizonte_var_dias,
        interval=interval,
    )

    return {
        "Rentabilidad anualizada": rent_anual_activo,
        "Volatilidad anualizada": vol_anual,
        "iSharpe": sharpe,
        "iTraynor": traynor,
        "Coef. Correlación Pearson": pearson,
        "BETA": beta,
        "CAPM": capm,
        "Alpha": alpha,
        "Valor z": z,
        "VaR %": var_pct,
        "VaR $$": var_dolares,
        "_ret_asset": ret_asset,
        "_ret_market": ret_market,
    }


# ---------------------------------------------------------------------------
# Configuración de página + estilo "fintech" azul / negro / blanco (Arial)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Métricas de Valuación de Activos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

AZUL_PRIMARIO = "#1565FF"
AZUL_OSCURO = "#0A1A3C"
NEGRO_FONDO = "#05070D"
BLANCO = "#F5F7FA"

CUSTOM_CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Arial', 'Helvetica Neue', sans-serif !important;
        color: {BLANCO};
    }}
    .stApp {{
        background: linear-gradient(180deg, {NEGRO_FONDO} 0%, {AZUL_OSCURO} 100%);
    }}
    section[data-testid="stSidebar"] {{
        background-color: {NEGRO_FONDO};
        border-right: 1px solid {AZUL_PRIMARIO};
    }}
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {BLANCO} !important;
    }}
    h1 {{
        border-bottom: 2px solid {AZUL_PRIMARIO};
        padding-bottom: 0.4rem;
    }}
    .stButton>button {{
        background-color: {AZUL_PRIMARIO};
        color: {BLANCO} !important;
        border: none;
        border-radius: 6px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
    }}
    .stButton>button:hover {{
        background-color: #0D4FCC;
        color: {BLANCO} !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: rgba(21, 101, 255, 0.12);
        border: 1px solid {AZUL_PRIMARIO};
        border-radius: 10px;
        padding: 0.8rem;
    }}
    .stDataFrame {{
        border: 1px solid {AZUL_PRIMARIO};
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {BLANCO};
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {AZUL_PRIMARIO};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=NEGRO_FONDO,
    plot_bgcolor=NEGRO_FONDO,
    font=dict(family="Arial", color=BLANCO),
    xaxis=dict(gridcolor="#1E2A4A", zerolinecolor="#1E2A4A"),
    yaxis=dict(gridcolor="#1E2A4A", zerolinecolor="#1E2A4A"),
)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("📈 Métricas de Valuación de Activos Financieros")
st.caption(
    "Renta Variable · Rentabilidad, riesgo, CAPM y Valor en Riesgo (VaR) "
    "a partir de precios de cierre de Yahoo Finance."
)

# ---------------------------------------------------------------------------
# Sidebar — Inputs
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Parámetros")

num_activos = st.sidebar.number_input(
    "Número de activos a valuar", min_value=1, max_value=15, value=2, step=1
)

st.sidebar.markdown("**Tickers de los activos** (formato Yahoo Finance)")
tickers = []
for i in range(int(num_activos)):
    default = "AAPL" if i == 0 else ("WALMEX.MX" if i == 1 else "")
    t = st.sidebar.text_input(f"Ticker #{i + 1}", value=default, key=f"ticker_{i}")
    if t.strip():
        tickers.append(t.strip().upper())

indice = st.sidebar.text_input(
    "Índice bursátil de referencia (benchmark)",
    value="^GSPC",
    help="Ej. ^GSPC (S&P 500), ^DJI (Dow Jones), ^MXX (IPC México), ^IXIC (Nasdaq).",
)

periodicidad_label = st.sidebar.selectbox(
    "Periodicidad de los precios", list(PERIODICIDAD_MAP.keys()), index=0
)
plazo_label = st.sidebar.selectbox(
    "Plazo a calcular", list(PLAZO_MAP.keys()), index=4
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parámetros de VaR**")
capital = st.sidebar.number_input(
    "Monto de capital a invertir", min_value=0.0, value=100000.0, step=1000.0, format="%.2f"
)
nivel_confianza_label = st.sidebar.selectbox(
    "Intervalo de confianza", list(NIVELES_CONFIANZA.keys()), index=1
)
horizonte_var = st.sidebar.number_input(
    "Plazo para VaR (días)", min_value=1, value=1, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tasa libre de riesgo**")
st.sidebar.caption(
    "Para activos de EUA se trae automáticamente del bono del Tesoro a 10 años "
    "(^TNX, Yahoo Finance). Para activos de otros países, captúrala manualmente."
)

rf_manual = {}
rf_auto_us = None
hay_activos_us = any(is_us_ticker(t) for t in tickers)
if hay_activos_us:
    rf_auto_us = fetch_us_risk_free_rate()
    if rf_auto_us is not None:
        st.sidebar.info(f"Tasa libre de riesgo EUA (^TNX): **{rf_auto_us * 100:.2f}%** anual")
    else:
        st.sidebar.warning(
            "No se pudo obtener ^TNX automáticamente. Captura la tasa manualmente."
        )

for t in tickers:
    if not is_us_ticker(t):
        rf_manual[t] = st.sidebar.number_input(
            f"Tasa libre de riesgo anual (%) — {t}",
            min_value=-5.0, max_value=100.0, value=8.0, step=0.1,
            key=f"rf_{t}",
        ) / 100.0
    elif rf_auto_us is None:
        rf_manual[t] = st.sidebar.number_input(
            f"Tasa libre de riesgo anual (%) — {t}",
            min_value=-5.0, max_value=100.0, value=4.5, step=0.1,
            key=f"rf_{t}",
        ) / 100.0

calcular = st.sidebar.button("🚀 Calcular indicadores", use_container_width=True)

# ---------------------------------------------------------------------------
# Cálculo y despliegue de resultados
# ---------------------------------------------------------------------------

if calcular:
    if not tickers:
        st.error("Captura al menos un ticker válido.")
        st.stop()
    if not indice.strip():
        st.error("Captura un ticker de índice bursátil de referencia.")
        st.stop()

    with st.spinner("Descargando precios de Yahoo Finance y calculando métricas..."):
        try:
            all_tickers = tickers + [indice.strip().upper()]
            prices = fetch_close_prices(all_tickers, plazo_label, periodicidad_label)
        except Exception as e:
            st.error(f"No se pudieron descargar los precios: {e}")
            st.stop()

        if indice.strip().upper() not in prices.columns:
            st.error(
                f"No se encontraron precios para el índice '{indice}'. "
                "Verifica el símbolo."
            )
            st.stop()

        interval = PERIODICIDAD_MAP[periodicidad_label]
        prices_market = prices[indice.strip().upper()].dropna()

        resultados = {}
        detalle_regresion = {}
        for t in tickers:
            if t not in prices.columns or prices[t].dropna().empty:
                st.warning(f"Sin datos para '{t}'. Se omite del análisis.")
                continue

            rf_anual = rf_auto_us if (is_us_ticker(t) and rf_auto_us is not None) else rf_manual.get(t)
            if rf_anual is None:
                st.warning(f"Sin tasa libre de riesgo para '{t}'. Se omite del análisis.")
                continue

            prices_asset = prices[t].dropna()
            aligned = pd.concat(
                [prices_asset.rename("asset"), prices_market.rename("market")], axis=1
            ).dropna()

            if len(aligned) < 5:
                st.warning(f"Muy pocos datos alineados para '{t}' vs el índice. Se omite.")
                continue

            metricas = calcular_metricas_activo(
                prices_asset=aligned["asset"],
                prices_market=aligned["market"],
                rf_anual=rf_anual,
                interval=interval,
                nivel_confianza_label=nivel_confianza_label,
                capital=capital,
                horizonte_var_dias=horizonte_var,
            )
            resultados[t] = metricas
            detalle_regresion[t] = (metricas.pop("_ret_asset"), metricas.pop("_ret_market"))

    if not resultados:
        st.error("No se pudo calcular ningún activo. Revisa tickers, índice y tasas.")
        st.stop()

    # --- Tabla resumen ---------------------------------------------------
    st.subheader("📊 Tabla de indicadores por activo")

    df_resultados = pd.DataFrame(resultados).T
    df_display = df_resultados.copy()
    pct_cols = ["Rentabilidad anualizada", "Volatilidad anualizada", "CAPM", "Alpha", "VaR %"]
    for c in pct_cols:
        df_display[c] = df_display[c].map(lambda x: f"{x * 100:.2f}%" if c != "VaR %" else f"{x:.2f}%")
    for c in ["iSharpe", "iTraynor", "Coef. Correlación Pearson", "BETA", "Valor z"]:
        df_display[c] = df_display[c].map(lambda x: f"{x:.3f}")
    df_display["VaR $$"] = df_resultados["VaR $$"].map(lambda x: f"${x:,.2f}")

    col_order = [
        "Rentabilidad anualizada", "Volatilidad anualizada", "iSharpe", "iTraynor",
        "Coef. Correlación Pearson", "BETA", "CAPM", "Alpha", "Valor z", "VaR %", "VaR $$",
    ]
    st.dataframe(df_display[col_order], use_container_width=True)

    st.caption(
        f"Tasa libre de riesgo, capital de ${capital:,.2f}, confianza {nivel_confianza_label}, "
        f"horizonte VaR {horizonte_var} día(s). Índice de referencia: {indice.strip().upper()}."
    )

    # --- Tarjetas por activo ---------------------------------------------
    st.subheader("🧮 Detalle por activo")
    tabs = st.tabs(tickers if len(tickers) == len(resultados) else list(resultados.keys()))
    for tab, t in zip(tabs, resultados.keys()):
        with tab:
            m = resultados[t]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rentabilidad anualizada", f"{m['Rentabilidad anualizada'] * 100:.2f}%")
            c2.metric("Volatilidad anualizada", f"{m['Volatilidad anualizada'] * 100:.2f}%")
            c3.metric("BETA", f"{m['BETA']:.3f}")
            c4.metric("iSharpe", f"{m['iSharpe']:.3f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("iTraynor", f"{m['iTraynor']:.3f}")
            c6.metric("CAPM (esperado)", f"{m['CAPM'] * 100:.2f}%")
            c7.metric("Alpha", f"{m['Alpha'] * 100:.2f}%")
            c8.metric("VaR", f"{m['VaR %']:.2f}%  ·  ${m['VaR $$']:,.0f}")

            # --- Gráfica de correlación y regresión vs índice ------------
            ret_asset, ret_market = detalle_regresion[t]
            x = ret_market.values * 100
            y = ret_asset.values * 100

            if len(x) >= 2:
                pendiente, intercepto = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                y_line = pendiente * x_line + intercepto

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers", name=f"{t} vs {indice.strip().upper()}",
                    marker=dict(color=AZUL_PRIMARIO, size=7, opacity=0.75),
                ))
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines", name="Regresión lineal",
                    line=dict(color="#66D2FF", width=2, dash="dash"),
                ))
                fig.update_layout(
                    title=f"Correlación y regresión: {t} vs {indice.strip().upper()} "
                          f"(Pearson r = {m['Coef. Correlación Pearson']:.3f})",
                    xaxis_title=f"Retorno {indice.strip().upper()} (%)",
                    yaxis_title=f"Retorno {t} (%)",
                    height=450,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay suficientes observaciones para graficar la regresión.")
else:
    st.info(
        "Configura los parámetros en el panel izquierdo y presiona "
        "**🚀 Calcular indicadores** para generar la tabla y las gráficas."
    )
    with st.expander("ℹ️ ¿Qué calcula esta app?"):
        st.markdown(
            """
- **Rentabilidad anualizada** — (Valor Final / Valor Inicial)^(1/n) − 1
- **Volatilidad anualizada** — σ (desviación estándar de retornos) × √n
- **iSharpe** — (Rp − Rf) / σp
- **iTraynor** — (Ra − Rf) / βa
- **Coef. Correlación Pearson** — correlación entre retornos del activo y el índice
- **BETA** — Cov(Ri, Rm) / Var(Rm)
- **CAPM** — Rf + β × (Rm − Rf)
- **Alpha** — Ri − CAPM(Ri)
- **Valor "z"** — valor crítico de la normal estándar para el nivel de confianza elegido
- **VaR %  /  VaR $$** — pérdida máxima esperada, en % y en monto de capital
            """
        )
