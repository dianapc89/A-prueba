"""
app.py
------
App de Streamlit: Indicadores de desempeño / valuación de activos
financieros (Renta Variable), a partir de precios de cierre de Yahoo
Finance.

Ejecutar localmente:
    streamlit run app.py

Desplegar en Streamlit Community Cloud:
    1. Sube este repo a GitHub (ver README.md).
    2. En share.streamlit.io, conecta el repo y selecciona app.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance_utils import (
    NIVELES_CONFIANZA,
    PERIODICIDAD_MAP,
    PLAZO_MAP,
    calcular_metricas_activo,
    fetch_close_prices,
    fetch_us_risk_free_rate,
    is_us_ticker,
)

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
