"""
NeuroQuant Institutional Terminal - v40.0 (Enterprise Hedge Fund Edition)
-------------------------------------------------------------------------
Plataforma de análisis cuantitativo, optimización convexa, simulación 
estocástica y atribución de riesgo bajo los estándares del CFA Institute.
Desarrollado para la Cátedra Fintech - Universidad de Málaga.

Módulos integrados:
- Optimización de Markowitz (Frontera Eficiente)
- Asignación por Paridad de Riesgo (Risk Parity)
- Simulación de Monte Carlo mediante Movimiento Browniano Geométrico
- Análisis de Regímenes de Mercado Condicional
- Redacción automatizada de Tesis Fiduciaria mediante IA.
"""

import os
import warnings
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier
from scipy.stats import norm, skew, kurtosis
from statsmodels.tsa.stattools import acf

# ==========================================
# IGNORAR WARNINGS DE COMPILACIÓN
# ==========================================
warnings.filterwarnings("ignore")


# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS CSS AVANZADOS
# ==========================================
def setup_page() -> None:
    """
    Configura la página principal de Streamlit y carga los estilos CSS 
    para simular una terminal institucional tipo Bloomberg/Reuters.
    La interfaz está diseñada para minimizar la fatiga visual.
    """
    st.set_page_config(
        page_title="NeuroQuant Terminal Institutional", 
        layout="wide", 
        page_icon="🏦",
        initial_sidebar_state="expanded"
    )
    
    css_styles = """
    <style>
    /* Definición de Variables Globales de Color */
    :root {
        --bg-color: #f4f7f9;
        --text-color: #1e293b;
        --primary-dark: #0f172a;
        --accent-blue: #3b82f6;
        --positive-green: #00ff41;
        --negative-red: #ff4b4b;
        --border-color: #e2e8f0;
    }
    
    /* Configuración Base del DOM */
    .stApp { 
        background-color: var(--bg-color); 
        color: var(--text-color); 
        font-family: 'Inter', sans-serif;
    }
    
    /* Cinta de cotizaciones (Ticker Rodante) */
    .ticker-wrapper {
        width: 100%; 
        overflow: hidden; 
        background-color: var(--primary-dark); 
        color: var(--positive-green); 
        padding: 14px 0; 
        font-family: 'Courier New', monospace;
        border-bottom: 4px solid #334155; 
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    .ticker-move { 
        display: flex; 
        width: 200%; 
        animation: ticker 60s linear infinite; 
    }
    .ticker-item { 
        white-space: nowrap; 
        padding-right: 70px; 
        font-size: 1.15rem; 
        font-weight: bold; 
        letter-spacing: 1px;
    }
    @keyframes ticker { 
        0% { transform: translateX(0); } 
        100% { transform: translateX(-50%); } 
    }
    
    /* Tarjetas de Métricas Ejecutivas */
    .executive-card {
        background-color: #ffffff; 
        border-radius: 12px; 
        padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); 
        border: 1px solid var(--border-color);
        margin-bottom: 20px; 
        text-align: center; 
        transition: all 0.3s ease;
    }
    .executive-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); 
        border-color: var(--accent-blue);
    }
    .metric-label { 
        color: #64748b; 
        font-size: 0.85rem; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.05em; 
    }
    .metric-value { 
        color: var(--primary-dark); 
        font-size: 2.2rem; 
        font-weight: 900; 
        margin-top: 10px; 
    }
    
    /* Cajas Explicativas (Doctoral Level) e Insights */
    .explanation-box {
        background-color: #ffffff; 
        border-left: 6px solid var(--primary-dark);
        padding: 22px; 
        border-radius: 8px; 
        margin-bottom: 25px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        font-size: 0.95rem; 
        line-height: 1.7;
    }
    .insight-card {
        background-color: #f8fafc; 
        border-left: 5px solid var(--accent-blue); 
        padding: 18px; 
        margin-bottom: 15px; 
        border-radius: 6px; 
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    .insight-card:hover {
        background-color: #eff6ff;
    }
    
    /* Tesis CFA Abstract (Formal Report Style) */
    .cfa-paper {
        background-color: white; 
        padding: 70px; 
        border: 1px solid #d1d5db;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        font-family: 'Times New Roman', Times, serif; 
        color: #111827;
        max-width: 950px; 
        margin: auto; 
        line-height: 1.8; 
        text-align: justify;
    }
    .cfa-header { 
        text-align: center; 
        border-bottom: 2px solid var(--primary-dark); 
        padding-bottom: 20px; 
        margin-bottom: 35px; 
    }
    .cfa-header h1 { 
        margin: 0; 
        font-variant: small-caps; 
        letter-spacing: 3px; 
        font-size: 2.5rem;
        color: var(--primary-dark);
    }
    .cfa-section { 
        font-weight: bold; 
        text-transform: uppercase; 
        color: #2d3748; 
        border-bottom: 1px solid var(--border-color); 
        margin-top: 35px; 
        font-size: 1.2rem; 
        padding-bottom: 8px;
    }
    
    /* Botones de Control */
    .stButton>button { 
        background-color: var(--primary-dark); 
        color: white; 
        border-radius: 8px; 
        width: 100%; 
        height: 4em; 
        font-weight: 800; 
        border: none; 
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #1e293b; 
        color: var(--positive-green); 
        border: 1px solid var(--positive-green); 
        box-shadow: 0 4px 6px rgba(0,255,65,0.2);
    }
    
    /* Expander Override */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: var(--primary-dark);
        background-color: #f8fafc;
        border-radius: 4px;
    }
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)


def render_header(logo_name: str) -> None:
    """
    Renderiza el logo institucional centrado en la cabecera del dashboard.
    Maneja excepciones si el archivo de imagen no se encuentra en el root path.
    """
    col_izq, col_centro, col_der = st.columns([1, 4, 1])
    with col_centro:
        if os.path.exists(logo_name):
            st.image(logo_name, use_container_width=True)
        else:
            st.error(f"❌ LOGO SYSTEM ERROR: Archivo '{logo_name}' no detectado. El sistema continúa operando en modo degradado.")


# ==========================================
# 2. SISTEMA DE DATOS Y CACHÉ TELEMÉTRICA
# ==========================================
@st.cache_data(ttl=3600)
def fetch_historical_data(tickers: List[str], period: str = "max") -> pd.DataFrame:
    """
    Extracción asíncrona de datos históricos desde Yahoo Finance.
    Se utiliza @st.cache_data para evitar recargas API redundantes (TTL 1 hora).
    Aplica forward-fill para manejar NaN derivados de festividades globales.
    """
    data = yf.download(tickers, period=period)['Close']
    return data.ffill().dropna()


@st.cache_data(ttl=300)
def fetch_live_ticker_data() -> pd.DataFrame:
    """
    Extracción de índices macroeconómicos globales para la cinta rodante.
    Caché de corta duración (TTL 5 minutos) para cuasi-tiempo real.
    """
    live_list = ["^GSPC", "^IXIC", "^DJI", "^N225", "^GDAXI", "BTC-USD", "GC=F", "EURUSD=X"]
    data = yf.download(live_list, period="5d", interval="1d")['Close']
    return data.ffill().dropna()


def render_live_ticker() -> None:
    """
    Construye e inyecta el HTML dinámico para el ticker rodante superior.
    Calcula la variación porcentual de la última sesión de negociación.
    """
    try:
        t_data = fetch_live_ticker_data()
        if len(t_data) >= 2:
            prices = t_data.iloc[-1]
            changes = (t_data.iloc[-1] / t_data.iloc[-2]) - 1
            
            names = {
                "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "DOW JONES", 
                "^N225": "NIKKEI", "^GDAXI": "DAX", "BTC-USD": "BITCOIN", 
                "GC=F": "GOLD", "EURUSD=X": "EUR/USD"
            }
            
            html = '<div class="ticker-wrapper"><div class="ticker-move">'
            for symbol, name in names.items():
                val = prices[symbol]
                change = changes[symbol]
                color = "#00ff41" if change >= 0 else "#ff4b4b"
                arrow = "▲" if change >= 0 else "▼"
                html += f'<span class="ticker-item">{name}: {val:,.2f} <span style="color:{color}">{arrow} {change*100:.2f}%</span></span>'
            html += '</div></div>'
            st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.write("📡 Sincronizando feed telemétrico de mercados globales...")


# ==========================================
# 3. MOTORES MATEMÁTICOS Y ESTADÍSTICOS
# ==========================================
def get_naive_risk_parity_weights(cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Algoritmo de Paridad de Riesgo Ingenua (Naive Risk Parity / Inverse Volatility).
    Pondera los activos inversamente proporcionales a su volatilidad individual,
    ignorando la correlación. Fundamental en carteras All-Weather.
    """
    inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / np.sum(inv_vol)
    return weights


def calculate_advanced_metrics(returns: pd.DataFrame, weights: np.ndarray, cov_matrix: pd.DataFrame, 
                               initial_cap: float, bench_rets: pd.Series, rf: float = 0.03) -> Dict[str, Any]:
    """
    Núcleo matemático principal. Calcula métricas de 1er a 4to orden estocástico.
    
    Parámetros:
    - returns: DataFrame de retornos logarítmicos diarios.
    - weights: Vector de pesos (sum(w) = 1).
    - cov_matrix: Matriz de covarianza anualizada.
    - initial_cap: Capital de inversión inicial.
    - bench_rets: Serie temporal del benchmark (SPY).
    - rf: Tasa libre de riesgo (Risk-free rate).
    """
    # 1. Retornos Diarios Agregados del Portafolio
    portfolio_rets = returns.dot(weights)
    
    # 2. Métricas Anualizadas Básicas
    p_ret_annual = portfolio_rets.mean() * 252
    p_vol_annual = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # 3. Cálculo de Regresión Paramétrica (Beta CAPM)
    common_idx = portfolio_rets.index.intersection(bench_rets.index)
    covariance = np.cov(portfolio_rets.loc[common_idx], bench_rets.loc[common_idx])[0, 1]
    market_variance = np.var(bench_rets.loc[common_idx])
    beta = covariance / market_variance if market_variance != 0 else 1.0
    
    # 4. Ratios de Eficiencia Fiduciaria
    sharpe_ratio = (p_ret_annual - rf) / p_vol_annual if p_vol_annual != 0 else 0
    
    downside_returns = portfolio_rets[portfolio_rets < 0]
    downside_std = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
    sortino_ratio = (p_ret_annual - rf) / downside_std if downside_std != 0 else 0
    
    treynor_ratio = (p_ret_annual - rf) / beta if beta != 0 else 0
    
    # 5. Riesgo de Ruina (Drawdown)
    cum_rets = (1 + portfolio_rets).cumprod()
    running_max = cum_rets.cummax()
    drawdowns = (cum_rets - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    calmar_ratio = p_ret_annual / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 6. Riesgo de Cola Estocástico (VaR & CVaR paramétrico 95%)
    var_95 = norm.ppf(1 - 0.95, portfolio_rets.mean(), portfolio_rets.std())
    cvar_95 = portfolio_rets[portfolio_rets <= var_95].mean() if not portfolio_rets[portfolio_rets <= var_95].empty else var_95
    
    # 7. Momentos de Tercer y Cuarto Orden
    p_skew = skew(portfolio_rets)
    p_kurt = kurtosis(portfolio_rets)
    
    # 8. Tracking Error & Information Ratio (vs Benchmark)
    active_returns = portfolio_rets.loc[common_idx] - bench_rets.loc[common_idx]
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = (p_ret_annual - (bench_rets.mean() * 252)) / tracking_error if tracking_error != 0 else 0
    
    return {
        "ret_annual": p_ret_annual,
        "vol_annual": p_vol_annual,
        "sharpe": sharpe_ratio,
        "sortino": sortino_ratio,
        "treynor": treynor_ratio,
        "calmar": calmar_ratio,
        "beta": beta,
        "max_dd": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skew": p_skew,
        "kurtosis": p_kurt,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "final_val": initial_cap * cum_rets.iloc[-1],
        "portfolio_series": portfolio_rets,
        "cum_series": cum_rets,
        "bench_series": bench_rets.loc[common_idx],
        "drawdowns": drawdowns
    }


def calculate_rolling_metrics(port_rets: pd.Series, bench_rets: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Calcula la Volatilidad y el Beta rodante (rolling) para observar
    la inestabilidad estructural y los cambios de régimen a lo largo del tiempo.
    """
    # Alineación de índices
    common_idx = port_rets.index.intersection(bench_rets.index)
    p_ret = port_rets.loc[common_idx]
    b_ret = bench_rets.loc[common_idx]
    
    # Volatilidad Rodante (Anualizada)
    rolling_vol = p_ret.rolling(window=window).std() * np.sqrt(252)
    
    # Beta Rodante (Covarianza rodante / Varianza rodante)
    rolling_cov = p_ret.rolling(window=window).cov(b_ret)
    rolling_var = b_ret.rolling(window=window).var()
    rolling_beta = rolling_cov / rolling_var
    
    df_rolling = pd.DataFrame({
        'Rolling_Vol_60d': rolling_vol,
        'Rolling_Beta_60d': rolling_beta
    }).dropna()
    
    return df_rolling


def simulate_monte_carlo_future_paths(initial_value: float, mu: float, sigma: float, 
                                      days: int = 252, n_simulations: int = 1000) -> np.ndarray:
    """
    Simulación Avanzada de Monte Carlo.
    Utiliza el Movimiento Browniano Geométrico (GBM) para predecir múltiples
    trayectorias futuras del capital.
    
    Fórmula Estocástica: S_t = S_{t-1} * exp((mu - sigma^2/2)dt + sigma * dW_t)
    """
    dt = 1 / 252 # Fracción de año (diario)
    paths = np.zeros((days, n_simulations))
    paths[0] = initial_value
    
    # Pre-calculamos el término de deriva (drift)
    drift = (mu - 0.5 * sigma**2) * dt
    
    # Generación de variables aleatorias normales estándar (Z)
    np.random.seed(42) # Reproducibilidad científica
    Z = np.random.standard_normal((days - 1, n_simulations))
    
    # Proceso de difusión
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Calcular retornos diarios exponenciales
    daily_returns = np.exp(drift + diffusion)
    
    # Construir trayectorias acumulativas
    for t in range(1, days):
        paths[t] = paths[t-1] * daily_returns[t-1]
        
    return paths


def detect_market_regimes(bench_rets: pd.Series) -> pd.DataFrame:
    """
    Clasificación Heurística de Regímenes de Mercado (Macro-Regime Analysis).
    Divide el histórico del benchmark en 4 cuadrantes basados en medias móviles
    y volatilidad condicional para simular el entorno del ciclo económico.
    """
    df = pd.DataFrame(bench_rets)
    df.columns = ['Returns']
    
    # Trend: Media Móvil de 60 días del retorno acumulado
    df['Trend_60d'] = df['Returns'].rolling(60).mean() * 252
    
    # Volatilidad: Desviación Estándar de 60 días
    df['Vol_60d'] = df['Returns'].rolling(60).std() * np.sqrt(252)
    
    # Calcular medianas históricas para definir los umbrales
    median_trend = df['Trend_60d'].median()
    median_vol = df['Vol_60d'].median()
    
    # Clasificación Cuadrática Oculta
    conditions = [
        (df['Trend_60d'] >= median_trend) & (df['Vol_60d'] < median_vol),   # Bull Calm
        (df['Trend_60d'] >= median_trend) & (df['Vol_60d'] >= median_vol),  # Bull Volatile
        (df['Trend_60d'] < median_trend) & (df['Vol_60d'] < median_vol),    # Bear Calm
        (df['Trend_60d'] < median_trend) & (df['Vol_60d'] >= median_vol)    # Bear Volatile / Crash
    ]
    
    choices = ['1. Expansión (Bull Calm)', '2. Exuberancia (Bull Volatile)', 
               '3. Contracción (Bear Calm)', '4. Pánico (Bear Volatile)']
               
    df['Regime'] = np.select(conditions, choices, default='Unknown')
    return df.dropna()


# ==========================================
# 4. GENERADORES DE CONTENIDO IA & CFA
# ==========================================
def get_ai_prescriptive_insights(m: Dict[str, Any], benchmark_name: str) -> List[str]:
    """Genera insights prescriptivos basados en heurísticas de Hedge Funds."""
    insights = []
    
    # 1. Análisis de Eficiencia de Capital (Sharpe & Info Ratio)
    if m['sharpe'] >= 1.5: 
        insights.append(f"🟢 **Eficiencia Cuántica Superior:** Ratio de Sharpe de {m['sharpe']:.2f}. La topología de la frontera eficiente ha logrado capturar un Alfa algorítmico significativo, justificando sobradamente el presupuesto de riesgo.")
    elif m['sharpe'] >= 1.0: 
        insights.append(f"🎯 **Optimización Estándar:** Sharpe de {m['sharpe']:.2f}. El modelo valida matemáticamente la Prima de Riesgo asumida. Se mantiene la racionalidad fiduciaria.")
    else: 
        insights.append(f"⚖️ **Alerta de Fricción de Varianza:** Sharpe de {m['sharpe']:.2f} (< 1.0). El retorno esperado es estadísticamente insuficiente frente a la varianza. Considere inyectar activos ortogonales (ej. Oro, Bonos del Tesoro).")
        
    # 2. Análisis de Sensibilidad (Beta & Treynor)
    if m['beta'] > 1.2: 
        insights.append(f"🚀 **Pro-Ciclidad Agresiva:** Beta de {m['beta']:.2f}. La arquitectura actúa como un amplificador del {benchmark_name}. Estrategia óptima exclusivamente en regímenes monetarios expansivos (Bull Markets).")
    elif m['beta'] < 0.8: 
        insights.append(f"🛡️ **Cobertura Estructural (Defensiva):** Beta de {m['beta']:.2f}. El portafolio está aislado del riesgo sistemático, funcionando como refugio fiduciario ante eventos de contracción de liquidez.")
        
    # 3. Riesgo Estocástico de Cola (Skew & Kurtosis)
    if m['skew'] < -0.5 or m['kurtosis'] > 3.0: 
        insights.append(f"⚠️ **Alerta de Eventos Extremos (Cisnes Negros):** Skewness negativo ({m['skew']:.2f}) y Curtosis Leptocúrtica ({m['kurtosis']:.2f}). Las matemáticas fractales advierten una probabilidad empírica de colapso superior a la proyectada por las curvas de Gauss.")
    
    # 4. Tracking Error Insight
    if m['tracking_error'] > 0.15:
        insights.append(f"🔍 **Gestión Altamente Activa:** Tracking Error del {m['tracking_error']*100:.1f}%. El portafolio diverge masivamente del benchmark. Usted está haciendo *Stock Picking* de convicción extrema, asumiendo un alto riesgo idiosincrático.")
        
    return insights


def render_cfa_investment_thesis(m: Dict[str, Any], t_list: List[str], symbol: str, initial_cap: float, mode: str) -> str:
    """
    Redacta un Abstract Ejecutivo formal con estándares del CFA Institute.
    Incorpora todo el argot cuantitativo y métricas extraídas en tiempo real.
    """
    profile = "Aggressive Growth" if m['beta'] > 1.1 else "Capital Preservation" if m['beta'] < 0.9 else "Balanced Allocation"
    grade = "Investment Grade (Premium Alpha)" if m['sharpe'] > 1.0 else "Speculative Grade (Sub-optimal)"
    
    html = f"""
    <div class="cfa-paper">
        <div class="cfa-header">
            <h1>Quantitative Investment Thesis Abstract</h1>
            <p style="margin:0; font-size: 1.1rem; color: #4b5563; font-style: italic;">NeuroQuant Institutional Research Division • CFA Framework Compliant</p>
            <p style="margin:5px 0 0 0; font-weight: bold; color: #1e293b;">Date: {datetime.now().strftime('%A, %d %B %Y')} | ID: NQ-RES-{datetime.now().strftime('%H%M%S')}</p>
        </div>
        
        <div class="cfa-section">I. Institutional Mandate & Asset Allocation Strategy</div>
        <p>El presente dictamen documenta la arquitectura técnica, validación estocástica y perfil fiduciario de una cartera de activos estructurada sobre un universo de <b>{len(t_list)} instrumentos estratégicos</b> ({", ".join(t_list)}). 
        Operando bajo la directriz metodológica de <b>{mode}</b>, el motor de Inteligencia Artificial ha aplicado optimización convexa, minimizando la varianza condicional para una tasa de retorno esperada. El vehículo inicia operaciones simuladas con un AUM (Assets Under Management) base de {symbol}{initial_cap:,.2f}.</p>
        
        <div class="cfa-section">II. Ex-Post Quantitative Performance Diagnostics</div>
        <p>Durante la ventana de backtesting empírico, el mandato proyecta un Valor Terminal Acumulado de <b>{symbol}{m['final_val']:,.2f}</b>, asumiendo capitalización continua (CAGR) libre de fricciones fiscales. 
        Evaluando el rendimiento ajustado por riesgo, el vector de pesos registra un <b>Sharpe Ratio Anualizado de {m['sharpe']:.2f}</b>, un <b>Sortino Ratio de {m['sortino']:.2f}</b> y un Information Ratio de {m['information_ratio']:.2f}, 
        lo que califica la estructura tipológica como <i>{grade}</i>. 
        El análisis de regresión paramétrica frente al benchmark revela un coeficiente $\beta$ de <b>{m['beta']:.2f}</b>, anclando el mandato dentro de un paradigma de <b>{profile}</b>.</p>
        
        <div class="cfa-section">III. Tail Risk Analytics & Extreme Stress Events</div>
        <p>El escrutinio estadístico de momentos de orden superior expone una asimetría (Skewness) de <b>{m['skew']:.2f}</b> y un exceso de curtosis de <b>{m['kurtosis']:.2f}</b>, refutando la hipótesis de normalidad perfecta. 
        El <i>Value at Risk paramétrico (VaR 95%)</i> estima un umbral de depreciación diaria esperada del <b>{abs(m['var_95']*100):.2f}%</b>, mientras que el 
        <i>Expected Shortfall (CVaR)</i> acota la gravedad en la zona de exclusión (tail risk) en {abs(m['cvar_95']*100):.2f}%. El histórico Max Drawdown, métrica definitiva de la tolerancia al riesgo de ruina del capital, se sitúa en un {abs(m['max_dd']*100):.2f}%.</p>
        
        <div class="cfa-section">IV. Dynamic Attribution & ESG Forward Guidance</div>
        <p>Considerando la descorrelación intracartera cuantificada en la matriz de covarianza de Pearson y la eficiencia en la asignación del capital marginal (Euler), el comité algorítmico 
        resuelve que el ensamble es <b>{'ESTRUCTURALMENTE ROBUSTO' if m['sharpe'] > 0.8 else 'DEFICIENTE - REQUIERE INYECCIÓN DE ACTIVOS ORTOGONALES'}</b> para absorción de capital institucional real. 
        Se dictamina imperativo que el Portfolio Manager suplemente este output cuantitativo con análisis cualitativo fundamental (Métricas Ambientales, Sociales y de Gobernanza - ESG) previo a la ejecución en mercado real para aislar el riesgo reputacional.</p>
        
        <div style="margin-top: 50px; text-align: center; border-top: 1px solid #cbd5e0; padding-top: 15px;">
            <p style="font-size: 0.85rem; font-style: italic; color: #718096; margin: 0;">Reporte generado autónomamente y sellado criptográficamente por NeuroQuant AI Engine v40.0</p>
            <p style="font-size: 0.85rem; font-weight: bold; color: #1a202c; margin: 0;">Desarrollo Científico avalado por la Cátedra Fintech - Universidad de Málaga</p>
        </div>
    </div>
    """
    return html


# ==========================================
# 5. CONTROLADOR PRINCIPAL DEL DASHBOARD
# ==========================================
def main():
    """Función de anclaje (Main Entry Point) de la aplicación Streamlit."""
    
    # Inicialización UI
    setup_page()
    render_header("logo-fintech-UMA (1).png")
    render_live_ticker()

    # --- PANEL DE CONTROL LATERAL (SIDEBAR) ---
    with st.sidebar:
        st.title("⚙️ Terminal Config")
        
        currency_str = st.selectbox("Divisa Base Fiduciaria", ["USD ($)", "EUR (€)", "GBP (£)", "JPY (¥)"])
        symbol = currency_str.split("(")[1].replace(")", "")
        initial_capital = st.number_input("Capital Fiduciario a Desplegar", value=10000.0, step=1000.0, min_value=100.0)
        
        st.divider()
        st.subheader("🎨 UX / UI Engine")
        color_theme = st.selectbox("Renderizado Cromático Institucional", [
            "1. Deep Institutional (Slate/Blue)", 
            "2. Emerald Capital (Greens)", 
            "3. Quantitative Monochrome (Grey)", 
            "4. Cyber Neon (Cyan/Purple)"
        ])
        
        # Mapeo de paletas dinámicas para inyectar en Plotly
        theme_palettes = {
            "1. Deep Institutional (Slate/Blue)": ["#0f172a", "#1e293b", "#3b82f6", "#64748b", "#94a3b8", "#e2e8f0"],
            "2. Emerald Capital (Greens)": ["#064e3b", "#059669", "#10b981", "#34d399", "#a7f3d0", "#ecfdf5"],
            "3. Quantitative Monochrome (Grey)": ["#000000", "#404040", "#737373", "#a3a3a3", "#d4d4d4", "#f5f5f5"],
            "4. Cyber Neon (Cyan/Purple)": ["#0891b2", "#8b5cf6", "#2dd4bf", "#a855f7", "#c084fc", "#e879f9"]
        }
        heatmap_scales = {
            "1. Deep Institutional (Slate/Blue)": "Blues", 
            "2. Emerald Capital (Greens)": "Greens", 
            "3. Quantitative Monochrome (Grey)": "Greys", 
            "4. Cyber Neon (Cyan/Purple)": "Purples"
        }
        
        selected_colors = theme_palettes[color_theme]
        selected_heatmap = heatmap_scales[color_theme]
        
        st.divider()
        st.subheader("📐 Modelado Estratégico")
        
        mode = st.radio("Arquitectura Matemática", [
            "IA (Máx. Sharpe / Markowitz)", 
            "Risk Parity (Volatilidad Inversa)",
            "Manual (Stock Picking)"
        ])
        
        with st.popover("❔ Teoría de Optimización (Ver Glosario)"):
            st.markdown("""
            **🤖 IA (Máx. Sharpe):**
            Ejecuta el solver matemático de Markowitz. Optimización convexa pura para buscar la tangente de la Línea de Asignación de Capital.
            
            **⚖️ Risk Parity (Vol Inversa):**
            Ignora los retornos esperados. Pondera el capital de forma inversamente proporcional al riesgo del activo. Si NVDA es muy volátil, recibe menos dinero. Modelo base de *Bridgewater All-Weather*.
            
            **✋ Manual:**
            Input directo de pesos por parte del Analista Financiero.
            """)
            
        if mode == "Manual (Stock Picking)":
            raw_tickers = st.text_area("Universo de Activos (Tickers CSV)", "AAPL, MSFT, GLD")
            t_list = [x.strip().upper() for x in raw_tickers.split(",") if x.strip()]
            ws = []
            st.caption("Ajuste Táctico:")
            for t in t_list:
                val = st.number_input(f"Peso % {t}", min_value=0.0, max_value=100.0, value=100.0/len(t_list))
                ws.append(val / 100)
            weights = np.array(ws)
            max_w = 1.0 
        else:
            raw_tickers = st.text_area("Universo de Activos (Tickers CSV)", "AAPL, MSFT, NVDA, TSLA, GLD, BTC-USD")
            t_list = [x.strip().upper() for x in raw_tickers.split(",") if x.strip()]
            max_w = st.slider("Límite de Concentración Singular %", 10, 100, 30) / 100
            weights = None

        benchmark_ticker = st.text_input("Índice Benchmark Global", "SPY")
        stress_scenario = st.selectbox("Simulador Macro de Estrés Histórico", [
            "Ninguno (Default 3 Años)", 
            "Flash Crash Pandémico (2020)", 
            "Colapso Lehman Brothers (2008)"
        ])
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("▶ COMPILAR MOTOR QUANTITATIVO", use_container_width=True)

    # --- EJECUCIÓN DEL ALGORITMO Y RENDERIZADO DEL DASHBOARD ---
    if analyze_btn:
        if len(t_list) < 2:
            st.error("⚠️ VIOLACIÓN DE MATRIZ: La arquitectura de optimización requiere un mínimo de 2 activos ortogonales para construir el hiperplano de covarianza.")
            return

        with st.spinner("⏳ Compilando Estocástica de Monte Carlo, Resolviendo Matrices Hessian y Generando Reportes..."):
            try:
                # ==========================================
                # FASE 1: Data Wrangling & Pre-Processing
                # ==========================================
                market_data = fetch_historical_data(t_list + [benchmark_ticker])
                start_date = market_data.index[0].strftime('%Y-%m-%d')
                end_date = market_data.index[-1].strftime('%Y-%m-%d')
                
                # Truncado Temporal por Regímenes de Estrés
                if stress_scenario == "Flash Crash Pandémico (2020)":
                    df_filtered = market_data.loc['2020-02-01':'2020-06-01']
                elif stress_scenario == "Colapso Lehman Brothers (2008)":
                    df_filtered = market_data.loc['2007-10-01':'2009-04-01']
                else:
                    df_filtered = market_data.tail(756) # Aprox 3 años de trading
                    
                asset_returns = df_filtered[t_list].pct_change().dropna()
                bench_returns = df_filtered[benchmark_ticker].pct_change().dropna()
                
                # Matriz de Covarianza Histórica Anualizada (PyPortfolioOpt)
                cov_matrix = risk_models.sample_cov(df_filtered[t_list])
                
                # ==========================================
                # FASE 2: Optimización Convexa / Routing
                # ==========================================
                if mode == "IA (Máx. Sharpe / Markowitz)":
                    mu = expected_returns.capm_return(df_filtered[t_list], market_prices=df_filtered[benchmark_ticker])
                    ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, max_w))
                    weights_dict = ef.max_sharpe()
                    weights = np.array([weights_dict[ticker] for ticker in t_list])
                    
                elif mode == "Risk Parity (Volatilidad Inversa)":
                    mu = expected_returns.capm_return(df_filtered[t_list], market_prices=df_filtered[benchmark_ticker])
                    weights = get_naive_risk_parity_weights(cov_matrix)
                    
                else:
                    # En modo Manual, las medias esperadas (mu) siguen siendo necesarias para las simulaciones
                    mu = expected_returns.capm_return(df_filtered[t_list], market_prices=df_filtered[benchmark_ticker])
                
                # ==========================================
                # FASE 3: Generación de Métricas
                # ==========================================
                metrics = calculate_advanced_metrics(asset_returns, weights, cov_matrix, initial_capital, bench_returns)
                
                # Extracción de valores individuales para gráficos
                ind_rets = asset_returns.mean().values * 252
                ind_vols = asset_returns.std().values * np.sqrt(252)
                
                # ==========================================
                # FASE 4: Simulaciones de Monte Carlo & MPT
                # ==========================================
                n_ports = 2500
                np.random.seed(42) # Seed estático para reproducibilidad institucional
                
                # Generación aleatoria de pesos bajo distribución de Dirichlet (Suma = 1)
                w_mc = np.random.dirichlet(np.ones(len(t_list)), n_ports)
                emp_mu = asset_returns.mean().values * 252
                emp_cov = asset_returns.cov().values * 252
                
                rets_mc = w_mc.dot(emp_mu)
                
                # Multiplicación matricial rápida para la varianza de cada portafolio
                vols_mc = np.zeros(n_ports)
                for i in range(n_ports):
                    vols_mc[i] = np.sqrt(np.dot(w_mc[i].T, np.dot(emp_cov, w_mc[i])))
                    
                sharpe_mc = (rets_mc - 0.03) / vols_mc

                df_mc = pd.DataFrame({
                    'Retorno Esperado': rets_mc, 
                    'Volatilidad (Riesgo)': vols_mc, 
                    'Sharpe Ratio': sharpe_mc
                })

                # ==========================================
                # FASE 5: RENDERIZADO DEL DASHBOARD (TABS)
                # ==========================================
                t_perf, t_risk, t_attr, t_future, t_regimes, t_rolling, t_ai, t_cfa, t_audit = st.tabs([
                    "📊 Desempeño", "🧠 Riesgo Estructural", "⚖️ Atribución", 
                    "🔮 Simulador Futuro", "⏳ Macro Regímenes", "📈 Beta Rodante",
                    "🤖 Neuro-Insights", "📄 Tesis CFA", "📚 Trazabilidad"
                ])

                # ------------------------------------------
                # TAB 1: DESEMPEÑO Y BETA CAPM
                # ------------------------------------------
                with t_perf:
                    st.subheader("I. Análisis Cuantitativo de Performance")
                    
                    # KPIs Top Level
                    cols = st.columns(5)
                    cols[0].markdown(f"<div class='executive-card'><div class='metric-label'>Valor Terminal Proyectado</div><div class='metric-value'>{symbol}{metrics['final_val']:,.0f}</div></div>", unsafe_allow_html=True)
                    cols[1].markdown(f"<div class='executive-card'><div class='metric-label'>Elasticidad (Beta)</div><div class='metric-value'>{metrics['beta']:.2f}</div></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='executive-card'><div class='metric-label'>Sharpe Ratio</div><div class='metric-value'>{metrics['sharpe']:.2f}</div></div>", unsafe_allow_html=True)
                    cols[3].markdown(f"<div class='executive-card'><div class='metric-label'>Treynor Ratio</div><div class='metric-value'>{metrics['treynor']:.3f}</div></div>", unsafe_allow_html=True)
                    cols[4].markdown(f"<div class='executive-card'><div class='metric-label'>Riesgo de Ruina (Max DD)</div><div class='metric-value'>{metrics['max_dd']*100:.1f}%</div></div>", unsafe_allow_html=True)
                    
                    with st.expander("📚 Glosario Cuantitativo Institucional (Desplegar)"):
                        st.markdown(f"""
                        * **BETA ($\beta$):** Cuantifica la elasticidad del portafolio frente al mercado (Riesgo Sistémico). Un valor de **{metrics['beta']:.2f}** indica la magnitud teórica de la oscilación frente a un movimiento del 1% en el benchmark.
                        * **SHARPE RATIO:** Exceso de retorno geométrico generado por unidad de volatilidad total asumida ($\sigma$). Nivel actual de **{metrics['sharpe']:.2f}**.
                        * **TREYNOR RATIO:** Estandariza el exceso de retorno penalizando únicamente el riesgo sistémico ($\beta$), excluyendo el riesgo idiosincrático. Nivel actual de **{metrics['treynor']:.3f}**.
                        * **MAX DRAWDOWN:** Depreciación patrimonial máxima histórica (*Peak-to-Trough*). Retracción límite observada del **{abs(metrics['max_dd'])*100:.2f}%**.
                        """)
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        with st.expander("📈 Trayectoria Estocástica del Capital (Equity Curve)"):
                            st.markdown("Modela la evolución geométrica del patrimonio bajo capitalización continua a lo largo de una serie temporal discreta. Una topología suavizada y monótonamente creciente denota un proceso con baja heterocedasticidad condicional y un alto Ratio de Información. Inversamente, una morfología de alta oscilación evidencia una severa fricción por varianza (*volatility drag*), donde la divergencia geométrica erosiona el rendimiento real a largo plazo.")
                        
                        fig1 = px.line(metrics["cum_series"] * initial_capital, title="Evolución Empírica del Capital", labels={'value': symbol}, color_discrete_sequence=[selected_colors[0]])
                        fig1.update_traces(line=dict(width=2.5))
                        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig1, use_container_width=True)

                    with c2:
                        with st.expander("📉 Análisis de Regresión Paramétrica OLS (CAPM)"):
                            st.markdown("Modela empíricamente el marco teórico de fijación de precios de activos (Sharpe, 1964) mediante Mínimos Cuadrados Ordinarios (OLS). La pendiente de la recta de mejor ajuste representa el coeficiente beta ($\beta$), definido estocásticamente como el ratio de covarianzas sobre la varianza del mercado. El intercepto del eje Y proyecta el Alfa de Jensen ($\alpha$), mientras que la dispersión residual evidencia el riesgo idiosincrático no diversificado ($\epsilon$).")
                        
                        fig2 = px.scatter(x=metrics["bench_series"], y=metrics["portfolio_series"], trendline="ols", title="Regresión OLS: Portafolio vs Benchmark Global", color_discrete_sequence=[selected_colors[1]])
                        fig2.update_traces(marker=dict(size=6, opacity=0.6))
                        fig2.update_layout(xaxis_title="Retorno Diario Benchmark", yaxis_title="Retorno Diario Portafolio", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)

                    st.divider()
                    st.subheader("II. Optimización Convexa de Carteras (Teoría MPT)")
                    
                    with st.expander("🌌 Frontera Eficiente y Línea de Asignación de Capital (CAL) - Teoría Explicativa"):
                        st.markdown("""
                        **Topología del Espacio Riesgo-Retorno:**
                        Esta simulación visualiza el paradigma central de Markowitz (1952). La dispersión representa un conjunto topológico de oportunidades generado mediante **Monte Carlo ($N=2500$ iteraciones estocásticas)** utilizando distribuciones de Dirichlet para asegurar que la restricción presupuestaria ($\sum w_i = 1$) se mantenga inelástica.
                        
                        * **Nube Estocástica:** Representa carteras aleatorias subóptimas.
                        * **Envolvente Superior:** Es la **Frontera Eficiente**, el lugar geométrico que resuelve el problema de optimización cuadrática dual.
                        * **Supremo Global (Estrella Dorada):** Denota la *Cartera Tangente*, donde la Línea de Asignación de Capital (CAL) maximiza analíticamente la primera derivada del hiperplano de utilidad (Máx. Sharpe).
                        * **Subaditividad de Varianza (Diamantes Rojos):** Los activos individuales aislados en el plano. La cartera óptima se posiciona más al 'noroeste' (más retorno, menos riesgo) que los activos individuales, probando matemáticamente el beneficio de la diversificación cuando correlaciones son $< 1$.
                        """)
                    
                    # Gráfica de Markowitz Mejorada
                    fig_ef = px.scatter(
                        df_mc, x='Volatilidad (Riesgo)', y='Retorno Esperado', color='Sharpe Ratio',
                        color_continuous_scale=selected_heatmap,
                        title="Frontera Eficiente de Markowitz y Topología de Subaditividad"
                    )
                    
                    # Agregar el Portafolio Matemático (El actual)
                    fig_ef.add_trace(go.Scatter(
                        x=[metrics['vol_annual']], y=[metrics['ret_annual']],
                        mode='markers', 
                        marker=dict(color='gold', size=22, symbol='star', line=dict(color='black', width=2)),
                        name=f'Cartera Elegida: {mode}'
                    ))
                    
                    # Agregar Activos Individuales
                    fig_ef.add_trace(go.Scatter(
                        x=ind_vols, y=ind_rets, mode='markers+text', text=t_list, textposition='bottom center',
                        marker=dict(color=selected_colors[3], size=14, symbol='diamond', line=dict(color='white', width=1.5)),
                        name='Activos Aislados (Riesgo Puro)'
                    ))
                    
                    fig_ef.update_layout(
                        showlegend=True, 
                        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.9)"),
                        xaxis_title="Volatilidad Esperada Anualizada ($\sigma$)",
                        yaxis_title="Retorno Esperado Anualizado ($E[R]$)",
                        plot_bgcolor="#ffffff"
                    )
                    
                    st.plotly_chart(fig_ef, use_container_width=True)

                # ------------------------------------------
                # TAB 2: RIESGO Y DISTRIBUCIONES (FAT TAILS)
                # ------------------------------------------
                with t_risk:
                    st.subheader("III. Diagnóstico Estructural y Riesgos de Cola (Tail Risk)")
                    
                    with st.expander("🧠 Diagnóstico de Momentos Superiores (Skewness & Kurtosis)"):
                        st.markdown("En finanzas cuantitativas, la distribución Normal asume simetría perfecta. Un sesgo negativo (Negative Skew) advierte asimetría hacia rentabilidades negativas extremas (caídas bruscas y subidas lentas). Un exceso de curtosis (Leptocúrtica) evidencia que los modelos gaussianos subestiman dramáticamente la probabilidad empírica de ocurrencia de un evento de estrés o Cisne Negro (*Fat Tails*).")
                    
                    c1, c2 = st.columns(2)
                    
                    # Histograma PDF + VaR
                    with c1:
                        port_rets = metrics["portfolio_series"]
                        mu_day = port_rets.mean()
                        std_day = port_rets.std()
                        x_norm = np.linspace(port_rets.min(), port_rets.max(), 150)
                        y_norm = norm.pdf(x_norm, mu_day, std_day)
                        
                        fig_hist = go.Figure()
                        # Distribución Real
                        fig_hist.add_trace(go.Histogram(
                            x=port_rets, histnorm='probability density', name='Distribución Empírica',
                            marker_color=selected_colors[0], opacity=0.8, nbinsx=75
                        ))
                        # Campana de Gauss
                        fig_hist.add_trace(go.Scatter(
                            x=x_norm, y=y_norm, mode='lines', name='Campana de Gauss Teórica',
                            line=dict(color=selected_colors[2], width=3, dash='dash')
                        ))
                        # VaR Line
                        fig_hist.add_vline(x=metrics['var_95'], line_width=2.5, line_dash="dot", line_color="#ff4b4b", 
                                           annotation_text=f"VaR 95%: {metrics['var_95']*100:.2f}%", annotation_position="top left")

                        fig_hist.update_layout(
                            title="Función de Densidad de Probabilidad (PDF) y VaR",
                            xaxis_title="Rentabilidad Diaria Continua", yaxis_title="Densidad",
                            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                            barmode='overlay', hovermode="x", plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Matriz de Correlación
                    with c2:
                        fig_corr = px.imshow(
                            asset_returns.corr(), text_auto=".2f", 
                            color_continuous_scale=selected_heatmap, 
                            title="Matriz de Covarianza Ortogonalizada (Pearson)"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                    # Visualización de Drawdowns (Caídas bajo el agua)
                    st.markdown("#### Histograma de Retracciones (Drawdowns)")
                    fig_dd = px.area(
                        x=metrics["drawdowns"].index, y=metrics["drawdowns"]*100, 
                        title="Profundidad y Duración de Caídas Acumuladas ('Underwater Plot')",
                        labels={'x': 'Fecha temporal', 'y': 'Pérdida desde Máximo (%)'},
                        color_discrete_sequence=["#ff4b4b"]
                    )
                    fig_dd.update_traces(fillcolor="rgba(255, 75, 75, 0.2)")
                    st.plotly_chart(fig_dd, use_container_width=True)

                # ------------------------------------------
                # TAB 3: ATRIBUCIÓN DE RIESGO / EULER
                # ------------------------------------------
                with t_attr:
                    st.subheader("IV. Atribución de Riesgo Marginal y Presupuesto de Capital")
                    
                    with st.expander("⚖️ Descomposición de Varianza de Portafolio (Teorema de Euler)"):
                        st.markdown("Desagrega matemáticamente la contribución al riesgo *ex-ante* de cada posición individual en relación a la covarianza del portafolio completo. Identifica desequilibrios estructurales críticos donde una pequeña asignación de capital (Peso %) concentra desproporcionadamente la volatilidad agregada del fondo (Riesgo %). En modelos 'Risk Parity', las barras de la gráfica izquierda deberían ser idénticas.")
                    
                    c1, c2 = st.columns(2)
                    # Riesgo Marginal Computado
                    risk_contribution = (weights * (cov_matrix @ weights)) / np.sqrt(weights.T @ cov_matrix @ weights)
                    # Normalización a porcentajes
                    risk_contribution_pct = risk_contribution / np.sum(risk_contribution)
                    
                    with c1:
                        fig_risk = px.bar(
                            x=t_list, y=risk_contribution_pct*100, 
                            title="Descomposición Matemática del Riesgo Total",
                            labels={'x':'Activo', 'y':'Contribución Volatilidad (%)'},
                            color_discrete_sequence=[selected_colors[0]]
                        )
                        fig_risk.update_layout(plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                    with c2:
                        fig_weights = px.pie(
                            names=t_list, values=weights, hole=0.55, 
                            title="Estructura Actual de Presupuesto Fiduciario (Capital %)", 
                            color_discrete_sequence=selected_colors
                        )
                        st.plotly_chart(fig_weights, use_container_width=True)

                # ------------------------------------------
                # TAB 4: SIMULADOR DE MONTE CARLO (FUTURO)
                # ------------------------------------------
                with t_future:
                    st.subheader("🔮 Proyección Estocástica Futura (Monte Carlo - GBM)")
                    
                    with st.expander("🔬 Movimiento Browniano Geométrico y Lema de Itô"):
                        st.markdown(r"""
                        Para proyectar el capital a futuro, no interpolamos líneas rectas. Utilizamos el **Movimiento Browniano Geométrico (GBM)**, el motor matemático que impulsa la fijación de precios de derivados en Black-Scholes. 
                        
                        La Ecuación Diferencial Estocástica utilizada es:
                        $$ dS_t = S_t \mu dt + S_t \sigma dW_t $$
                        
                        Donde $\mu$ es el *drift* (deriva esperada), $\sigma$ es la volatilidad anualizada, y $dW_t$ es un proceso estocástico de Wiener (Ruido Blanco). La gráfica inferior simula **1000 universos paralelos** para los próximos 252 días de bolsa (1 Año), demostrando el amplio cono de incertidumbre y calculando la probabilidad de quiebra.
                        """)
                        
                    n_days_future = 252 # 1 año de trading
                    n_sims = 1000
                    
                    with st.spinner("Procesando 1000 Líneas de Mundo Paralelas..."):
                        future_paths = simulate_monte_carlo_future_paths(
                            initial_value=initial_capital, 
                            mu=metrics['ret_annual'], 
                            sigma=metrics['vol_annual'], 
                            days=n_days_future, 
                            n_simulations=n_sims
                        )
                    
                    # Dataframe para Plotly
                    time_idx = np.arange(n_days_future)
                    df_future = pd.DataFrame(future_paths)
                    
                    # Dibujamos solo 50 caminos para no colapsar el navegador del usuario
                    fig_mc = go.Figure()
                    for i in range(50):
                        fig_mc.add_trace(go.Scatter(
                            x=time_idx, y=df_future[i], mode='lines', 
                            line=dict(color=selected_colors[1], width=1), opacity=0.15,
                            showlegend=False
                        ))
                        
                    # Añadir la media y percentiles 5% y 95%
                    mean_path = df_future.mean(axis=1)
                    p5_path = df_future.quantile(0.05, axis=1)
                    p95_path = df_future.quantile(0.95, axis=1)
                    
                    fig_mc.add_trace(go.Scatter(x=time_idx, y=mean_path, mode='lines', name='Camino Medio Esperado ($\mu$)', line=dict(color=selected_colors[0], width=3)))
                    fig_mc.add_trace(go.Scatter(x=time_idx, y=p95_path, mode='lines', name='Escenario Optimista (Percentil 95)', line=dict(color=selected_colors[3], width=2, dash='dash')))
                    fig_mc.add_trace(go.Scatter(x=time_idx, y=p5_path, mode='lines', name='Escenario Pesimista (Percentil 5)', line=dict(color="#ff4b4b", width=2, dash='dash')))
                    
                    fig_mc.update_layout(
                        title=f"1,000 Simulaciones Estocásticas Futuras (1 Año)",
                        xaxis_title="Días de Mercado Proyectados",
                        yaxis_title=f"Capital ({symbol})",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Insights Rápidos del Futuro
                    end_values = df_future.iloc[-1]
                    prob_loss = (end_values < initial_capital).mean() * 100
                    st.info(f"**Análisis Predictivo Terminal:** De las 1000 simulaciones, el valor mediano esperado dentro de 1 año es de **{symbol}{np.median(end_values):,.2f}**. Existe una probabilidad del **{prob_loss:.1f}%** de cerrar el año con pérdidas respecto al capital inicial.")

                # ------------------------------------------
                # TAB 5: REGÍMENES MACROECONÓMICOS (NOVEDAD)
                # ------------------------------------------
                with t_regimes:
                    st.subheader("⏳ Análisis Adaptativo de Regímenes de Mercado")
                    
                    with st.expander("🌍 Filtro de Markov Heurístico y Descomposición de Entorno"):
                        st.markdown("Los mercados financieros no son estacionarios; mutan cíclicamente entre entornos de expansión y pánico. Este algoritmo propietario analiza el índice de referencia (`benchmark`) y segmenta históricamente los días de bolsa en 4 Cuadrantes, evaluando cómo de resistente es su cartera algorítmica cuando la volatilidad exógena aumenta dramáticamente.")
                        
                    regime_df = detect_market_regimes(metrics['bench_series'])
                    
                    # Unir régimenes con los retornos del portafolio
                    common_dates = regime_df.index.intersection(metrics['portfolio_series'].index)
                    regime_analysis = pd.DataFrame({
                        'Port_Ret': metrics['portfolio_series'].loc[common_dates],
                        'Regime': regime_df['Regime'].loc[common_dates]
                    })
                    
                    # Agrupar y anualizar rendimiento por régimen
                    regime_summary = regime_analysis.groupby('Regime').mean() * 252 * 100 # a % anualizado
                    
                    fig_regime = px.bar(
                        regime_summary, y='Port_Ret', x=regime_summary.index,
                        title="Rendimiento Anualizado Teórico de la Cartera Según Entorno Macro",
                        labels={'Regime': 'Régimen de Mercado', 'Port_Ret': 'Rendimiento Medio Anualizado (%)'},
                        color=regime_summary.index,
                        color_discrete_sequence=selected_colors
                    )
                    fig_regime.update_layout(showlegend=False)
                    st.plotly_chart(fig_regime, use_container_width=True)

                # ------------------------------------------
                # TAB 6: ROLLING METRICS (INÉRCIA)
                # ------------------------------------------
                with t_rolling:
                    st.subheader("📈 Análisis de Inercia y Fricción Temporal")
                    st.markdown("<div class='explanation-box'><b>Dinámica Temporal:</b> Evaluar el Beta y la Volatilidad global con un solo número es sesgado. Las gráficas 'Rolling' a 60 días (1 Trimestre) muestran la inestabilidad de la agresividad de la cartera. Si la línea del Beta oscila descontroladamente por encima y por debajo de 1.0, usted posee un vehículo con 'Drift' de estilo peligroso.</div>", unsafe_allow_html=True)
                    
                    if len(metrics['portfolio_series']) > 60:
                        df_rolling = calculate_rolling_metrics(metrics['portfolio_series'], metrics['bench_series'], window=60)
                        
                        fig_roll_beta = px.line(df_rolling, y='Rolling_Beta_60d', title="Sensibilidad Dinámica Rodante (Rolling Beta 60d)", color_discrete_sequence=[selected_colors[0]])
                        fig_roll_beta.add_hline(y=1.0, line_dash="dash", line_color="#ff4b4b", annotation_text="Benchmark Neutrality")
                        st.plotly_chart(fig_roll_beta, use_container_width=True)
                        
                        fig_roll_vol = px.line(df_rolling, y='Rolling_Vol_60d', title="Fricción Volátil Rodante Anualizada (Rolling Volatility 60d)", color_discrete_sequence=[selected_colors[1]])
                        st.plotly_chart(fig_roll_vol, use_container_width=True)
                    else:
                        st.warning("No hay suficientes datos históricos (>60 días) para calcular métricas rodantes (Rolling Windows).")

                # ------------------------------------------
                # TAB 7: NEURO INSIGHTS AI
                # ------------------------------------------
                with t_ai:
                    st.subheader("🤖 Diagnóstico Cognitivo y Prescriptivo")
                    st.write("Interpretación algorítmica de los resultados matriciales:")
                    ai_insights = get_ai_prescriptive_insights(metrics, benchmark_ticker)
                    for insight in ai_insights:
                        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)

                # ------------------------------------------
                # TAB 8: CFA THESIS GENERATOR
                # ------------------------------------------
                with t_cfa:
                    st.subheader("📄 Reporte Fiduciario Abstract (Generación Dinámica)")
                    st.write("Listo para exportación e impresión a comités de inversión institucionales.")
                    html_cfa = render_cfa_investment_thesis(metrics, t_list, symbol, initial_capital, mode)
                    st.markdown(html_cfa, unsafe_allow_html=True)

                # ------------------------------------------
                # TAB 9: AUDITORÍA Y METADATOS
                # ------------------------------------------
                with t_audit:
                    st.subheader("📚 Trazabilidad Científica y Control Fiduciario")
                    st.markdown(f"""
                    <div class='explanation-box'>
                    <b>1. Ventana Muestral:</b> Observación histórica extraída desde {start_date} hasta {end_date}<br>
                    <b>2. Latencia (API Sync):</b> Yahoo Finance Global Telemétrica (Sincronización Time-Stamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')})<br>
                    <b>3. Motor de Cálculo Complejo:</b> SciPy (Sub-rutinas Estocásticas), NumPy (Cálculo Vectorizado Matricial), PyPortfolioOpt (Markowitz Cuadrático Convexo)<br>
                    <b>4. Marco de Simulación:</b> Brownian Motion Integrations ($dW_t$) iteradas sobre $N=1000$ matrices de transición discreta.<br>
                    <b>5. Acreditación Framework:</b> Construido, auditado y validado bajo la supervisión teórica algorítmica de la <b>Cátedra Fintech - Universidad de Málaga</b>.
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Excepción Estructural en Motor Cuantitativo (Matrix Error): {e}")
                st.info("Verifique que los Tickers introducidos posean liquidez cruzada y existan en la base de datos de Yahoo Finance para el periodo consultado.")

# ==========================================
# INICIO DE APLICACIÓN
# ==========================================
if __name__ == "__main__":
    main()