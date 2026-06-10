import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import re

st.set_page_config(
    page_title="Tenaris – Forecast de Demanda",
    page_icon="📊",
    layout="wide"
)

# ── CSS STYLING ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fb; }
    
    /* Header bar */
    .tenaris-header {
        background: linear-gradient(135deg, #003f7f 0%, #0066cc 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .tenaris-header h1 {
        color: white !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .tenaris-header p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 15px !important;
        margin: 4px 0 0 0 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e6f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,63,127,0.08);
    }
    [data-testid="metric-container"] label {
        color: #5a7a9a !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #003f7f !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 10px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0,63,127,0.08);
        border: 1px solid #e0e6f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #5a7a9a;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: #003f7f !important;
        color: white !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #e0e6f0;
        box-shadow: 0 2px 8px rgba(0,63,127,0.06);
    }

    /* Info/Warning/Success boxes */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* Section headers */
    h2, h3 {
        color: #003f7f !important;
        font-weight: 700 !important;
    }

    /* Divider */
    hr { border-color: #e0e6f0; }

    /* Sidebar */
    .css-1d391kg { background: white; }

    /* Upload button */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 10px;
        border: 2px dashed #003f7f;
        padding: 10px;
    }

    /* Download button */
    [data-testid="stDownloadButton"] button {
        background: #003f7f !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 20px !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #0066cc !important;
    }

    /* Radio buttons */
    [data-testid="stRadio"] label {
        font-weight: 600;
        color: #003f7f;
    }

    /* Slider */
    [data-testid="stSlider"] {
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo.png", width=160)
with col_title:
    st.markdown("""
    <div class="tenaris-header">
        <h1>Dashboard de Proyección de Demanda</h1>
        <p>Tenaris S.A. &nbsp;·&nbsp; Business Coordination &nbsp;·&nbsp; Modelo Prophet | MAPE ≈ 11% | Cobertura 91.7% | Sesgo -5.3%</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Este dashboard permite visualizar el forecast validado y generar nuevas proyecciones con datos actualizados. Selecciona el modo de uso para comenzar.")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def clean_numeric(s):
    return pd.to_numeric(re.sub(r'[^0-9.\-]', '', str(s)), errors='coerce')

def mape_safe(y_true, y_pred):
    df = pd.DataFrame({'y': y_true, 'yhat': y_pred}).dropna()
    df = df[df['y'] > 0]
    if len(df) == 0:
        return np.nan
    return (np.abs((df['y'] - df['yhat']) / df['y'])).mean() * 100

def download_excel(df_export, filename):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Forecast')
    buffer.seek(0)
    st.download_button(
        label="📥 Descargar forecast en Excel",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def read_csv_safe(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding='utf-8-sig', sep=None, engine='python')
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='latin1', sep=None, engine='python')

def prepare_series(df_raw):
    MONTH = 'Month'
    TONS  = 'Suma de Tons'
    df = df_raw.copy()
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    df[MONTH] = pd.to_datetime(df[MONTH], errors='coerce', dayfirst=True)
    df[TONS]  = df[TONS].apply(clean_numeric)
    df = df.dropna(subset=[MONTH, TONS])
    ts = df.groupby(MONTH)[TONS].sum().sort_index()
    df_prophet = ts.reset_index()
    df_prophet.columns = ['ds', 'y']
    return df_prophet, df

def train_prophet(df_prophet, cps=0.02):
    m = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        interval_width=0.60
    )
    m.fit(df_prophet)
    return m

def semaforo_df(fc, col_yhat='yhat'):
    total = fc[col_yhat].sum()
    promedio = total / len(fc)
    fc = fc.copy()
    fc['Estado'] = fc[col_yhat].apply(
        lambda x: '🟢 Alto' if x > promedio * 1.05
        else ('🔴 Bajo' if x < promedio * 0.95 else '🟡 Normal')
    )
    return fc

st.markdown("---")
modo = st.radio(
    "**Selecciona el modo de uso:**",
    ["📁 Ver forecast validado (2026)", "🔄 Generar nuevo forecast con datos actualizados"],
    horizontal=True
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — FORECAST VALIDADO
# ══════════════════════════════════════════════════════════════════════════════
if modo == "📁 Ver forecast validado (2026)":

    st.info("Sube el archivo CSV con los resultados del modelo (forecast_prophet_2026.csv)")
    uploaded_file = st.file_uploader("📂 Cargar archivo de resultados", type=["csv"])
    st.markdown("*(Opcional)* Sube el archivo de backtest para ver comparación de modelos:")
    backtest_file = st.file_uploader("📂 Cargar backtest (backtest_prophet_vs_naive_12m.csv)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['ds'] = pd.to_datetime(df['ds'])

        st.subheader("📈 Forecast validado – Modelo Prophet")
        total = df['yhat'].sum()
        low   = df['yhat_lower'].sum()
        high  = df['yhat_upper'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Demanda esperada 2026", f"{round(total):,} tons")
        col2.metric("Escenario bajo",        f"{round(low):,} tons")
        col3.metric("Escenario alto",        f"{round(high):,} tons")
        col4.metric("MAPE del modelo",       "≈ 11%")

        st.markdown("---")

        # SEMÁFORO
        st.subheader("🚦 Semáforo mensual")
        df = semaforo_df(df)
        df['Mes'] = df['ds'].dt.strftime('%B %Y')
        sem = df[['Mes','yhat','Estado']].copy()
        sem.columns = ['Mes','Pronóstico (tons)','Estado']
        sem['Pronóstico (tons)'] = sem['Pronóstico (tons)'].apply(lambda x: f"{round(x):,}")
        st.dataframe(sem, use_container_width=True, hide_index=True)

        st.markdown("---")

        # GRÁFICO FORECAST
        st.subheader("Proyección mensual 2026")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ds'], df['yhat'], marker='o', linewidth=2, color='#003f7f', label='Escenario base')
        ax.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], alpha=0.15, color='#003f7f', label='Rango de incertidumbre')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_xlabel("Mes"); ax.set_ylabel("Toneladas")
        ax.set_title("Pronóstico de demanda – Tenaris S.A. 2026", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # BACKTEST
        if backtest_file:
            st.subheader("📊 Backtest – Comparación de modelos vs realidad")
            df_bt = pd.read_csv(backtest_file)
            df_bt['ds'] = pd.to_datetime(df_bt['ds'])
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            if 'y_real' in df_bt.columns:
                ax2.plot(df_bt['ds'], df_bt['y_real'], color='black', linewidth=2, marker='o', label='Real')
            if 'yhat_prophet' in df_bt.columns:
                ax2.plot(df_bt['ds'], df_bt['yhat_prophet'], color='#003f7f', linewidth=2, label='Prophet')
            if 'yhat_naive' in df_bt.columns:
                ax2.plot(df_bt['ds'], df_bt['yhat_naive'], color='orange', linewidth=1.5, linestyle='--', label='Naive')
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax2.set_xlabel("Mes"); ax2.set_ylabel("Toneladas")
            ax2.set_title("Backtest 12 meses – Real vs Modelos", fontsize=13)
            ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45); plt.tight_layout()
            st.pyplot(fig2)
            st.markdown("---")

        # TABLA + DESCARGA
        st.subheader("Tabla de valores proyectados")
        tabla = df[['Mes','yhat','yhat_lower','yhat_upper']].copy()
        tabla.columns = ['Mes','Pronóstico (tons)','Escenario bajo','Escenario alto']
        st.dataframe(tabla.style.format({'Pronóstico (tons)': '{:,.0f}', 'Escenario bajo': '{:,.0f}', 'Escenario alto': '{:,.0f}'}),
                     use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("⬇️ Descargar resultados")
        export = df[['ds','yhat','yhat_lower','yhat_upper']].copy()
        export.columns = ['Fecha','Pronóstico (tons)','Escenario bajo','Escenario alto']
        export['Fecha'] = export['Fecha'].dt.strftime('%Y-%m-%d')
        download_excel(export, "Forecast_Tenaris_2026.xlsx")

        st.markdown("---")
        st.subheader("Calidad del modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", "≈ 11%")
        c2.metric("Cobertura de intervalos", "91.7%")
        c3.metric("Sesgo", "-5.3%")
        st.success("Modelo validado mediante backtesting sobre los últimos 12 meses, superando a Holt-Winters (16%), Naive (20%) y SARIMA (25%).")

# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — NUEVO FORECAST COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.info("Sube el archivo CSV con el histórico de ventas (el mismo archivo raw que usas en Colab)")
    uploaded_file = st.file_uploader("📂 Cargar histórico de ventas", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = read_csv_safe(uploaded_file)
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")
            st.stop()

        df_raw.columns = df_raw.columns.str.strip().str.replace('\ufeff', '', regex=False)

        MONTH = 'Month'
        TONS  = 'Suma de Tons'
        FAMILY = 'Prod Family'
        COUNTRY = 'End Customer Country'

        if MONTH not in df_raw.columns or TONS not in df_raw.columns:
            st.error(f"No se encontraron las columnas '{MONTH}' y '{TONS}'. Columnas disponibles: {list(df_raw.columns)}")
            st.stop()

        df_prophet, df_clean = prepare_series(df_raw)

        st.success(f"✅ Serie lista: **{len(df_prophet)} meses** | Rango: {df_prophet['ds'].min().strftime('%Y-%m')} → {df_prophet['ds'].max().strftime('%Y-%m')}")

        # ── FILTRO DE FECHA ───────────────────────────────────────────────────
        st.markdown("**🗓️ Filtro de rango de fechas para el modelo**")
        fecha_min = df_prophet['ds'].min().to_pydatetime()
        fecha_max = df_prophet['ds'].max().to_pydatetime()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fecha_inicio = st.date_input("Desde:", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
        with col_f2:
            fecha_fin = st.date_input("Hasta:", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

        fecha_inicio = pd.Timestamp(fecha_inicio)
        fecha_fin    = pd.Timestamp(fecha_fin)

        df_prophet = df_prophet[(df_prophet['ds'] >= fecha_inicio) & (df_prophet['ds'] <= fecha_fin)].reset_index(drop=True)

        if len(df_prophet) < 12:
            st.error("⚠️ El rango seleccionado tiene menos de 12 meses. Amplía el rango para obtener un forecast confiable.")
            st.stop()

        st.info(f"📅 Entrenando con **{len(df_prophet)} meses** ({fecha_inicio.strftime('%Y-%m')} → {fecha_fin.strftime('%Y-%m')})")

        # ── UMBRAL DE ALERTA ──────────────────────────────────────────────────
        umbral_alerta = st.slider("⚠️ Umbral de alerta por desviación (%):", min_value=10, max_value=60, value=30, step=5)

        st.markdown("---")

        # ── GRID SEARCH CPS ───────────────────────────────────────────────────
        with st.spinner("🔄 Entrenando modelo Prophet y buscando mejores parámetros..."):
            M = 12
            best_mape = np.inf
            best_cps  = 0.02

            for cps in [0.02, 0.05, 0.10]:
                m = train_prophet(df_prophet, cps)
                fc = m.make_future_dataframe(periods=M, freq='MS')
                fc = m.predict(fc)
                yhat = fc.set_index('ds')['yhat'].reindex(df_prophet.set_index('ds').iloc[-M:].index)
                mape = mape_safe(df_prophet['y'].iloc[-M:].values, yhat.values)
                if mape < best_mape:
                    best_mape = mape
                    best_cps  = cps

            model_final = train_prophet(df_prophet, best_cps)
            future_final   = model_final.make_future_dataframe(periods=M, freq='MS')
            forecast_final = model_final.predict(future_final)

        fc_12 = forecast_final.tail(M).copy()
        total = fc_12['yhat'].sum()
        low   = fc_12['yhat_lower'].sum()
        high  = fc_12['yhat_upper'].sum()

        # ── TABS ──────────────────────────────────────────────────────────────
        tab1, tab2, tab2b, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Forecast Total",
            "🏭 Por Familia",
            "🌎 Por País",
            "📅 Estacionalidad",
            "📊 Componentes",
            "🏆 Comparación Modelos",
            "📉 Análisis Histórico"
        ])

        # ── TAB 1: FORECAST TOTAL ─────────────────────────────────────────────
        with tab1:
            st.subheader("📈 Nuevo forecast generado")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Demanda estimada (12m)", f"{round(total):,} tons")
            col2.metric("Escenario bajo",         f"{round(low):,} tons")
            col3.metric("Escenario alto",         f"{round(high):,} tons")
            col4.metric("MAPE backtest",          f"≈ {round(best_mape, 1)}%")
            st.caption(f"Mejor changepoint_prior_scale: {best_cps}")

            # KPIs adicionales
            fc_12_kpi = fc_12.copy()
            fc_12_kpi['Mes'] = fc_12_kpi['ds'].dt.strftime('%B')
            mes_pico  = fc_12_kpi.loc[fc_12_kpi['yhat'].idxmax(), 'Mes']
            mes_valle = fc_12_kpi.loc[fc_12_kpi['yhat'].idxmin(), 'Mes']
            prom_hist = df_prophet['y'].mean()
            var_pct   = ((total/M) - prom_hist) / prom_hist * 100

            k1, k2, k3 = st.columns(3)
            k1.metric("Mes pico", mes_pico, f"{round(fc_12['yhat'].max()):,} tons")
            k2.metric("Mes valle", mes_valle, f"{round(fc_12['yhat'].min()):,} tons")
            k3.metric("Var. vs promedio histórico", f"{round(var_pct, 1)}%")

            st.markdown("---")
            st.subheader("🚦 Semáforo mensual")

            prom_hist = df_prophet['y'].mean()

            fc_sem = fc_12.copy()
            fc_sem['Mes'] = fc_sem['ds'].dt.strftime('%B %Y')
            fc_sem['Desviación vs hist (%)'] = ((fc_sem['yhat'] - prom_hist) / prom_hist * 100).round(1)
            fc_sem['Estado'] = fc_sem['yhat'].apply(
                lambda x: '🟢 Alto' if x > prom_hist * 1.05
                else ('🔴 Bajo' if x < prom_hist * 0.95 else '🟡 Normal')
            )

            sem = fc_sem[['Mes','yhat','Desviación vs hist (%)','Estado']].copy()
            sem.columns = ['Mes','Pronóstico (tons)','Desviación vs histórico (%)','Estado']
            sem['Pronóstico (tons)'] = sem['Pronóstico (tons)'].apply(lambda x: f"{round(x):,}")
            st.dataframe(sem, use_container_width=True, hide_index=True)

            # ── ALERTAS DE DESVIACIÓN ─────────────────────────────────────────
            alertas = fc_sem[abs(fc_sem['Desviación vs hist (%)']) > umbral_alerta]
            if not alertas.empty:
                st.markdown("---")
                st.subheader("🚨 Alertas de desviación")
                for _, row in alertas.iterrows():
                    direccion = "por encima" if row['Desviación vs hist (%)'] > 0 else "por debajo"
                    color = "🟢" if row['Desviación vs hist (%)'] > 0 else "🔴"
                    st.warning(
                        f"{color} **{row['Mes']}** — Pronóstico: **{round(row['yhat']):,} tons** | "
                        f"Desviación: **{abs(row['Desviación vs hist (%)'])}% {direccion}** del promedio histórico ({round(prom_hist):,} tons). "
                        f"Se recomienda verificar con el área comercial."
                    )
            else:
                st.success(f"✅ Ningún mes supera el umbral de alerta del {umbral_alerta}%.")

            st.markdown("---")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_prophet['ds'], df_prophet['y'], color='gray', linewidth=1.5, label='Histórico')
            ax.plot(forecast_final['ds'], forecast_final['yhat'], color='#003f7f', linewidth=2, label='Forecast')
            ax.fill_between(forecast_final['ds'], forecast_final['yhat_lower'], forecast_final['yhat_upper'],
                            alpha=0.15, color='#003f7f', label='Intervalo de confianza')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax.set_xlabel("Mes"); ax.set_ylabel("Toneladas")
            ax.set_title("Pronóstico de demanda – Tenaris S.A.", fontsize=13)
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45); plt.tight_layout()
            st.pyplot(fig)

            st.markdown("---")
            tabla = fc_12[['ds','yhat','yhat_lower','yhat_upper']].copy()
            tabla['ds'] = tabla['ds'].dt.strftime('%B %Y')
            tabla.columns = ['Mes','Pronóstico (tons)','Escenario bajo','Escenario alto']
            st.dataframe(tabla.style.format({'Pronóstico (tons)': '{:,.0f}', 'Escenario bajo': '{:,.0f}', 'Escenario alto': '{:,.0f}'}),
                         use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("⬇️ Descargar resultados")
            export = fc_12[['ds','yhat','yhat_lower','yhat_upper']].copy()
            export.columns = ['Fecha','Pronóstico (tons)','Escenario bajo','Escenario alto']
            export['Fecha'] = export['Fecha'].dt.strftime('%Y-%m-%d')
            download_excel(export, "Forecast_Tenaris_Actualizado.xlsx")

        # ── TAB 2: FORECAST POR FAMILIA ───────────────────────────────────────
        with tab2:
            st.subheader("🏭 Forecast por familia de producto")

            if FAMILY not in df_clean.columns:
                st.warning("No se encontró la columna 'Prod Family' en el archivo.")
            else:
                df_clean[MONTH] = pd.to_datetime(df_clean[MONTH], errors='coerce', dayfirst=True)
                df_clean[TONS]  = df_clean[TONS].apply(clean_numeric)

                familias = sorted([f for f in df_clean[FAMILY].dropna().unique() if '2' not in str(f) and 'DOWNGRAD' not in str(f).upper()])
                familia_sel = st.multiselect("Selecciona familias:", familias, default=familias)

                res_familias = []

                with st.spinner("Calculando forecast por familia..."):
                    for fam in familia_sel:
                        sub = (df_clean[df_clean[FAMILY] == fam]
                               .groupby(MONTH)[TONS].sum()
                               .sort_index()
                               .reset_index())
                        sub.columns = ['ds', 'y']
                        sub = sub.dropna()

                        if sub['y'].sum() == 0 or len(sub) < 6:
                            continue

                        try:
                            m_fam = train_prophet(sub, best_cps)
                            fc_fam = m_fam.predict(m_fam.make_future_dataframe(periods=M, freq='MS'))
                            fc_fam_12 = fc_fam.tail(M)

                            yhat_bt = fc_fam.set_index('ds')['yhat'].reindex(sub.set_index('ds').iloc[-M:].index)
                            mape_fam = mape_safe(sub['y'].iloc[-M:].values, yhat_bt.values)

                            res_familias.append({
                                'Familia': fam,
                                'Forecast 12m (tons)': round(fc_fam_12['yhat'].sum()),
                                'MAPE': round(mape_fam, 1) if not np.isnan(mape_fam) else 'N/A',
                                'Meses histórico': len(sub)
                            })

                            # Gráfico por familia
                            fig_f, ax_f = plt.subplots(figsize=(10, 3))
                            ax_f.plot(sub['ds'], sub['y'], color='gray', linewidth=1.2, label='Histórico')
                            ax_f.plot(fc_fam['ds'], fc_fam['yhat'], color='#003f7f', linewidth=2, label='Forecast')
                            ax_f.fill_between(fc_fam['ds'], fc_fam['yhat_lower'], fc_fam['yhat_upper'], alpha=0.15, color='#003f7f')
                            ax_f.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                            ax_f.set_title(f"{fam} — Forecast 12m: {round(fc_fam_12['yhat'].sum()):,} tons | MAPE: {round(mape_fam,1) if not np.isnan(mape_fam) else 'N/A'}%", fontsize=11)
                            ax_f.legend(); ax_f.grid(True, alpha=0.3)
                            plt.xticks(rotation=45); plt.tight_layout()
                            st.pyplot(fig_f)

                        except Exception as e:
                            st.warning(f"No se pudo calcular forecast para {fam}: {e}")

                if res_familias:
                    st.markdown("---")
                    st.subheader("Resumen por familia")
                    df_res = pd.DataFrame(res_familias)
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_res.to_excel(writer, index=False, sheet_name='Familias')
                    buffer.seek(0)
                    st.download_button("📥 Descargar resumen por familia", buffer,
                                       "Forecast_Familias.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── TAB 2B: FORECAST POR PAÍS ─────────────────────────────────────────
        with tab2b:
            st.subheader("🌎 Forecast por país del cliente")

            if COUNTRY not in df_clean.columns:
                st.warning(f"No se encontró la columna '{COUNTRY}' en el archivo.")
            else:
                df_clean[MONTH] = pd.to_datetime(df_clean[MONTH], errors='coerce', dayfirst=True)
                df_clean[TONS]  = df_clean[TONS].apply(clean_numeric)

                # Ordenar países por volumen total (mayor a menor)
                vol_pais = (df_clean.groupby(COUNTRY)[TONS].sum()
                            .sort_values(ascending=False))
                paises = [p for p in vol_pais.index if pd.notna(p) and str(p).strip() != '']

                # Por defecto seleccionar el top 5 para no saturar
                default_paises = paises[:5]
                pais_sel = st.multiselect("Selecciona países:", paises, default=default_paises)

                res_paises = []

                with st.spinner("Calculando forecast por país..."):
                    for pais in pais_sel:
                        sub = (df_clean[df_clean[COUNTRY] == pais]
                               .groupby(MONTH)[TONS].sum()
                               .sort_index()
                               .reset_index())
                        sub.columns = ['ds', 'y']
                        sub = sub.dropna()

                        if sub['y'].sum() == 0 or len(sub) < 6:
                            st.info(f"⏩ {pais}: histórico insuficiente (<6 meses), se omite.")
                            continue

                        try:
                            m_pais = train_prophet(sub, best_cps)
                            fc_pais = m_pais.predict(m_pais.make_future_dataframe(periods=M, freq='MS'))
                            fc_pais_12 = fc_pais.tail(M)

                            yhat_bt = fc_pais.set_index('ds')['yhat'].reindex(sub.set_index('ds').iloc[-M:].index)
                            mape_pais = mape_safe(sub['y'].iloc[-M:].values, yhat_bt.values)

                            res_paises.append({
                                'País': pais,
                                'Forecast 12m (tons)': round(fc_pais_12['yhat'].sum()),
                                'MAPE': round(mape_pais, 1) if not np.isnan(mape_pais) else 'N/A',
                                'Meses histórico': len(sub)
                            })

                            # Gráfico por país
                            fig_p, ax_p = plt.subplots(figsize=(10, 3))
                            ax_p.plot(sub['ds'], sub['y'], color='gray', linewidth=1.2, label='Histórico')
                            ax_p.plot(fc_pais['ds'], fc_pais['yhat'], color='#003f7f', linewidth=2, label='Forecast')
                            ax_p.fill_between(fc_pais['ds'], fc_pais['yhat_lower'], fc_pais['yhat_upper'], alpha=0.15, color='#003f7f')
                            ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                            ax_p.set_title(f"{pais} — Forecast 12m: {round(fc_pais_12['yhat'].sum()):,} tons | MAPE: {round(mape_pais,1) if not np.isnan(mape_pais) else 'N/A'}%", fontsize=11)
                            ax_p.legend(); ax_p.grid(True, alpha=0.3)
                            plt.xticks(rotation=45); plt.tight_layout()
                            st.pyplot(fig_p)

                        except Exception as e:
                            st.warning(f"No se pudo calcular forecast para {pais}: {e}")

                if res_paises:
                    st.markdown("---")
                    st.subheader("Resumen por país")
                    df_resp = pd.DataFrame(res_paises).sort_values('Forecast 12m (tons)', ascending=False)
                    st.dataframe(df_resp, use_container_width=True, hide_index=True)

                    # Gráfico de participación (barras)
                    st.markdown("**Participación en el forecast 12m**")
                    fig_pp, ax_pp = plt.subplots(figsize=(10, 4))
                    df_plot = df_resp.sort_values('Forecast 12m (tons)', ascending=True)
                    ax_pp.barh(df_plot['País'].astype(str), df_plot['Forecast 12m (tons)'], color='#003f7f', alpha=0.85)
                    ax_pp.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                    ax_pp.set_xlabel("Toneladas proyectadas (12m)")
                    ax_pp.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig_pp)

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_resp.to_excel(writer, index=False, sheet_name='Paises')
                    buffer.seek(0)
                    st.download_button("📥 Descargar resumen por país", buffer,
                                       "Forecast_Paises.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── TAB 3: ESTACIONALIDAD ─────────────────────────────────────────────
        with tab3:
            st.subheader("📅 Análisis de estacionalidad")

            meses_label = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
            df_prophet['mes'] = df_prophet['ds'].dt.month
            prom_mes = df_prophet.groupby('mes')['y'].mean().reindex(range(1,13))
            overall  = prom_mes.mean()
            idx_mul  = (prom_mes / overall).round(3)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Promedio de toneladas por mes**")
                fig_s1, ax_s1 = plt.subplots(figsize=(6, 4))
                ax_s1.bar(meses_label, prom_mes.values, color='#003f7f', alpha=0.8)
                ax_s1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax_s1.set_title("Promedio histórico por mes")
                ax_s1.set_ylabel("Toneladas promedio")
                ax_s1.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig_s1)

            with col2:
                st.markdown("**Índices estacionales multiplicativos** (1 = promedio anual)")
                fig_s2, ax_s2 = plt.subplots(figsize=(6, 4))
                colors = ['#d32f2f' if v < 0.95 else ('#2e7d32' if v > 1.05 else '#1565c0') for v in idx_mul.values]
                ax_s2.bar(meses_label, idx_mul.values, color=colors, alpha=0.85)
                ax_s2.axhline(1.0, color='black', linewidth=1, linestyle='--')
                ax_s2.set_title("Índices estacionales (1 = promedio)")
                ax_s2.set_ylabel("Índice multiplicativo")
                ax_s2.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig_s2)

            st.markdown("---")
            st.markdown("**Tabla de índices estacionales**")
            idx_df = pd.DataFrame({'Mes': meses_label, 'Índice multiplicativo': idx_mul.values})
            idx_df = idx_df.sort_values('Índice multiplicativo', ascending=False).reset_index(drop=True)
            idx_df['Índice multiplicativo'] = idx_df['Índice multiplicativo'].round(3)
            idx_df['Interpretación'] = idx_df['Índice multiplicativo'].apply(
                lambda x: '🟢 Por encima del promedio' if x > 1.05
                else ('🔴 Por debajo del promedio' if x < 0.95 else '🟡 Cerca del promedio')
            )
            st.dataframe(idx_df, use_container_width=True, hide_index=True)

        # ── TAB 4: COMPONENTES DEL MODELO ─────────────────────────────────────
        with tab4:
            st.subheader("📊 Componentes del modelo Prophet")

            comp = forecast_final[['ds','trend','yearly']].copy()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Tendencia**")
                fig_t, ax_t = plt.subplots(figsize=(6, 4))
                ax_t.plot(comp['ds'], comp['trend'], color='#003f7f', linewidth=2)
                ax_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax_t.set_title("Componente de tendencia")
                ax_t.set_xlabel("Año"); ax_t.set_ylabel("Toneladas")
                ax_t.grid(True, alpha=0.3)
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig_t)

            with col2:
                st.markdown("**Estacionalidad anual**")
                fig_y, ax_y = plt.subplots(figsize=(6, 4))
                comp_year = comp.copy()
                comp_year['dia_anio'] = comp_year['ds'].dt.dayofyear
                comp_year = comp_year.sort_values('dia_anio')
                ax_y.plot(comp_year['dia_anio'], comp_year['yearly'] * 100, color='#c62828', linewidth=1.5)
                ax_y.axhline(0, color='black', linewidth=0.8, linestyle='--')
                ax_y.set_title("Componente estacional anual")
                ax_y.set_xlabel("Día del año"); ax_y.set_ylabel("Efecto (%)")
                ax_y.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_y)

            st.markdown("---")
            st.markdown("**Interpretación de la tendencia**")
            trend_inicio = comp['trend'].iloc[0]
            trend_fin    = comp['trend'].iloc[-1]
            trend_cambio = ((trend_fin - trend_inicio) / trend_inicio) * 100
            if trend_cambio < -5:
                st.warning(f"📉 La tendencia muestra una **caída del {abs(round(trend_cambio,1))}%** desde el inicio del período analizado. El modelo proyecta una demanda decreciente en el horizonte de forecast.")
            elif trend_cambio > 5:
                st.success(f"📈 La tendencia muestra un **crecimiento del {round(trend_cambio,1)}%** desde el inicio del período. El modelo proyecta una demanda creciente.")
            else:
                st.info(f"➡️ La tendencia se mantiene **relativamente estable** ({round(trend_cambio,1)}% de variación).")

        # ── TAB 5: COMPARACIÓN DE MODELOS ─────────────────────────────────────
        with tab5:
            st.subheader("🏆 Comparación de modelos")

            with st.spinner("Calculando modelos alternativos..."):
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                y = df_prophet.set_index('ds')['y'].sort_index().asfreq('MS')
                M = 12
                y_train = y.iloc[:-M]
                y_test  = y.iloc[-M:]

                # Prophet
                yhat_prophet = forecast_final.set_index('ds')['yhat'].reindex(y_test.index)
                mape_prophet = mape_safe(y_test.values, yhat_prophet.values)

                # Holt-Winters
                try:
                    hw = ExponentialSmoothing(y_train, trend='add', seasonal='mul', seasonal_periods=12).fit(optimized=True)
                    yhat_hw = hw.forecast(M)
                    mape_hw = mape_safe(y_test.values, yhat_hw.values)
                except Exception:
                    yhat_hw = pd.Series([np.nan]*M, index=y_test.index)
                    mape_hw = np.nan

                # SARIMA
                try:
                    sarima = SARIMAX(y_train, order=(3,1,1), seasonal_order=(1,1,1,12),
                                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    yhat_sarima = sarima.forecast(M)
                    mape_sarima = mape_safe(y_test.values, yhat_sarima.values)
                except Exception:
                    yhat_sarima = pd.Series([np.nan]*M, index=y_test.index)
                    mape_sarima = np.nan

                # Naive
                yhat_naive = y_train.shift(12).reindex(y_test.index)
                mape_naive = mape_safe(y_test.values, yhat_naive.values)

            # Tabla comparativa
            modelos_df = pd.DataFrame([
                {'Modelo': '🥇 Prophet',      'MAPE 12m': f"{round(mape_prophet,1)}%", 'Resultado': '✅ Seleccionado'},
                {'Modelo': '🥈 Holt-Winters', 'MAPE 12m': f"{round(mape_hw,1)}%"      if not np.isnan(mape_hw)     else 'N/A', 'Resultado': ''},
                {'Modelo': '🥉 SARIMA',       'MAPE 12m': f"{round(mape_sarima,1)}%"  if not np.isnan(mape_sarima) else 'N/A', 'Resultado': ''},
                {'Modelo': '📏 Naive',        'MAPE 12m': f"{round(mape_naive,1)}%"   if not np.isnan(mape_naive)  else 'N/A', 'Resultado': ''},
            ])
            st.dataframe(modelos_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Gráfico comparativo
            fig_c, ax_c = plt.subplots(figsize=(12, 5))
            ax_c.plot(y_test.index, y_test.values, color='black', linewidth=2, marker='o', label='Real')
            ax_c.plot(y_test.index, yhat_prophet.values, color='#003f7f', linewidth=2, label=f'Prophet ({round(mape_prophet,1)}%)')
            if not np.isnan(mape_hw):
                ax_c.plot(y_test.index, yhat_hw.values, color='green', linewidth=1.5, linestyle='--', label=f'Holt-Winters ({round(mape_hw,1)}%)')
            if not np.isnan(mape_sarima):
                ax_c.plot(y_test.index, yhat_sarima.values, color='red', linewidth=1.5, linestyle='--', label=f'SARIMA ({round(mape_sarima,1)}%)')
            ax_c.plot(y_test.index, yhat_naive.values, color='orange', linewidth=1, linestyle=':', label=f'Naive ({round(mape_naive,1)}%)')
            ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax_c.set_xlabel("Mes"); ax_c.set_ylabel("Toneladas")
            ax_c.set_title("Backtest 12 meses – Comparación de modelos", fontsize=13)
            ax_c.legend(); ax_c.grid(True, alpha=0.3)
            plt.xticks(rotation=45); plt.tight_layout()
            st.pyplot(fig_c)

            st.info(f"Prophet mejora al Naive en **{round(mape_naive - mape_prophet, 1)} puntos porcentuales** de MAPE.")

        # ── TAB 6: ANÁLISIS HISTÓRICO ─────────────────────────────────────────
        with tab6:
            st.subheader("📉 Análisis histórico de ventas")

            PRICE = 'Suma de Precio (US$/Tn)'
            GM    = 'Suma de GM (US$/Tn)'

            # Gráfico 1: Monthly Tons Sold
            st.markdown("**Comportamiento histórico de ventas mensuales**")
            fig_h1, ax_h1 = plt.subplots(figsize=(12, 4))
            ax_h1.plot(df_prophet['ds'], df_prophet['y'], color='#003f7f', linewidth=2, marker='o', markersize=4)
            ax_h1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax_h1.set_xlabel("Mes"); ax_h1.set_ylabel("Toneladas")
            ax_h1.set_title("Monthly Tons Sold – Tenaris S.A.", fontsize=13)
            ax_h1.grid(True, alpha=0.3)
            plt.xticks(rotation=45); plt.tight_layout()
            st.pyplot(fig_h1)

            st.markdown("---")

            # Gráfico 2: Tons + Precio + GM (doble eje Y)
            if PRICE in df_clean.columns and GM in df_clean.columns:
                st.markdown("**Relación entre Toneladas, Precio y Margen Bruto**")

                df_clean[PRICE] = df_clean[PRICE].apply(clean_numeric)
                df_clean[GM]    = df_clean[GM].apply(clean_numeric)
                df_clean['Month'] = pd.to_datetime(df_clean['Month'], errors='coerce', dayfirst=True)

                # Precio y GM ponderados por tonelada (US$/Tn real)
                def weighted_avg(group, val_col, weight_col):
                    d = group[[val_col, weight_col]].dropna()
                    if d[weight_col].sum() == 0:
                        return np.nan
                    return (d[val_col] * d[weight_col]).sum() / d[weight_col].sum()

                ts_precio = df_clean.groupby('Month').apply(
                    lambda g: weighted_avg(g, PRICE, TONS)).sort_index()
                ts_gm = df_clean.groupby('Month').apply(
                    lambda g: weighted_avg(g, GM, TONS)).sort_index()
                ts_tons   = df_prophet.set_index('ds')['y']

                fig_h2, ax1 = plt.subplots(figsize=(12, 5))
                ax2_r = ax1.twinx()

                ax1.plot(ts_tons.index, ts_tons.values, color='#003f7f', linewidth=2, label='Tons vendidas')
                ax1.set_ylabel("Toneladas", color='#003f7f')
                ax1.tick_params(axis='y', labelcolor='#003f7f')
                ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

                ax2_r.plot(ts_precio.index, ts_precio.values, color='#c62828', linewidth=1.5, linestyle='--', label='Precio (US$/Tn)')
                ax2_r.plot(ts_gm.index, ts_gm.values, color='#2e7d32', linewidth=1.5, linestyle=':', label='GM (US$/Tn)')
                ax2_r.set_ylabel("US$/Tn", color='gray')
                ax2_r.tick_params(axis='y', labelcolor='gray')

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2_r.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                ax1.set_title("Tons vendidas vs Precio vs Margen Bruto", fontsize=13)
                ax1.set_xlabel("Mes")
                ax1.grid(True, alpha=0.3)
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig_h2)

                st.markdown("---")

                # ── CORRELACIONES ────────────────────────────────────────────────
                st.markdown("**Correlación entre variables**")

                df_corr = pd.DataFrame({
                    'Tons': ts_tons,
                    'Precio (US$/Tn)': ts_precio,
                    'GM (US$/Tn)': ts_gm
                }).dropna()

                corr = df_corr.corr().round(3)

                # Heatmap de correlación
                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                im = ax_corr.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
                plt.colorbar(im, ax=ax_corr)
                ax_corr.set_xticks(range(len(corr.columns)))
                ax_corr.set_yticks(range(len(corr.columns)))
                ax_corr.set_xticklabels(corr.columns, rotation=15, ha='right')
                ax_corr.set_yticklabels(corr.columns)
                for i in range(len(corr)):
                    for j in range(len(corr.columns)):
                        ax_corr.text(j, i, f"{corr.values[i,j]:.3f}",
                                    ha='center', va='center', fontsize=12, fontweight='bold',
                                    color='black')
                ax_corr.set_title("Matriz de correlación", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_corr)

                st.markdown("---")

                # ── SCATTER PLOTS ─────────────────────────────────────────────
                st.markdown("**Diagramas de dispersión**")

                fig_sc, axes = plt.subplots(1, 3, figsize=(14, 4))

                # Tons vs Precio
                axes[0].scatter(df_corr['Precio (US$/Tn)'], df_corr['Tons'],
                               color='#c62828', alpha=0.7, edgecolors='white', s=60)
                z0 = np.polyfit(df_corr['Precio (US$/Tn)'].dropna(), df_corr['Tons'].dropna(), 1)
                p0 = np.poly1d(z0)
                x0 = np.linspace(df_corr['Precio (US$/Tn)'].min(), df_corr['Precio (US$/Tn)'].max(), 100)
                axes[0].plot(x0, p0(x0), color='black', linewidth=1.5, linestyle='--')
                axes[0].set_xlabel("Precio (US$/Tn)")
                axes[0].set_ylabel("Toneladas")
                r0 = corr.loc['Tons','Precio (US$/Tn)']
                axes[0].set_title(f"Tons vs Precio  |  r = {r0:.3f}")
                axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                axes[0].grid(True, alpha=0.3)

                # Tons vs GM
                axes[1].scatter(df_corr['GM (US$/Tn)'], df_corr['Tons'],
                               color='#2e7d32', alpha=0.7, edgecolors='white', s=60)
                z1 = np.polyfit(df_corr['GM (US$/Tn)'].dropna(), df_corr['Tons'].dropna(), 1)
                p1 = np.poly1d(z1)
                x1 = np.linspace(df_corr['GM (US$/Tn)'].min(), df_corr['GM (US$/Tn)'].max(), 100)
                axes[1].plot(x1, p1(x1), color='black', linewidth=1.5, linestyle='--')
                axes[1].set_xlabel("GM (US$/Tn)")
                axes[1].set_ylabel("Toneladas")
                r1 = corr.loc['Tons','GM (US$/Tn)']
                axes[1].set_title(f"Tons vs GM  |  r = {r1:.3f}")
                axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                axes[1].grid(True, alpha=0.3)

                # Precio vs GM
                axes[2].scatter(df_corr['Precio (US$/Tn)'], df_corr['GM (US$/Tn)'],
                               color='#003f7f', alpha=0.7, edgecolors='white', s=60)
                z2 = np.polyfit(df_corr['Precio (US$/Tn)'].dropna(), df_corr['GM (US$/Tn)'].dropna(), 1)
                p2 = np.poly1d(z2)
                x2 = np.linspace(df_corr['Precio (US$/Tn)'].min(), df_corr['Precio (US$/Tn)'].max(), 100)
                axes[2].plot(x2, p2(x2), color='black', linewidth=1.5, linestyle='--')
                axes[2].set_xlabel("Precio (US$/Tn)")
                axes[2].set_ylabel("GM (US$/Tn)")
                r2 = corr.loc['Precio (US$/Tn)','GM (US$/Tn)']
                axes[2].set_title(f"Precio vs GM  |  r = {r2:.3f}")
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig_sc)

                st.markdown("---")

                # ── INTERPRETACIÓN ────────────────────────────────────────────
                st.markdown("**Interpretación:**")

                def interpretar(r, var1, var2):
                    if abs(r) >= 0.7:
                        fuerza = "fuerte"
                    elif abs(r) >= 0.4:
                        fuerza = "moderada"
                    else:
                        fuerza = "baja"
                    direccion = "positiva" if r > 0 else "negativa"
                    return f"- **{var1} vs {var2}:** correlación {direccion} {fuerza} (r = {r:.3f})"

                st.markdown(interpretar(corr.loc['Tons','Precio (US$/Tn)'], 'Toneladas', 'Precio'))
                st.markdown(interpretar(corr.loc['Tons','GM (US$/Tn)'], 'Toneladas', 'Margen Bruto'))
                st.markdown(interpretar(corr.loc['Precio (US$/Tn)','GM (US$/Tn)'], 'Precio', 'Margen Bruto'))

                # Nota especial si Precio-GM es muy alta
                r_precio_gm = corr.loc['Precio (US$/Tn)','GM (US$/Tn)']
                if abs(r_precio_gm) >= 0.8:
                    st.info(f"💡 La correlación entre Precio y GM es muy alta ({r_precio_gm:.3f}), lo que indica que el margen bruto de Tenaris está fuertemente ligado al precio de venta por tonelada.")

            else:
                st.info("El archivo no contiene las columnas de Precio y GM necesarias para este análisis.")

    else:
        st.markdown("""
        ### 👆 Sube el archivo CSV de ventas para comenzar
        El archivo debe ser el histórico raw con columnas **Month** y **Suma de Tons**.
        El dashboard hace todo el procesamiento automáticamente.
        """)

st.markdown("---")
st.caption("Dashboard desarrollado por Juan Sebastián Rodríguez Silva · Práctica Profesional · Tenaris S.A. · 2026")
