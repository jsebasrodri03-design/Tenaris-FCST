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

# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Tenaris_logo.svg/320px-Tenaris_logo.svg.png", width=180)
with col_title:
    st.markdown("## Dashboard de Proyección de Demanda")
    st.markdown("**Tenaris S.A. – Business Coordination**")

st.divider()
st.markdown("""
Este dashboard permite visualizar el forecast validado del modelo de pronóstico de ventas
y generar nuevas proyecciones con datos actualizados.

> 📊 El modelo seleccionado (Prophet) presenta un error promedio (**MAPE ≈ 11%**),
> con cobertura de intervalos del **91.7%** y sesgo de **-5.3%**.
""")

st.markdown("---")
modo = st.radio(
    "**Selecciona el modo de uso:**",
    ["📁 Ver forecast validado (2026)", "🔄 Generar nuevo forecast con datos actualizados"],
    horizontal=True
)
st.markdown("---")

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
        promedio = total / 12
        df['Mes'] = df['ds'].dt.strftime('%B %Y')
        df['Estado'] = df['yhat'].apply(
            lambda x: '🟢 Alto' if x > promedio * 1.05
            else ('🔴 Bajo' if x < promedio * 0.95 else '🟡 Normal')
        )
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

        # TABLA
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
# MODO 2 — NUEVO FORECAST CON PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.info("Sube el archivo CSV con el histórico de ventas (el mismo archivo raw que usas en Colab)")
    uploaded_file = st.file_uploader("📂 Cargar histórico de ventas", type=["csv", "xlsx"])

    if uploaded_file:

        # ── PASO 1: LEER ARCHIVO ──────────────────────────────────────────────
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file)
            else:
                try:
                    df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig', sep=None, engine='python')
                except Exception:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding='latin1', sep=None, engine='python')
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")
            st.stop()

        df_raw.columns = df_raw.columns.str.strip().str.replace('\ufeff', '', regex=False)

        # ── PASO 2: LIMPIAR Y PREPARAR ────────────────────────────────────────
        MONTH = 'Month'
        TONS  = 'Suma de Tons'

        if MONTH not in df_raw.columns or TONS not in df_raw.columns:
            st.error(f"No se encontraron las columnas '{MONTH}' y '{TONS}'. Columnas disponibles: {list(df_raw.columns)}")
            st.stop()

        df = df_raw.copy()

        # Limpiar strings HTML
        for c in df.select_dtypes(include='object').columns:
            df[c] = df[c].astype(str).str.strip()

        # Convertir fechas y numéricos
        df[MONTH] = pd.to_datetime(df[MONTH], errors='coerce', dayfirst=True)
        df[TONS]  = df[TONS].apply(clean_numeric)
        df = df.dropna(subset=[MONTH, TONS])

        # ── PASO 3: AGRUPAR POR MES ───────────────────────────────────────────
        ts = (df.groupby(MONTH)[TONS]
                .sum()
                .sort_index())

        df_prophet = ts.reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

        st.success(f"✅ Serie lista: **{len(df_prophet)} meses** | Rango: {df_prophet['ds'].min().strftime('%Y-%m')} → {df_prophet['ds'].max().strftime('%Y-%m')}")

        if len(df_prophet) < 12:
            st.warning("⚠️ Se recomienda al menos 24 meses de histórico para un pronóstico confiable.")

        # ── PASO 4: ENTRENAR PROPHET ──────────────────────────────────────────
        with st.spinner("🔄 Entrenando modelo Prophet..."):
            M = 12
            cps_options = [0.02, 0.05, 0.10]
            best_mape = np.inf
            best_cps  = 0.02

            # Grid search rápido de changepoint_prior_scale
            for cps in cps_options:
                m = Prophet(
                    growth='linear',
                    seasonality_mode='multiplicative',
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=cps
                )
                m.fit(df_prophet)
                fc = m.make_future_dataframe(periods=M, freq='MS')
                fc = m.predict(fc)
                yhat = fc.set_index('ds')['yhat'].reindex(df_prophet.set_index('ds').iloc[-M:].index)
                mape = mape_safe(df_prophet['y'].iloc[-M:].values, yhat.values)
                if mape < best_mape:
                    best_mape = mape
                    best_cps  = cps

            # Modelo final con mejor cps
            model_final = Prophet(
                growth='linear',
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=best_cps
            )
            model_final.fit(df_prophet)
            future_final   = model_final.make_future_dataframe(periods=M, freq='MS')
            forecast_final = model_final.predict(future_final)

        fc_2026 = forecast_final.tail(M)[['ds','yhat','yhat_lower','yhat_upper']].copy()
        total   = fc_2026['yhat'].sum()
        low     = fc_2026['yhat_lower'].sum()
        high    = fc_2026['yhat_upper'].sum()

        # ── PASO 5: MOSTRAR RESULTADOS ────────────────────────────────────────
        st.subheader("📈 Nuevo forecast generado")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Demanda estimada (12m)", f"{round(total):,} tons")
        col2.metric("Escenario bajo",         f"{round(low):,} tons")
        col3.metric("Escenario alto",         f"{round(high):,} tons")
        col4.metric("MAPE backtest",          f"≈ {round(best_mape, 1)}%")

        st.caption(f"Mejor changepoint_prior_scale: {best_cps}")

        st.markdown("---")

        # SEMÁFORO
        st.subheader("🚦 Semáforo mensual")
        promedio = total / 12
        fc_2026['Mes'] = fc_2026['ds'].dt.strftime('%B %Y')
        fc_2026['Estado'] = fc_2026['yhat'].apply(
            lambda x: '🟢 Alto' if x > promedio * 1.05
            else ('🔴 Bajo' if x < promedio * 0.95 else '🟡 Normal')
        )
        sem = fc_2026[['Mes','yhat','Estado']].copy()
        sem.columns = ['Mes','Pronóstico (tons)','Estado']
        sem['Pronóstico (tons)'] = sem['Pronóstico (tons)'].apply(lambda x: f"{round(x):,}")
        st.dataframe(sem, use_container_width=True, hide_index=True)

        st.markdown("---")

        # GRÁFICO HISTÓRICO + FORECAST
        st.subheader("Histórico + Proyección")
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

        # TABLA
        st.subheader("Valores proyectados")
        tabla = fc_2026[['Mes','yhat','yhat_lower','yhat_upper']].copy()
        tabla.columns = ['Mes','Pronóstico (tons)','Escenario bajo','Escenario alto']
        st.dataframe(tabla.style.format({'Pronóstico (tons)': '{:,.0f}', 'Escenario bajo': '{:,.0f}', 'Escenario alto': '{:,.0f}'}),
                     use_container_width=True, hide_index=True)

        st.markdown("---")

        # DESCARGA
        st.subheader("⬇️ Descargar resultados")
        export = fc_2026[['ds','yhat','yhat_lower','yhat_upper']].copy()
        export.columns = ['Fecha','Pronóstico (tons)','Escenario bajo','Escenario alto']
        export['Fecha'] = export['Fecha'].dt.strftime('%Y-%m-%d')
        download_excel(export, "Forecast_Tenaris_Actualizado.xlsx")

        st.markdown("---")

        # ESTACIONALIDAD
        st.subheader("Patrón estacional del histórico")
        df_prophet['mes'] = df_prophet['ds'].dt.month
        meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        prom = df_prophet.groupby('mes')['y'].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(meses, prom.values, color='#003f7f', alpha=0.8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax2.set_title("Promedio histórico de demanda por mes")
        ax2.set_ylabel("Toneladas promedio")
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig2)

    else:
        st.markdown("""
        ### 👆 Sube el archivo CSV de ventas para comenzar

        El archivo debe ser el histórico raw con columnas **Month** y **Suma de Tons**.
        El dashboard hace todo el procesamiento automáticamente.
        """)

st.markdown("---")
st.caption("Dashboard desarrollado por Juan Sebastián Rodríguez Silva · Práctica Profesional · Tenaris S.A. · 2026")
