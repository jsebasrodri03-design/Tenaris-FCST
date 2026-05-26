import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

st.set_page_config(
    page_title="Tenaris – Forecast de Demanda",
    page_icon="📊",
    layout="wide"
)

# -----------------------
# HEADER
# -----------------------
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

# -----------------------
# MODO DE USO
# -----------------------
st.markdown("---")
modo = st.radio(
    "**Selecciona el modo de uso:**",
    ["📁 Ver forecast validado (2026)", "🔄 Generar nuevo forecast con datos actualizados"],
    horizontal=True
)

st.markdown("---")

# -----------------------
# CARGA ARCHIVO
# -----------------------
if modo == "📁 Ver forecast validado (2026)":
    st.info("Sube el archivo CSV con los resultados del modelo (forecast_2026.csv)")
    uploaded_file = st.file_uploader("📂 Cargar archivo de resultados", type=["csv"])
else:
    st.info("Sube el archivo Excel con el histórico de ventas para generar un nuevo forecast")
    uploaded_file = st.file_uploader("📂 Cargar histórico de ventas", type=["xlsx", "csv"])

# -----------------------
# EJECUCIÓN
# -----------------------
if uploaded_file:

    # =========================================
    # MODO 1 — FORECAST VALIDADO
    # =========================================
    if modo == "📁 Ver forecast validado (2026)":

        df = pd.read_csv(uploaded_file)

        st.subheader("📈 Forecast validado – Modelo Prophet")

        # MÉTRICAS
        total = df['yhat'].sum()
        low   = df['yhat_lower'].sum()
        high  = df['yhat_upper'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Demanda esperada 2026", f"{round(total):,} tons")
        col2.metric("Escenario bajo",        f"{round(low):,} tons")
        col3.metric("Escenario alto",        f"{round(high):,} tons")
        col4.metric("MAPE del modelo",       "≈ 11%")

        st.markdown("---")

        # GRÁFICO
        st.subheader("Proyección mensual 2026")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ds'], df['yhat'], marker='o', linewidth=2,
                color='#003f7f', label='Escenario base')
        ax.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'],
                        alpha=0.15, color='#003f7f', label='Rango de incertidumbre')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_xlabel("Mes")
        ax.set_ylabel("Toneladas")
        ax.set_title("Pronóstico de demanda – Tenaris S.A. 2026", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # TABLA
        st.subheader("Tabla de valores proyectados")
        st.dataframe(df, use_container_width=True)

        # CALIDAD
        st.markdown("---")
        st.subheader("Calidad del modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", "≈ 11%", help="Error porcentual absoluto medio")
        c2.metric("Cobertura de intervalos", "91.7%", help="% de valores reales dentro del rango")
        c3.metric("Sesgo", "-5.3%", help="Tendencia leve a subestimar la demanda")

        st.success("""
        El modelo fue validado mediante backtesting sobre los últimos 12 meses del histórico,
        superando a los modelos Holt-Winters (16%), Naive (20%) y SARIMA (25%).
        """)

    # =========================================
    # MODO 2 — NUEVO FORECAST DINÁMICO
    # =========================================
    else:

        if uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, header=2)
        else:
            df_raw = pd.read_csv(uploaded_file)

        df_raw.columns = df_raw.columns.str.strip()

        # AUTO DETECCIÓN COLUMNAS
        fecha_col  = next((c for c in df_raw.columns if "month" in c.lower()), None)
        ventas_col = next((c for c in df_raw.columns if "ton" in c.lower()), None)

        if not fecha_col or not ventas_col:
            st.error("No se pudo detectar las columnas de fecha o toneladas. "
                     "Verifica que el archivo tenga columnas con 'month' y 'ton' en el nombre.")
            st.stop()

        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(df_raw[fecha_col], errors='coerce')
        df['y']  = pd.to_numeric(df_raw[ventas_col], errors='coerce')
        df = df.dropna(subset=['ds','y'])
        df = df.groupby('ds')['y'].sum().reset_index()

        st.write(f"✅ Datos cargados: **{len(df)} meses** de histórico")

        if len(df) < 12:
            st.warning("Se recomienda tener al menos 12 meses de histórico para un pronóstico confiable.")

        if len(df) < 2:
            st.error("No hay suficientes datos para generar un forecast.")
            st.stop()

        # MODELO
        with st.spinner("Entrenando modelo Prophet y generando forecast..."):
            model = Prophet(changepoint_prior_scale=0.02, yearly_seasonality=True)
            model.fit(df)
            future   = model.make_future_dataframe(periods=12, freq='MS')
            forecast = model.predict(future)

        forecast_12 = forecast.tail(12).copy()
        total = forecast_12['yhat'].sum()

        st.subheader("📈 Nuevo forecast generado")
        col1, col2, col3 = st.columns(3)
        col1.metric("Demanda estimada (12m)", f"{round(total):,} tons")
        col2.metric("Promedio mensual",       f"{round(total/12):,} tons")
        col3.metric("Periodos proyectados",   "12 meses")

        st.markdown("---")

        # GRÁFICO COMPLETO
        st.subheader("Histórico + Proyección")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ds'], df['y'], color='gray', linewidth=1.5, label='Histórico')
        ax.plot(forecast['ds'], forecast['yhat'], color='#003f7f',
                linewidth=2, label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        alpha=0.15, color='#003f7f', label='Intervalo de confianza')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_xlabel("Mes")
        ax.set_ylabel("Toneladas")
        ax.set_title("Pronóstico de demanda – Tenaris S.A.", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # TABLA
        st.subheader("Valores proyectados")
        tabla = forecast_12[['ds','yhat','yhat_lower','yhat_upper']].copy()
        tabla.columns = ['Mes','Pronóstico (tons)','Escenario bajo','Escenario alto']
        tabla = tabla.set_index('Mes')
        st.dataframe(tabla.style.format("{:,.0f}"), use_container_width=True)

        st.markdown("---")

        # PATRÓN ESTACIONAL
        st.subheader("Patrón estacional del histórico")
        df['mes'] = df['ds'].dt.month
        meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        promedio = df.groupby('mes')['y'].mean()

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars = ax2.bar(meses, promedio.values, color='#003f7f', alpha=0.8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax2.set_title("Promedio histórico de demanda por mes")
        ax2.set_ylabel("Toneladas promedio")
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig2)

else:
    st.markdown("""
    ### 👆 Sube un archivo para comenzar

    **Modo 1 – Forecast validado:** sube el archivo `forecast_2026.csv` con las columnas
    `ds`, `yhat`, `yhat_lower`, `yhat_upper`.

    **Modo 2 – Nuevo forecast:** sube el histórico de ventas en Excel (.xlsx)
    con columnas de fecha (`month`) y volumen (`tons`).
    """)

st.markdown("---")
st.caption("Dashboard desarrollado por Juan Sebastián Rodríguez Silva · Práctica Profesional · Tenaris S.A. · 2026")
