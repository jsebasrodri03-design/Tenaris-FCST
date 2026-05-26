import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io

st.set_page_config(
    page_title="Tenaris – Forecast de Demanda",
    page_icon="📊",
    layout="wide"
)

# HEADER
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

if modo == "📁 Ver forecast validado (2026)":
    st.info("Sube el archivo CSV con los resultados del modelo (forecast_2026.csv)")
    uploaded_file = st.file_uploader("📂 Cargar archivo de resultados", type=["csv"])
    
    st.markdown("*(Opcional)* Sube también el archivo de backtest para ver comparación de modelos:")
    backtest_file = st.file_uploader("📂 Cargar backtest (backtest_prophet_vs_naive_12m.csv)", type=["csv"])
else:
    st.info("Sube el archivo Excel con el histórico de ventas para generar un nuevo forecast")
    uploaded_file = st.file_uploader("📂 Cargar histórico de ventas", type=["xlsx", "csv"])
    backtest_file = None

if uploaded_file:

    if modo == "📁 Ver forecast validado (2026)":

        df = pd.read_csv(uploaded_file)

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

        # SEMÁFORO POR MES
        st.subheader("🚦 Semáforo mensual")
        promedio_mensual = total / 12
        df['ds'] = pd.to_datetime(df['ds'])
        df['Mes'] = df['ds'].dt.strftime('%B %Y')
        df['Estado'] = df['yhat'].apply(
            lambda x: '🟢 Alto' if x > promedio_mensual * 1.05
            else ('🔴 Bajo' if x < promedio_mensual * 0.95 else '🟡 Normal')
        )

        semaforo = df[['Mes', 'yhat', 'Estado']].copy()
        semaforo.columns = ['Mes', 'Pronóstico (tons)', 'Estado']
        semaforo['Pronóstico (tons)'] = semaforo['Pronóstico (tons)'].apply(lambda x: f"{round(x):,}")
        st.dataframe(semaforo, use_container_width=True, hide_index=True)

        st.markdown("---")

        # GRÁFICO FORECAST
        st.subheader("Proyección mensual 2026")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ds'], df['yhat'], marker='o', linewidth=2, color='#003f7f', label='Escenario base')
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

        # GRÁFICO BACKTEST
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
            ax2.set_xlabel("Mes")
            ax2.set_ylabel("Toneladas")
            ax2.set_title("Backtest 12 meses – Real vs Modelos", fontsize=13)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            st.markdown("---")

        # TABLA COMPLETA
        st.subheader("Tabla de valores proyectados")
        tabla_display = df[['Mes', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        tabla_display.columns = ['Mes', 'Pronóstico (tons)', 'Escenario bajo', 'Escenario alto']
        st.dataframe(tabla_display.style.format({'Pronóstico (tons)': '{:,.0f}',
                                                  'Escenario bajo': '{:,.0f}',
                                                  'Escenario alto': '{:,.0f}'}),
                     use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── BOTÓN DESCARGA EXCEL ──────────────────────────────────────────────
        st.subheader("⬇️ Descargar resultados")

        export = df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        export.columns = ['Fecha', 'Pronóstico (tons)', 'Escenario bajo', 'Escenario alto']
        export['Fecha'] = export['Fecha'].dt.strftime('%Y-%m-%d')

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export.to_excel(writer, index=False, sheet_name='Forecast 2026')
        buffer.seek(0)

        st.download_button(
            label="📥 Descargar forecast en Excel",
            data=buffer,
            file_name="Forecast_Tenaris_2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")

        # CALIDAD
        st.subheader("Calidad del modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", "≈ 11%")
        c2.metric("Cobertura de intervalos", "91.7%")
        c3.metric("Sesgo", "-5.3%")
        st.success("El modelo fue validado mediante backtesting sobre los últimos 12 meses, superando a Holt-Winters (16%), Naive (20%) y SARIMA (25%).")

    else:
        if uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, header=2)
        else:
            df_raw = pd.read_csv(uploaded_file)

        df_raw.columns = df_raw.columns.str.strip()
        fecha_col  = next((c for c in df_raw.columns if "month" in c.lower()), None)
        ventas_col = next((c for c in df_raw.columns if "ton" in c.lower()), None)

        if not fecha_col or not ventas_col:
            st.error("No se pudo detectar las columnas de fecha o toneladas.")
            st.stop()

        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(df_raw[fecha_col], errors='coerce')
        df['y']  = pd.to_numeric(df_raw[ventas_col], errors='coerce')
        df = df.dropna(subset=['ds','y'])
        df = df.groupby('ds')['y'].sum().reset_index()

        st.write(f"✅ Datos cargados: **{len(df)} meses** de histórico")

        with st.spinner("Entrenando modelo Prophet..."):
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

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ds'], df['y'], color='gray', linewidth=1.5, label='Histórico')
        ax.plot(forecast['ds'], forecast['yhat'], color='#003f7f', linewidth=2, label='Forecast')
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
        tabla = forecast_12[['ds','yhat','yhat_lower','yhat_upper']].copy()
        tabla.columns = ['Mes','Pronóstico (tons)','Escenario bajo','Escenario alto']
        tabla['Mes'] = tabla['Mes'].dt.strftime('%B %Y')
        st.dataframe(tabla.style.format({'Pronóstico (tons)': '{:,.0f}',
                                          'Escenario bajo': '{:,.0f}',
                                          'Escenario alto': '{:,.0f}'}),
                     use_container_width=True, hide_index=True)

        st.markdown("---")

        # BOTÓN DESCARGA
        st.subheader("⬇️ Descargar resultados")
        export = tabla.copy()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export.to_excel(writer, index=False, sheet_name='Forecast')
        buffer.seek(0)
        st.download_button(
            label="📥 Descargar forecast en Excel",
            data=buffer,
            file_name="Forecast_Tenaris_Actualizado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")

        # ESTACIONALIDAD
        st.subheader("Patrón estacional del histórico")
        df['mes'] = df['ds'].dt.month
        meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        promedio = df.groupby('mes')['y'].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(meses, promedio.values, color='#003f7f', alpha=0.8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax2.set_title("Promedio histórico de demanda por mes")
        ax2.set_ylabel("Toneladas promedio")
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig2)

else:
    st.markdown("""
    ### 👆 Sube un archivo para comenzar

    **Modo 1 – Forecast validado:** sube `forecast_prophet_2026.csv`

    **Modo 2 – Nuevo forecast:** sube el histórico de ventas en Excel (.xlsx)
    """)

st.markdown("---")
st.caption("Dashboard desarrollado por Juan Sebastián Rodríguez Silva · Práctica Profesional · Tenaris S.A. · 2026")
