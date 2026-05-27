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
        changepoint_prior_scale=cps
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

        if MONTH not in df_raw.columns or TONS not in df_raw.columns:
            st.error(f"No se encontraron las columnas '{MONTH}' y '{TONS}'. Columnas disponibles: {list(df_raw.columns)}")
            st.stop()

        df_prophet, df_clean = prepare_series(df_raw)

        st.success(f"✅ Serie lista: **{len(df_prophet)} meses** | Rango: {df_prophet['ds'].min().strftime('%Y-%m')} → {df_prophet['ds'].max().strftime('%Y-%m')}")

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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Forecast Total",
            "🏭 Por Familia",
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
            fc_sem = semaforo_df(fc_12)
            fc_sem['Mes'] = fc_sem['ds'].dt.strftime('%B %Y')
            sem = fc_sem[['Mes','yhat','Estado']].copy()
            sem.columns = ['Mes','Pronóstico (tons)','Estado']
            sem['Pronóstico (tons)'] = sem['Pronóstico (tons)'].apply(lambda x: f"{round(x):,}")
            st.dataframe(sem, use_container_width=True, hide_index=True)

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

                familias = sorted(df_clean[FAMILY].dropna().unique())
                familia_sel = st.multiselect("Selecciona familias:", familias, default=familias[:3] if len(familias) >= 3 else familias)

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

                ts_precio = df_clean.groupby('Month')[PRICE].mean().sort_index()
                ts_gm     = df_clean.groupby('Month')[GM].mean().sort_index()
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

                # Correlaciones
                st.markdown("**Correlación entre variables**")
                df_corr = pd.DataFrame({
                    'Tons': ts_tons,
                    'Precio (US$/Tn)': ts_precio,
                    'GM (US$/Tn)': ts_gm
                }).dropna()

                corr = df_corr.corr().round(3)
                st.dataframe(corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
                             use_container_width=True)

                # Interpretación automática
                corr_tons_precio = corr.loc['Tons', 'Precio (US$/Tn)']
                corr_tons_gm     = corr.loc['Tons', 'GM (US$/Tn)']

                st.markdown("**Interpretación:**")
                if abs(corr_tons_precio) > 0.5:
                    direccion = "positiva" if corr_tons_precio > 0 else "negativa"
                    st.write(f"- Existe una correlación **{direccion} moderada-alta ({corr_tons_precio})** entre las toneladas vendidas y el precio unitario.")
                else:
                    st.write(f"- La correlación entre toneladas y precio es **baja ({corr_tons_precio})**, lo que sugiere que el volumen no depende directamente del precio en este período.")

                if abs(corr_tons_gm) > 0.5:
                    direccion = "positiva" if corr_tons_gm > 0 else "negativa"
                    st.write(f"- Existe una correlación **{direccion} moderada-alta ({corr_tons_gm})** entre las toneladas vendidas y el margen bruto.")
                else:
                    st.write(f"- La correlación entre toneladas y margen bruto es **baja ({corr_tons_gm})**.")

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
