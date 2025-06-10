import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Pronóstico de Demanda", layout="wide")
st.title("📈 Aplicación de Pronóstico de Demanda")

uploaded_file = st.file_uploader("Carga tu archivo Excel con las ventas", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.drop_duplicates(subset=["FECHA REAL", "PRODUCTO"], keep="first")
    df_pivot = df.pivot(index="FECHA REAL", columns="PRODUCTO", values="VENTAS")

    def seleccionar_mejor_modelo(serie):
        train_size = int(len(serie) * 0.7)
        train, test = serie[:train_size], serie[train_size:]

        modelos = {}

        try:
            arima_model = sm.tsa.ARIMA(train, order=(1,1,1)).fit()
            arima_rmse = sqrt(mean_squared_error(test, arima_model.forecast(steps=len(test))))
            modelos["ARIMA"] = (arima_model, arima_rmse)
        except:
            modelos["ARIMA"] = (None, np.inf)

        try:
            sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
            sarima_rmse = sqrt(mean_squared_error(test, sarima_model.forecast(steps=len(test))))
            modelos["SARIMA"] = (sarima_model, sarima_rmse)
        except:
            modelos["SARIMA"] = (None, np.inf)

        try:
            hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12).fit()
            hw_rmse = sqrt(mean_squared_error(test, hw_model.forecast(len(test))))
            modelos["Holt-Winters"] = (hw_model, hw_rmse)
        except:
            modelos["Holt-Winters"] = (None, np.inf)

        try:
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(np.arange(len(train)).reshape(-1, 1), train)
            rf_rmse = sqrt(mean_squared_error(test, rf_model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))))
            modelos["Random Forest"] = (rf_model, rf_rmse)
        except:
            modelos["Random Forest"] = (None, np.inf)

        mejor_modelo = min(modelos, key=lambda x: modelos[x][1])
        mejor_modelo_entrenado = modelos[mejor_modelo][0]

        if mejor_modelo == "Random Forest":
            pronostico = mejor_modelo_entrenado.predict(np.arange(len(serie), len(serie) + 3).reshape(-1, 1))
        elif mejor_modelo_entrenado:
            pronostico = mejor_modelo_entrenado.forecast(steps=3)
        else:
            pronostico = [np.nan] * 3

        pronostico = [int(round(p)) if not np.isnan(p) else np.nan for p in pronostico]

        return mejor_modelo, pronostico

    resultados = {}
    for producto in df_pivot.columns:
        mejor_modelo, pronostico = seleccionar_mejor_modelo(df_pivot[producto].dropna())
        resultados[producto] = {"Mejor Modelo": mejor_modelo, "Pronóstico": pronostico}

    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')

    st.subheader("📊 Resultados del Pronóstico")
    st.dataframe(df_resultados)

    df_resultados["Total Pronóstico"] = df_resultados["Pronóstico"].apply(lambda x: sum(x) if isinstance(x, list) else 0)
    top15 = df_resultados.sort_values("Total Pronóstico", ascending=False).head(15)

    st.subheader("🏆 Top 15 Productos con Mayor Demanda Pronosticada")
    st.dataframe(top15[["Mejor Modelo", "Total Pronóstico"]])

    st.subheader("📈 Tendencia de Pronóstico - Top 15")
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = plt.cm.tab20.colors
    for i, (producto, fila) in enumerate(top15.iterrows()):
        if isinstance(fila['Pronóstico'], list):
            ax.plot(["Julio", "Agosto", "Septiembre"], fila['Pronóstico'], label=producto, color=colores[i % len(colores)])

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel("Unidades Pronosticadas")
    st.pyplot(fig)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_resultados.to_excel(writer, sheet_name="Pronostico")
    output.seek(0)
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=output,
        file_name="Pronostico_Demanda.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
