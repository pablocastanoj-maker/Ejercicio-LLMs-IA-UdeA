import pandas as pd
import numpy as np

def generar_caso_de_uso_agrupar_consumo_diario():

    n = np.random.randint(24, 72)
    print("Número de registros:", n)

    fechas = pd.date_range("2024-01-01", periods=n, freq="H")
    consumo = np.random.uniform(10, 100, n)

    df = pd.DataFrame({
        "timestamp": fechas.astype(str),
        "consumo": consumo
    })

    df_temp = df.copy()
    df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
    df_temp["fecha"] = df_temp["timestamp"].dt.date

    result = df_temp.groupby("fecha")["consumo"].sum().reset_index()
    result.columns = ["fecha", "consumo_total"]

    input_dict = {"df": df}
    output = result

    print("Input:\n", df.head())
    print("Output:\n", result.head())

    return input_dict, output
