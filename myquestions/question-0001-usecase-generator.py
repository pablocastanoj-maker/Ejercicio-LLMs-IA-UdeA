import pandas as pd
import numpy as np

def generar_caso_de_uso_limpiar_sensores():

    n = np.random.randint(5, 12)
    print("Filas generadas:", n)

    data = np.random.randn(n)

    # insertar NaNs
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(0, n)
        print("NaN en posición:", idx)
        data[idx] = np.nan

    df = pd.DataFrame({"sensor": data})

    # duplicar filas
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    print("Se agregaron duplicados")

    # OUTPUT esperado
    df_clean = df.interpolate().ffill().bfill().drop_duplicates()

    input_dict = {"df": df}
    output = df_clean

    print("Input generado:\n", df.head())
    print("Output esperado:\n", df_clean.head())

    return input_dict, output
