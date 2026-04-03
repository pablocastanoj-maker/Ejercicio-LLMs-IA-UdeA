import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_preparar_datos_clasificacion():

    n = np.random.randint(6, 15)
    m = np.random.randint(2, 5)

    print("Filas:", n, "| Features:", m)

    X = np.random.randn(n, m)
    y = np.random.randint(0, 2, n)

    # insertar NaNs
    for _ in range(np.random.randint(1, 5)):
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        print(f"NaN en ({i},{j})")
        X[i, j] = np.nan

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(m)])
    df["target_variable"] = y

    # OUTPUT esperado
    X_data = df.drop(columns=["target_variable"])
    y_data = df["target_variable"].values

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X_data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    input_dict = {
        "df": df,
        "target_col": "target_variable"
    }

    output = (X_scaled, y_data)

    print("Input DF:\n", df.head())
    print("Output X:\n", X_scaled[:3])
    print("Output y:\n", y_data[:3])

    return input_dict, output
