import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def generar_caso_de_uso_entrenar_clasificador():

    n = np.random.randint(20, 50)
    m = np.random.randint(2, 5)

    print("Filas:", n, "| Features:", m)

    X = np.random.randn(n, m)
    y = (X[:, 0] > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(m)])
    df["target"] = y

    split = int(0.8 * n)

    X_data = df.drop(columns=["target"]).values
    y_data = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train = y_data[:split]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    input_dict = {
        "df": df,
        "target_col": "target"
    }

    output = preds

    print("Input DF:\n", df.head())
    print("Predicciones:\n", preds[:5])

    return input_dict, output
