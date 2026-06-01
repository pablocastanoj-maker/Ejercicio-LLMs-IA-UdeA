import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def seleccion_features_varianza(X, y, threshold, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X_train)
    X_train_filt = vt.transform(X_train)
    X_test_filt = vt.transform(X_test)

    if X_train_filt.shape[1] == 0:
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X_train)
        X_train_filt = vt.transform(X_train)
        X_test_filt = vt.transform(X_test)

    dt_full = DecisionTreeClassifier(random_state=random_state)
    dt_full.fit(X_train, y_train)
    acc_full = accuracy_score(y_test, dt_full.predict(X_test))

    dt_filt = DecisionTreeClassifier(random_state=random_state)
    dt_filt.fit(X_train_filt, y_train)
    acc_filt = accuracy_score(y_test, dt_filt.predict(X_test_filt))

    return {
        "n_features_original": int(X.shape[1]),
        "n_features_filtrado": int(X_train_filt.shape[1]),
        "features_eliminados": int(X.shape[1] - X_train_filt.shape[1]),
        "acc_original": round(float(acc_full), 6),
        "acc_filtrado": round(float(acc_filt), 6),
        "diferencia_acc": round(float(acc_filt - acc_full), 6),
    }
