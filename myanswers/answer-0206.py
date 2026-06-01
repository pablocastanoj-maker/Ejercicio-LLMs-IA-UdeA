
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def mean_cv_mse(X: pd.DataFrame, y: np.ndarray, k: int) -> float:
    model = LinearRegression()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_list.append(mean_squared_error(y_test, y_pred))
    return np.mean(mse_list)
