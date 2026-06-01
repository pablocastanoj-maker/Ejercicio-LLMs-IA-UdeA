
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def tasa_error_por_clase(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> np.ndarray:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    clases = np.unique(y)
    tasas = np.array([
        np.mean(y_pred[y_test == c] != c)
        for c in clases
    ])

    return tasas
