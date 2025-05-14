import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, CatBoostRegressor
from tabpfn import TabPFNClassifier, TabPFNRegressor


class ModelEvaluator:
    def __init__(self, random_state=42):
        self.random_state = random_state

        self.classifier_models = {
            "TabPFN": lambda: TabPFNClassifier(),
            "CatBoost": lambda: CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                random_state=random_state,
            ),
            "LogisticRegression": lambda: LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
        }

        self.regressor_models = {
            "TabPFN": lambda: TabPFNRegressor(),
            "RandomForest": lambda: RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=random_state
            ),
            "CatBoostRegressor": lambda: CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                random_state=random_state,
            ),
            "LinearRegression": lambda: LinearRegression(),
        }

    def evaluateClassifier(self, X, y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        results = []
        for model_name, model_fn in self.classifier_models.items():
            model = model_fn()
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            try:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except (ValueError, IndexError):
                auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
            accuracy = accuracy_score(y_test, y_pred)
            results.append(
                {
                    "model": model_name,
                    "auc": auc,
                    "accuracy": accuracy,
                }
            )
        return results

    def evaluateRegressor(self, X, y):
        """
        Evaluiert verschiedene Regressionsmodelle auf dem gegebenen Datensatz.

        Args:
            X: Feature-Matrix
            y: Zielwerte (kontinuierlich)

        Returns:
            results: Liste mit Performance-Metriken für jedes Modell
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        results = []

        for model_name, model_fn in self.regressor_models.items():
            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append(
                {"model": model_name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
            )

        return results
