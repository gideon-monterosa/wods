import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier


class ModelEvaluator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
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

    def evaluate_single_dataset(self, X, y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        results = []

        for model_name, model_fn in self.models.items():
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
