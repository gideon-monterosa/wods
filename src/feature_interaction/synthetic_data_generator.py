import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """
    Generator für synthetische Datensätze mit kontrollierten Feature-Interaktionen.
    """

    def __init__(self, random_state=42):
        """
        Initialisiert den Datengenerator.

        Args:
            random_state: Seed für reproduzierbare Ergebnisse
        """
        self.random_state = random_state
        np.random.seed(random_state)

        self.interaction_types = {
            "polynomial_quadratic": self.polynomial_quadratic,
            "polynomial_cubic": self.polynomial_cubic,
            "logical_xor": self.logical_xor,
            "logical_and": self.logical_and,
            "logical_or": self.logical_or,
            "conditional_threshold": self.conditional_threshold,
            "spatial_euclidean": self.spatial_euclidean,
            "spatial_manhattan": self.spatial_manhattan,
        }

    def polynomial_quadratic(self, X, n_interact=2):
        interaction = X[:, :n_interact].prod(axis=1) ** 2
        return interaction > np.median(interaction)

    def polynomial_cubic(self, X, n_interact=2):
        interaction = X[:, :n_interact].prod(axis=1) ** 3
        return interaction > np.median(interaction)

    def logical_xor(self, X, n_interact=2):
        binary_features = X[:, :n_interact] > 0
        result = binary_features[:, 0]
        for i in range(1, n_interact):
            result = np.logical_xor(result, binary_features[:, i])
        return result

    def logical_and(self, X, n_interact=2):
        binary_features = X[:, :n_interact] > 0
        return np.all(binary_features, axis=1)

    def logical_or(self, X, n_interact=2):
        binary_features = X[:, :n_interact] > 0
        return np.any(binary_features, axis=1)

    def spatial_euclidean(self, X, n_interact=2):
        center = np.zeros(n_interact)
        distances = np.sqrt(np.sum((X[:, :n_interact] - center) ** 2, axis=1))
        return distances < np.median(distances)

    def spatial_manhattan(self, X, n_interact=2):
        center = np.zeros(n_interact)
        distances = np.sum(np.abs(X[:, :n_interact] - center), axis=1)
        return distances < np.median(distances)

    def conditional_threshold(self, X, n_interact=2):
        if n_interact == 2:
            cond_a = X[:, 0] > 0
            cond_b = X[:, 1] > 0

            return np.where(
                cond_a,
                np.where(cond_b, 0, 3),
                np.where(cond_b, 1, 2),
            )

        elif n_interact == 3:
            cond_a = X[:, 0] > 0
            cond_b = X[:, 1] > 0
            cond_c = X[:, 2] > 0.5

            result_high = np.where(
                cond_a,
                np.where(cond_b, 0, 3),
                np.where(cond_b, 1, 2),
            )

            result_low = np.where(
                abs(X[:, 0]) > abs(X[:, 1]),
                np.where(X[:, 0] > 0.5, 0, 2),
                np.where(X[:, 1] > 0.5, 1, 3),
            )

            return np.where(cond_c, result_high, result_low)

        else:
            cond_a = X[:, 0] > 0
            cond_b = X[:, 1] > 0
            cond_c = X[:, 2] > 0.5
            cond_d = X[:, 3] > 0

            result_rule1 = np.where(
                cond_a,
                np.where(cond_b, 0, 3),  # Regel 1: Quadranten
                np.where(cond_b, 1, 2),
            )

            dist = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
            result_rule2 = np.where(
                dist < 1.0,
                np.where(cond_c, 0, 1),  # nahe am Ursprung
                np.where(cond_c, 2, 3),  # weit vom Ursprung
            )

            if X.shape[1] > 4 and n_interact > 4:
                # Nutze Feature 5 als weiteren Selektor
                cond_e = X[:, 4] > 0
                # Dritte Regel: Kombination von Features 1 und 3
                result_rule3 = np.where(
                    X[:, 0] * X[:, 2] > 0,  # gleiche Vorzeichen
                    np.where(cond_d, 0, 2),
                    np.where(cond_d, 1, 3),
                )

                return np.where(
                    cond_e,
                    result_rule1,  # wenn Feature 5 positiv
                    np.where(
                        cond_d, result_rule2, result_rule3
                    ),  # sonst Feature 4 entscheidet
                )

            return np.where(cond_d, result_rule1, result_rule2)

    def generate_dataset(
        self,
        interaction_type,
        n_samples,
        n_interact,
        irrelevant_ratio=0,
        noise_level=0.0,
    ):
        if interaction_type not in self.interaction_types:
            raise ValueError(
                f"Unbekannter Interaktionstyp: {interaction_type}. "
                f"Verfügbare Typen: {list(self.interaction_types.keys())}"
            )

        n_irrelevant = int(n_interact * irrelevant_ratio)
        n_features = n_interact + n_irrelevant

        X = np.random.randn(n_samples, n_features)

        y = self.interaction_types[interaction_type](X, n_interact)

        if noise_level > 0:
            noise_mask = np.random.random(n_samples) < noise_level
            y[noise_mask] = ~y[noise_mask]

        feature_info = {
            "interaction_features": list(range(n_interact)),
            "irrelevant_features": (
                list(range(n_interact, n_features)) if n_irrelevant > 0 else []
            ),
            "interaction_type": interaction_type,
            "n_samples": n_samples,
            "n_interact": n_interact,
            "irrelevant_ratio": irrelevant_ratio,
            "noise_level": noise_level,
        }

        feature_names = []
        for i in range(n_features):
            if i < n_interact:
                feature_names.append(f"F{i+1}_relevant")
            else:
                feature_names.append(f"F{i+1}_irrelevant")

        X_df = pd.DataFrame(X)
        X_df.columns = feature_names
        y_series = pd.Series(y.astype(int), name="target")
        df = pd.concat([X_df, y_series], axis=1)

        return df, X, y.astype(int), feature_info
