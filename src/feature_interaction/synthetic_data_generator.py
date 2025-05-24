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

    def generate_complex_synthetic_dataset(
        self, n_samples, irrelevant_ratio=0, noise_level=0.0
    ):
        """
        Erzeugt einen komplexen synthetischen Datensatz für Regressionsaufgaben mit
        verschiedenen Feature-Interaktionen, optionalen irrelevanten Features und Rauschen.

        Args:
            n_samples: Anzahl der zu erzeugenden Datenpunkte
            irrelevant_ratio: Verhältnis von irrelevanten zu relevanten Features (0 bis 1)
            noise_level: Stärke des Rauschens (0 bis 1)
            random_state: Zufallsseed (falls None, wird self.random_state verwendet)

        Returns:
            X: Feature-Matrix
            y: Zielwerte (kontinuierlich)
        """
        np.random.seed(self.random_state)

        n_relevant = 16
        n_irrelevant = int(n_relevant * irrelevant_ratio)
        n_features = n_relevant + n_irrelevant

        X = np.random.uniform(low=-10, high=10, size=(n_samples, n_features))
        y = np.zeros(n_samples)

        # Polynomiale Interaktion (quadratisch)
        poly_interaction = X[:, 0] * X[:, 1] * X[:, 2]
        poly_contribution = 0.0003 * poly_interaction**2

        # Logische Interaktion (XOR-ähnlich)
        logical_features = X[:, 3:6] > 0
        logical_contribution = 100 * (
            (logical_features[:, 0] & logical_features[:, 1] & ~logical_features[:, 2])
            | (
                ~logical_features[:, 0]
                & ~logical_features[:, 1]
                & logical_features[:, 2]
            )
        )

        # Räumliche Interaktion
        center = np.mean(X[:, 6:9], axis=0)
        distances = np.linalg.norm(X[:, 6:9] - center, axis=1)
        alpha = -np.log(1 / 99) / distances.max()
        spatial_contribution = 1 + 150 * np.exp(-alpha * distances)

        # 4. Bedingte Schwellenwert-Interaktion
        cond_A = X[:, 9] > 0.5
        cond_B = X[:, 10] < -0.2
        cond_C = np.abs(X[:, 11]) < 0.3
        cond_D = X[:, 12] > X[:, 9]

        conditional_contribution = np.zeros(n_samples)
        conditional_contribution[cond_A & cond_B] = 40.0
        conditional_contribution[cond_A & ~cond_B & cond_C] = 20.0
        conditional_contribution[~cond_A & cond_D] = -25.0
        conditional_contribution[~cond_A & ~cond_D & cond_C] = 1.0

        # Lineare Effekte
        linear_contribution = 4.8 * X[:, 13] - 2.2 * X[:, 14] + 3.5 * X[:, 15]

        print("poly_contribution:", poly_contribution)
        print("logical_contribution:", logical_contribution)
        print("spatial_contribution:", spatial_contribution)
        print("conditional_contribution:", conditional_contribution)
        print("linear_contribution:", linear_contribution)

        y = (
            poly_contribution
            + logical_contribution
            + spatial_contribution
            + conditional_contribution
            + linear_contribution
        )

        if noise_level > 0:
            y_std = np.std(y)
            noise = np.random.normal(0, y_std * noise_level, n_samples)
            y += noise

        return X, y.astype(int)
