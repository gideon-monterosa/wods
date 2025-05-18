import os
import pandas as pd
import numpy as np
from feature_interaction import model_evaluator
from feature_interaction.model_evaluator import ModelEvaluator

data_dir = "./data/"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
csv_files.sort()

model_evaluator = ModelEvaluator()

for csv_file in csv_files:
    filepath = os.path.join(data_dir, csv_file)
    df = pd.read_csv(filepath)

    X = df.drop("target", axis=1).values
    y = df["target"].values

    results = model_evaluator.evaluateRegressor(X, y)

    print(f"\n{'='*80}")
    print(f"Auswertung für Datensatz: {csv_file}")
    print(f"{'='*80}")

    print(f"{'Modell':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print(f"{'-'*60}")

    for result in results:
        model_name = result["model"]
        mse = result["mse"]
        rmse = result["rmse"]
        mae = result["mae"]
        r2 = result["r2"]

        print(f"{model_name:<20} {mse:<12.4f} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")
