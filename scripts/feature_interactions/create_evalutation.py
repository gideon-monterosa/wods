import os
import pandas as pd
import numpy as np
from feature_interaction.model_evaluator import ModelEvaluator
from tqdm import tqdm

data_dir = "./data/synthetic_datasets/"

results_dir = "./data/results/feature_interactions/"
os.makedirs(results_dir, exist_ok=True)

csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
csv_files.sort()

model_evaluator = ModelEvaluator()

for csv_file in tqdm(csv_files, desc="Datensätze verarbeiten"):
    filepath = os.path.join(data_dir, csv_file)
    df = pd.read_csv(filepath)

    X = df.drop("target", axis=1).values
    y = df["target"].values

    print(f"\nEvaluiere Datensatz: {csv_file}...")
    results = model_evaluator.evaluateRegressor(X, y)

    print(f"\n{'='*80}")
    print(f"Auswertung für Datensatz: {csv_file}")
    print(f"{'='*80}")
    print(
        f"{'Modell':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MSE Std':<12} {'MAE Std':<12} {'R² Std':<12}"
    )
    print(f"{'-'*100}")

    for result in results:
        model_name = result["model"]
        mse = result["mse"]
        rmse = result["rmse"]
        mae = result["mae"]
        r2 = result["r2"]
        mse_std = result.get("mse_std", 0)
        mae_std = result.get("mae_std", 0)
        r2_std = result.get("r2_std", 0)

        print(
            f"{model_name:<20} {mse:<12.4f} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f} {mse_std:<12.4f} {mae_std:<12.4f} {r2_std:<12.4f}"
        )

    results_df = pd.DataFrame(results)

    dataset_info = {
        "dataset": csv_file,
        "n_samples": len(df),
        "n_features": X.shape[1],
    }

    parts = csv_file.replace(".csv", "").split("_")
    for part in parts:
        if part.startswith("n"):
            dataset_info["n_samples_name"] = part
        elif part.startswith("noise"):
            dataset_info["noise_level"] = part
        elif part.startswith("irrelevant"):
            dataset_info["irrelevant_ratio"] = part

    for col, val in dataset_info.items():
        results_df[col] = val

    result_filename = f"{results_dir}cv_results_{csv_file.replace('.csv', '')}.csv"
    results_df.to_csv(result_filename, index=False)
    print(f"Ergebnisse gespeichert unter: {result_filename}")

print("\nAlle Auswertungen abgeschlossen!")
