import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import numpy as np
    import marimo as mo
    import pandas as pd
    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt

    return alt, mo, np, os, pd, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
        # T1 - Feature Interaction

        ## Generate Syncetic Data
        """
    )
    return


@app.cell
def _(mo):
    selected_dataset = mo.ui.radio(
        options={
            "Complex Dataset (size 200, noise 0%, irrelevant ratio 1:0)": "complex_dataset_n200_noise0_irrelevant0.csv",
            "Complex Dataset (size 1000, noise 0%, irrelevant ratio 1:0)": "complex_dataset_n1000_noise0_irrelevant0.csv",
            "Complex Dataset (size 1000, noise 0%, irrelevant ratio 1:1)": "complex_dataset_n1000_noise0_irrelevant1.csv",
            "Complex Dataset (size 1000, noise 0%, irrelevant ratio 1:2)": "complex_dataset_n1000_noise0_irrelevant2.csv",
            "Complex Dataset (size 1000, noise 0%, irrelevant ratio 1:5)": "complex_dataset_n1000_noise0_irrelevant5.csv",
            "Complex Dataset (size 1000, noise 10%, irrelevant ratio 1:0)": "complex_dataset_n1000_noise10_irrelevant0.csv",
            "Complex Dataset (size 1000, noise 20%, irrelevant ratio 1:0)": "complex_dataset_n1000_noise20_irrelevant0.csv",
            "Complex Dataset (size 5000, noise 0%, irrelevant ratio 1:0)": "complex_dataset_n5000_noise0_irrelevant0.csv",
            "Complex Dataset (size 25000, noise 0%, irrelevant ratio 1:0)": "complex_dataset_n25000_noise0_irrelevant0.csv",
            "Complex Dataset (size 1000, noise 10%, irrelevant ratio 1:2)": "complex_dataset_n1000_noise10_irrelevant2.csv"
        }, 
        label="Selected Dataset"
    )

    selected_dataset
    return (selected_dataset,)


@app.cell
def _(os, pd, selected_dataset):
    data_dir = "./data/synthetic_datasets/"
    results_dir = "./data/results/feature_interactions/"

    def load_dataset(dataset_filename):
        filepath = os.path.join(data_dir, dataset_filename)
        df = pd.read_csv(filepath)
        return df

    def load_evaluation_results(dataset_filename):
        result_filename = f"cv_results_{dataset_filename.replace('.csv', '')}.csv"
        filepath = os.path.join(results_dir, result_filename)
    
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            return pd.DataFrame({"message": ["Keine Evaluierungsergebnisse gefunden. Führe zuerst die Evaluierung durch."]})

    dataset_df = load_dataset(selected_dataset.value)
    evaluation_df = load_evaluation_results(selected_dataset.value)

    return (
        data_dir,
        dataset_df,
        evaluation_df,
        load_dataset,
        load_evaluation_results,
        results_dir,
    )


@app.cell
def _(dataset_df):
    dataset_df
    return


@app.cell
def _(evaluation_df):
    evaluation_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
