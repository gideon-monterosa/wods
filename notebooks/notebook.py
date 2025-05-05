import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt

    from feature_interaction.synthetic_data_generator import SyntheticDataGenerator
    from feature_interaction.model_evaluator import ModelEvaluator

    return ModelEvaluator, SyntheticDataGenerator, alt, mo, pd


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
    dataset_type = mo.ui.radio(
        options={
            "Polynominal Quadratic": "polynomial_quadratic",
            "Polynomial Cubic": "polynomial_cubic",
            "Logical xor": "logical_xor",
            "Logical and": "logical_and",
            "Logical or": "logical_or",
            "Conditional Threshold": "conditional_threshold",
            "Spatial Euclidean": "spatial_euclidean",
            "Spatial Manhattan": "spatial_manhattan",
        },
        value="Polynominal Quadratic",
        label="Type",
    )

    n_samples = mo.ui.radio(
        options={"200": 200, "1000": 1000, "5000": 5000}, value="1000", label="samples"
    )

    n_features = mo.ui.radio(
        options={"2": 2, "3": 3, "4": 4}, value="2", label="features"
    )

    irrelevant_ratio = mo.ui.radio(
        options={"1:0": 0, "1:1": 1, "1:2": 2, "1:5": 5}, value="1:0", label="irrelevant features ratio"
    )

    noise_level = mo.ui.radio(
        options={"0%": 0.0, "10%": 0.1, "20%": 0.2}, value="0%", label="noise level"
    )

    mo.hstack([dataset_type, n_samples, n_features, irrelevant_ratio, noise_level])
    return dataset_type, irrelevant_ratio, n_features, n_samples, noise_level


@app.cell
def _(
    SyntheticDataGenerator,
    alt,
    dataset_type,
    irrelevant_ratio,
    n_features,
    n_samples,
    noise_level,
):
    generator = SyntheticDataGenerator()

    df, X, y, _ = generator.generate_dataset(
        dataset_type.value,
        n_samples.value,
        n_features.value,
        irrelevant_ratio.value,
        noise_level.value,
    )

    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(x=df.columns[0], y=df.columns[1], color=alt.Color("target:N"))
        .properties(title=dataset_type.value)
    )

    chart
    return X, chart, df, generator, y


@app.cell
def _(mo):
    mo.md(r"""## Evaluate""")
    return


@app.cell
def _(ModelEvaluator, X, mo, y):
    model_evaluator = ModelEvaluator()
    output = {}

    def run_evaluation(_):
        output, overall_strengths = model_evaluator.evaluate(X=X, y=y)

        mo.output.append(button)
        mo.output.append(output)

    button = mo.ui.button(label="Run Evaluation", on_click=run_evaluation)
    return button, model_evaluator, output, run_evaluation


if __name__ == "__main__":
    app.run()
