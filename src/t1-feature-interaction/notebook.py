import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    from synthetic_data_generator import SyntheticDataGenerator

    return SyntheticDataGenerator, alt, mo, pd


@app.cell
def _(mo):
    mo.md(r"""# T1 - Feature Interaction""")
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
            "Spatial Manhattan": "spatial_manhattan"   
        }, value="Polynominal Quadratic", label="Type"
    )


    n_samples = mo.ui.radio(
        options={
            "200": 200, 
            "1000": 1000, 
            "5000": 5000
        }, value="1000", label="samples"
    )

    n_features = mo.ui.radio(
        options={
            "2": 2, 
            "3": 3, 
            "4": 4
        }, value="2", label="features"
    )

    mo.hstack([dataset_type, n_samples, n_features])
    return dataset_type, n_features, n_samples


@app.cell
def _(SyntheticDataGenerator, alt, dataset_type, n_features, n_samples, pd):
    generator = SyntheticDataGenerator()

    X_df, y_series, feature_info = generator.generate_dataset(
        dataset_type.value, 
        n_samples.value, 
        n_features.value, 
        return_df=True
    )
    df_combined = pd.concat([X_df, y_series], axis=1)

    chart = alt.Chart(df_combined).mark_point().encode(
        x=X_df.columns[0],
        y=X_df.columns[1],
        color=alt.Color("target:N")
    ).properties(title=dataset_type.value)

    chart    
    return X_df, chart, df_combined, feature_info, generator, y_series


if __name__ == "__main__":
    app.run()
