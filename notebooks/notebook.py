import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import itertools
    import numpy as np
    import marimo as mo
    import pandas as pd
    import altair as alt
    from PIL import Image
    import seaborn as sns
    from graphviz import Digraph
    import matplotlib.pyplot as plt

    return Digraph, Image, alt, io, itertools, mo, np, os, pd, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
        # T1 - Feature Interaction

        ## Synthetic Dataset
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
            "Complex Dataset (size 1000, noise 10%, irrelevant ratio 1:2)": "complex_dataset_n1000_noise10_irrelevant2.csv"
        }, 
        value="Complex Dataset (size 1000, noise 10%, irrelevant ratio 1:2)",
        label="Selected Dataset"
    )

    # selected_dataset
    return (selected_dataset,)


@app.cell
def _(os, pd):
    data_dir = "./data/synthetic_datasets/"
    results_dir = "./data/results/feature_interactions/"

    dataset = "complex_dataset_n1000_noise10_irrelevant2.csv"

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

    df = load_dataset(dataset)
    evaluation_df = load_evaluation_results(dataset)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    return (
        X,
        data_dir,
        dataset,
        df,
        evaluation_df,
        load_dataset,
        load_evaluation_results,
        results_dir,
        y,
    )


@app.cell
def _(df, np, plt):
    def _():
        f1_vals = np.linspace(df["feature_1"].min(), df["feature_1"].max(), 100)
        f2_vals = np.linspace(df["feature_2"].min(), df["feature_2"].max(), 100)
        f1_grid, f2_grid = np.meshgrid(f1_vals, f2_vals)

        fixed_f3 = 5

        poly_contrib = 0.0003 * (f1_grid * f2_grid * fixed_f3) ** 2

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        surface = ax.plot_surface(f1_grid, f2_grid, poly_contrib,
                                  cmap="viridis", edgecolor='none', alpha=0.9)

        ax.set_title("Polynomiale Interaktion: (f1 × f2 × f3)² · 0.0003 (f3 fixiert)")
        ax.set_xlabel("feature_1")
        ax.set_ylabel("feature_2")
        ax.set_zlabel("Beitrag")

        fig.colorbar(surface, shrink=0.5, aspect=10, label="Beitrag")
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(df, plot_polinomial2):
    def _():
        import numpy as np
        import matplotlib.pyplot as plt

        features = ["feature_1", "feature_2", "feature_3"]
        f1 = df["feature_1"].values
        f2 = df["feature_2"].values
        f3 = df["feature_3"].values
        X_poly = np.stack([f1, f2, f3], axis=1)
        poly_contribution = 0.0003 * (f1 * f2 * f3) ** 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for i, ax in enumerate(axes):
            xi = X_poly[:, i]
            sc = ax.scatter(xi, poly_contribution, c=poly_contribution, cmap="viridis", alpha=0.7)
            coef = np.polyfit(xi, poly_contribution, 2)
            x_fit = np.linspace(xi.min(), xi.max(), 100)
            y_fit = np.polyval(coef, x_fit)
            ax.plot(x_fit, y_fit, color="red", linewidth=2, linestyle="--", label="Trendlinie (quadratisch)")
            ax.set_xlabel(features[i], fontsize=13)
            if i == 0:
                ax.set_ylabel("poly_contribution", fontsize=13)
            ax.set_title(f"{features[i]} vs. poly_contribution")
            ax.legend()

        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.03)
        cbar.set_label("poly_contribution", fontsize=13)
        plt.suptitle("Polynomiale Beziehung: Feature vs. Beitrag mit quadratischer Trendlinie", fontsize=16)
        return plt.gca()

    plot_polinomial2()
    return


@app.cell
def _(itertools, np, plt):
    def _():
        combinations = np.array(list(itertools.product([0, 1], repeat=3)))
        labels = [f"{a}{b}{c}" for a, b, c in combinations]
        # Die Logik von deinem Datensatz:
        contrib = ((combinations[:,0] & combinations[:,1] & ~combinations[:,2]) | 
                   (~combinations[:,0] & ~combinations[:,1] & combinations[:,2])) * 100
    
        plt.bar(labels, contrib)
        plt.xlabel("Kombinationen X3 X4 X5 (>0 als 1, sonst 0)")
        plt.ylabel("Logischer Beitrag")
        plt.title("Logische Interaktion (XOR-ähnlich)")
        return plt.gca()

    _()
    return


@app.cell
def _(X, df, np, plt):
    def _():
        center = np.mean(X[:, 6:9], axis=0)
        distances = np.linalg.norm(X[:, 6:9] - center, axis=1)
        alpha = -np.log(1 / 99) / distances.max()
        spatial_contribution = 1 + 150 * np.exp(-alpha * distances)
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        sc = ax.scatter(
            df["feature_7"],
            df["feature_8"],
            df["feature_9"],
            c=spatial_contribution,
            cmap="coolwarm",
            alpha=0.7,
            vmin=1, vmax=50
        )
    
        ax.set_title("Räumliche Interaktion (skaliert 1–100) – Abstand zum Mittelwertszentrum")
        ax.set_xlabel("feature_7")
        ax.set_ylabel("feature_8")
        ax.set_zlabel("feature_9")
    
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
    
        fig.colorbar(sc, label="Beitrag (1–100)")
    
        ax.scatter(*center, color='black', s=80, label="Zentrum (Mittelwert)", marker='x')
        ax.legend()
    
        plt.tight_layout()
    
        return plt.gca()
    
    _()
    return


@app.cell
def _(Digraph, Image, io, np, plt):
    def _():
        dot = Digraph(comment='Conditional Contribution Logic')
        dot.attr(rankdir='TB', size='8,8', nodesep='1', ranksep='0.75')
    
        decision_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#dbeafe', 'fontsize': '14', 'fontname': 'Arial', 'color': '#2563eb'}
        leaf_styles = {
            'high':   {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#bbf7d0', 'fontsize': '14', 'fontname': 'Arial', 'color': '#16a34a'},
            'mid':    {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#fef9c3', 'fontsize': '14', 'fontname': 'Arial', 'color': '#ca8a04'},
            'neg':    {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#fecaca', 'fontsize': '14', 'fontname': 'Arial', 'color': '#dc2626'},
            'zero':   {'shape': 'box', 'style': 'filled',      'fillcolor': '#e5e7eb', 'fontsize': '14', 'fontname': 'Arial', 'color': '#6b7280'},
        }
    
        dot.node('A', 'X9 > 0.5?', **decision_style)
        dot.node('B', 'X10 < -0.2?', **decision_style)
        dot.node('C1', '|X11| < 0.3?', **decision_style)
        dot.node('C2', '|X11| < 0.3?', **decision_style)
        dot.node('D', 'X12 > X9?', **decision_style)
    
        dot.node('E', '40', **leaf_styles['high'])
        dot.node('F', '20', **leaf_styles['mid'])
        dot.node('G', '-25', **leaf_styles['neg'])
        dot.node('H', '1', **leaf_styles['mid'])
        dot.node('I', '0', **leaf_styles['zero'])
    
        dot.edge('A', 'B', 'Ja', color='#22c55e', fontcolor='#22c55e', penwidth='2', fontsize='13')
        dot.edge('B', 'E', 'Ja', color='#22c55e', fontcolor='#22c55e', penwidth='2', fontsize='13')
        dot.edge('B', 'C1', 'Nein', color='#64748b', fontcolor='#64748b', penwidth='2', fontsize='13')
        dot.edge('C1', 'F', 'Ja', color='#22c55e', fontcolor='#22c55e', penwidth='2', fontsize='13')
        dot.edge('A', 'D', 'Nein', color='#64748b', fontcolor='#64748b', penwidth='2', fontsize='13')
        dot.edge('D', 'G', 'Ja', color='#22c55e', fontcolor='#22c55e', penwidth='2', fontsize='13')
        dot.edge('D', 'C2', 'Nein', color='#64748b', fontcolor='#64748b', penwidth='2', fontsize='13')
        dot.edge('C2', 'H', 'Ja', color='#22c55e', fontcolor='#22c55e', penwidth='2', fontsize='13')
        dot.edge('C1', 'I', 'Nein', color='#64748b', fontcolor='#64748b', penwidth='2', fontsize='13')
        dot.edge('C2', 'I', 'Nein', color='#64748b', fontcolor='#64748b', penwidth='2', fontsize='13')
    
        graph_png = dot.pipe(format='png')
        graph_img = Image.open(io.BytesIO(graph_png))
    
        x9 = np.linspace(-2, 2, 200)
        x10 = np.linspace(-2, 2, 200)
        X9, X10 = np.meshgrid(x9, x10)
        X11 = 0
        X12 = 0
    
        cond_A = X9 > 0.5
        cond_B = X10 < -0.2
        cond_C = np.abs(X11) < 0.3
        cond_D = X12 > X9
    
        contribution = np.zeros_like(X9)
        contribution[cond_A & cond_B] = 40.0
        contribution[cond_A & ~cond_B & cond_C] = 20.0
        contribution[~cond_A & cond_D] = -25.0
        contribution[~cond_A & ~cond_D & cond_C] = 1.0
    
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
        axes[0].imshow(graph_img)
        axes[0].axis('off')
        axes[0].set_title('Entscheidungsbaum\n(konditionale Logik)')
    
        c = axes[1].contourf(X9, X10, contribution, levels=[-30, 0, 1, 20, 40, 45], cmap="viridis", alpha=0.85)
        fig.colorbar(c, ax=axes[1], label="Beitrag")
        axes[1].set_xlabel("X9")
        axes[1].set_ylabel("X10")
        axes[1].set_title("Conditional Contribution Heatmap\n(X11=0, X12=0)")
    
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(X, np, plt):
    def _():
        features = ["Feature 13", "Feature 14", "Feature 15"]
        coefs = [4.8, -2.2, 3.5]
        X_linear = X[:, 13:16]
        linear_contribution = 4.8 * X_linear[:,0] - 2.2 * X_linear[:,1] + 3.5 * X_linear[:,2]
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
        for i, ax in enumerate(axes):
            xi = X_linear[:, i]
            sc = ax.scatter(xi, linear_contribution, c=linear_contribution, cmap="coolwarm", alpha=0.7)
            coef = np.polyfit(xi, linear_contribution, 1)
            x_fit = np.linspace(xi.min(), xi.max(), 100)
            y_fit = np.polyval(coef, x_fit)
            ax.plot(x_fit, y_fit, color="black", linewidth=2, linestyle="--", label="Trendlinie")
            ax.set_xlabel(features[i], fontsize=13)
            if i == 0:
                ax.set_ylabel("linear_contribution", fontsize=13)
            ax.set_title(f"{features[i]} vs. linear_contribution")
            ax.legend()
    
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.03)
        cbar.set_label("linear_contribution (Farbcodierung)", fontsize=13)
        plt.suptitle("Lineare Beziehung: Feature vs. Beitrag mit Trendlinie und Farblegende", fontsize=16)
        return plt.gca()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Evaluation Results""")
    return


@app.cell
def _(evaluation_df):
    evaluation_df
    return


@app.cell
def _(evaluation_df, plt, sns):
    plt.figure(figsize=(8,5))
    sns.barplot(x='model', y='r2', data=evaluation_df)
    plt.title('R² Score je Modell')
    plt.ylabel('R² Score')
    plt.xlabel('Modell')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.gca()

    return


@app.cell
def _(df, evaluation_df, np, plt):
    def _():
        if 'target' in df.columns:
            y = df['target'].values
        else:
            y = np.zeros(len(evaluation_df))

        rmse = evaluation_df['rmse'].values
        norm_rmse = rmse / (y.max() - y.min())
        neg_norm_rmse = -norm_rmse
        r2 = evaluation_df['r2'].values
        model_names = evaluation_df['model'].tolist()
        x = np.arange(len(model_names))

        bar_width = 0.4
        fig, ax1 = plt.subplots(figsize=(10,6))

        b1 = ax1.bar(x - bar_width/2, r2, width=bar_width, color='tab:blue', label='R² Score')
        b2 = ax1.bar(x + bar_width/2, neg_norm_rmse, width=bar_width, color='tab:orange', label='- normalized RMSE')

        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.set_ylabel('Score')
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_title('R² Score und negativer normalisierter RMSE pro Modell')
        ax1.legend()
        plt.tight_layout()
        return plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
