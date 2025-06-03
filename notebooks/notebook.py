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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import font_manager as fm

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DM Sans'] + plt.rcParams['font.sans-serif']

    plt.rcParams.update({
        "figure.facecolor": "#F9FAFB",        # Haupt-Hintergrund (Slide)
        "axes.facecolor": "#FFFFFF",          # Plot-Hintergrund (Container)
        "axes.edgecolor": "#E5E7EB",          # Rahmen
        "axes.labelcolor": "#374151",         # Achsentitel (sekundärer Text)
        "xtick.color": "#6B7280",             # Tick-Labels (tertiärer Text)
        "ytick.color": "#6B7280",             # Tick-Labels (tertiärer Text)
        "grid.color": "#E5E7EB",              # Gitternetzlinien (dezent)
        "text.color": "#111827",              # Haupttext
        "axes.prop_cycle": plt.cycler(color=["#3B82F6", "#2563EB"]),  # Akzentfarben
        "axes.grid": True,                    # Grid aktivieren
        "grid.linestyle": "-",                # Linienstil für Grid
        "grid.linewidth": 1.0,                # Gitternetzlinien dünn
        "axes.spines.top": False,             # Obere Rahmenlinie aus
        "axes.spines.right": False,           # Rechte Rahmenlinie aus
    })
    return (
        Digraph,
        Image,
        LinearSegmentedColormap,
        alt,
        fm,
        io,
        itertools,
        mo,
        mpl,
        np,
        os,
        pd,
        plt,
        sns,
    )


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
def _(LinearSegmentedColormap, df, np, plt):
    def _():
        f1_vals = np.linspace(df["feature_1"].min(), df["feature_1"].max(), 100)
        f2_vals = np.linspace(df["feature_2"].min(), df["feature_2"].max(), 100)
        f1_grid, f2_grid = np.meshgrid(f1_vals, f2_vals)

        fixed_f3 = 5
        poly_contrib = 0.0003 * (f1_grid * f2_grid * fixed_f3) ** 2

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        blues = LinearSegmentedColormap.from_list("blues", [
            "#3B82F6",
            "#C4B5FD",
            "#DC2626",
        ])

        surface = ax.plot_surface(
            f1_grid, f2_grid, poly_contrib,
            cmap=blues,
            edgecolor='none',
            alpha=0.95
        )

        fig.patch.set_facecolor("#F9FAFB")
        ax.set_facecolor("#FFFFFF")

        ax.set_title(
            "Polynomiale Interaktion",
            color="#111827", fontsize=15, weight="bold", pad=16
        )
        ax.set_xlabel("feature_1", color="#374151", fontsize=12, labelpad=12)
        ax.set_ylabel("feature_2", color="#374151", fontsize=12, labelpad=12)
        ax.set_zlabel("Beitrag", color="#374151", fontsize=12, labelpad=12)

        ax.tick_params(axis='x', colors="#6B7280", labelsize=11)
        ax.tick_params(axis='y', colors="#6B7280", labelsize=11)
        ax.tick_params(axis='z', colors="#6B7280", labelsize=11)

        cb = fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.1)
        cb.set_label("Beitrag", color="#374151")
        cb.ax.yaxis.set_tick_params(color="#6B7280")
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="#6B7280")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        $$
        a = \text{feature}_1 > 0 \\
        b = \text{feature}_2 > 0 \\
        c = \text{feature}_3 > 0 \\
        \text{contrib} =
        \begin{cases}
        100, & \text{wenn}\ (a \land b \land \lnot c)\ \lor\ (\lnot a \land \lnot b \land c) \\
        0, & \text{sonst}
        \end{cases}
        $$
        """
    )
    return


@app.cell
def _(LinearSegmentedColormap, np, pd, plt):
    def _():
        np.random.seed(42)
        num_points = 10000
        X_features = np.random.uniform(-10, 10, size=(num_points, 3))
        X = np.random.rand(num_points, 9) 
        X[:, 6:9] = X_features

        df = pd.DataFrame({
            "feature_7": X_features[:, 0],
            "feature_8": X_features[:, 1],
            "feature_9": X_features[:, 2]
        })

        center = np.mean(X[:, 6:9], axis=0)
        distances = np.linalg.norm(X[:, 6:9] - center, axis=1)

        if distances.max() == 0:
            spatial_contribution = np.full(X.shape[0], 1 + 150.0)
        else:
            alpha_decay_param = -np.log(1 / 99) / distances.max()
            spatial_contribution = 1 + 150 * np.exp(-alpha_decay_param * distances)

        s_c_min = spatial_contribution.min()
        s_c_max = spatial_contribution.max()
        if s_c_max == s_c_min:
            alpha_values = np.ones_like(spatial_contribution) 
        else:
            alpha_values = 0.7 * (spatial_contribution - s_c_min) / (s_c_max - s_c_min) + 0.3

        my_cmap = LinearSegmentedColormap.from_list(
            "lavendel_blau_weiss", [
                "#FFFFFF",
                "#E3F2FD",
                "#90CAF9",
                "#B39DDB",
                "#DC2626"
            ]
        )

        fig = plt.figure(figsize=(10, 8)) 
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            df["feature_7"],
            df["feature_8"],
            df["feature_9"],
            c=spatial_contribution,
            cmap=my_cmap,
            alpha=alpha_values,
            vmin=1, vmax=50,
            s=10
        )

        # CI/CD Farben für Hintergrund und Achsen
        fig.patch.set_facecolor("#F9FAFB")
        ax.set_facecolor("#FFFFFF")

        ax.set_title(
            "Räumliche Interaktion",
            color="#111827", fontsize=16, weight="bold", pad=18
        )
        ax.set_xlabel("feature_7", color="#374151", fontsize=12, labelpad=10)
        ax.set_ylabel("feature_8", color="#374151", fontsize=12, labelpad=10)
        ax.set_zlabel("feature_9", color="#374151", fontsize=12, labelpad=10)
        ax.tick_params(axis='x', colors="#6B7280", labelsize=11)
        ax.tick_params(axis='y', colors="#6B7280", labelsize=11)
        ax.tick_params(axis='z', colors="#6B7280", labelsize=11)

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])

        cbar = fig.colorbar(sc, label="Beitrag (Farbskala 1-50)", shrink=0.7, aspect=20, pad=0.02)
        cbar.set_label("Beitrag (Farbskala 1-50)", color="#374151")
        cbar.ax.yaxis.set_tick_params(color="#6B7280")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#6B7280")

        # Zentrum hervorheben
        ax.scatter(*center, color="#00F700", s=100, label="Zentrum (Mittelwert)", marker='x', depthshade=False)
        ax.legend(facecolor="#FFFFFF", edgecolor="#E5E7EB", labelcolor="#2563EB")

        plt.tight_layout()
        print("Plot saved as spatial_interaction_plot_lavendel_blau_weiss.png")

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
            'high':   {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#40A99B', 'fontsize': '14', 'fontname': 'Arial', 'color': '#40A99B', 'fontcolor': 'white'},
            'mid':    {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#8FD76B', 'fontsize': '14', 'fontname': 'Arial', 'color': '#8FD76B'},
            'one':    {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#498E9F', 'fontsize': '14', 'fontname': 'Arial', 'color': '#498E9F', 'fontcolor': 'white'},
            'neg':    {'shape': 'box', 'style': 'filled,bold', 'fillcolor': '#5D6099', 'fontsize': '14', 'fontname': 'Arial', 'color': '#FFFFFF', 'fontcolor': 'white'},
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
        dot.node('H', '1', **leaf_styles['one'])
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
        plt.suptitle("Lineare Beziehung", fontsize=16)
        return plt.gca()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Evaluation Results""")
    return


@app.cell
def _():
    def _():
        import glob
        import re
        import pandas as pd

        files = glob.glob("./data/results/feature_interactions/cv_results_complex_dataset_*.csv")

        dfs = []
        for file in files:
            match = re.search(r'n(\d+)_noise(\d+)_irrelevant(\d+)', file)
            if match:
                n_samples = int(match.group(1))
                noise = int(match.group(2))
                irr = int(match.group(3))
            else:
                n_samples, noise, irr = 0, 0, 0

            df = pd.read_csv(file)
            df['n_samples'] = n_samples
            df['noise'] = noise
            df['irrelevant'] = irr
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    all_evaluations_df = _()
    return (all_evaluations_df,)


@app.cell
def _(evaluation_df, np, plt):

    def _():
        rmse = evaluation_df['rmse'].values
        r2 = evaluation_df['r2'].values
        model_names = evaluation_df['model'].tolist()
        x = np.arange(len(model_names))
        print(rmse, r2, model_names)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: R² Score
        axes[0].bar(x, r2, width=0.6, color='tab:blue')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].set_title('R² Score pro Modell')

        # Plot 2: 1 - normalized RMSE
        axes[1].bar(x, rmse, width=0.6, color='tab:orange')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names)
        axes[1].set_ylabel('1 - normalisierter RMSE')
        axes[1].set_title('1 - normalisierter RMSE pro Modell')

        plt.tight_layout()
        return plt.gca()

    _()

    return


@app.cell
def _(all_evaluations_df, plt, sns):
    df_noise = all_evaluations_df[
        (all_evaluations_df['n_samples'] == 1000) &
        (all_evaluations_df['irrelevant'] == 0) &
        (all_evaluations_df['model'].str.lower().isin(['tabpfn', 'catboostregressor']))
    ]

    plt.figure(figsize=(8,6))
    sns.lineplot(data=df_noise, x="noise", y="r2", hue="model", marker="o")
    plt.title("Modellrobustheit gegenüber Rauschen (R²-Score, n=1000, irrelevant=0)")
    plt.ylabel("R² Score")
    plt.xlabel("Rauschlevel [%]")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return (df_noise,)


@app.cell
def _(all_evaluations_df, plt, sns):
    df_irr = all_evaluations_df[
        (all_evaluations_df['n_samples'] == 1000) &
        (all_evaluations_df['noise'] == 0) &
        (all_evaluations_df['model'].str.lower().isin(['tabpfn', 'catboostregressor']))
    ]

    plt.figure(figsize=(8,6))
    sns.lineplot(data=df_irr, x="irrelevant", y="r2", hue="model", marker="o")
    plt.title("Robustheit gegenüber irrelevanten Features (R²-Score, n=1000, noise=0)")
    plt.ylabel("R² Score")
    plt.xlabel("Anzahl irrelevanter Features")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.gca()

    return (df_irr,)


@app.cell
def _(mo):
    mo.md(r"""# T2 - Fine Tuning""")
    return


@app.cell
def _(np, plt):
    def _():
        models = ["TabPFN", "TabPFN + Adapter"]
        test_accuracy = [0.769, 0.769]

        epochs = np.arange(1, 6)
        loss = [0.3147, 0.3002, 0.2718, 0.2190, 0.1827]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(models, test_accuracy, color=["#8C52FF", "#FF7043"], width=0.6)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Test Accuracy")
        axes[0].set_title("Test Accuracy Vergleich")

        for i, v in enumerate(test_accuracy):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

        axes[1].plot(epochs, loss, marker='o', color="#8C52FF")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Trainings-Loss pro Epoche")
        axes[1].grid(True)

        plt.tight_layout()
        return plt.gca()

    _()
    return


if __name__ == "__main__":
    app.run()
