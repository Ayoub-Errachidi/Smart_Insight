import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse

from .views_importation import SESSION_KEYS, load_dataset_from_session

# -------------------- Helpers --------------------

def encode_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_histograms(numeric_data):
    plots = []
    for col in numeric_data.columns:
        data = numeric_data[col].dropna()
        fig, ax = plt.subplots(figsize=(12, 5))

        mean, median = data.mean(), data.median()
        std = data.std()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        mode_val = data.mode().iloc[0] if not data.mode().empty else None

        sns.histplot(data, bins=30, kde=True, color='cornflowerblue',
                     edgecolor='black', alpha=0.8, stat='density', ax=ax)

        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f"Moyenne: {mean:.2f}")
        ax.axvline(median, color='green', linestyle='-', linewidth=2, label=f"Médiane: {median:.2f}")
        ax.axvline(q1, color='orange', linestyle=':', linewidth=1.5, label=f"Q1: {q1:.2f}")
        ax.axvline(q3, color='purple', linestyle=':', linewidth=1.5, label=f"Q3: {q3:.2f}")
        if mode_val is not None:
            ax.axvline(mode_val, color='blue', linestyle='-.', linewidth=1.5, label=f"Mode: {mode_val:.2f}")

        ax.annotate(f"Écart-type: {std:.2f}", xy=(0.98, 0.90), xycoords='axes fraction',
                    ha='right', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

        ax.set_title(f"Histogramme : {col}", fontsize=15)
        ax.set_xlabel(col)
        ax.set_ylabel("Densité")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plots.append({
            'col_name': col,
            'image': encode_plot_to_base64(fig),
            'mean': round(mean, 2),
            'median': round(median, 2),
            'std': round(std, 2),
            'q1': round(q1, 2),
            'q3': round(q3, 2),
            'mode': round(mode_val, 2) if mode_val is not None else None
        })

    return plots


def generate_single_plot(numeric_data, plot_type):
    fig, ax = plt.subplots(figsize=(12, 8))

    if plot_type == "box":
        sns.boxplot(data=numeric_data, ax=ax)
        title = "Box Plot"

    elif plot_type == "scatter" and numeric_data.shape[1] >= 2:
        sns.scatterplot(x=numeric_data.columns[0], y=numeric_data.columns[1],
                        data=numeric_data, ax=ax)
        title = f"Scatter Plot : {numeric_data.columns[0]} vs {numeric_data.columns[1]}"

    elif plot_type == "heatmap":
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        title = "Heatmap"

    elif plot_type == "bar":
        means = numeric_data.mean()
        colors = plt.cm.viridis(np.linspace(0, 1, len(means)))
        bars = ax.bar(means.index, means.values, color=colors)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_ylabel("Valeur moyenne")
        ax.set_xlabel("Variables")
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        title = "Bar Plot"

    elif plot_type == "line":
        numeric_data.plot(ax=ax)
        title = "Line Plot"

    elif plot_type == "violin":
        sns.violinplot(data=numeric_data, ax=ax)
        title = "Violin Plot"

    elif plot_type == "pair":
        fig = sns.pairplot(numeric_data).fig
        title = "Pair Plot"

    else:
        raise ValueError("Type de graphe non reconnu")

    encoded_image = encode_plot_to_base64(fig)
    return {'image': encoded_image, 'title': title}


def generate_pairplot(numeric_data, col1, col2):
    clean_data = numeric_data[[col1, col2]].dropna()

    if clean_data.empty:
        raise ValueError("Pas assez de données valides pour générer un pair plot avec les colonnes sélectionnées.")

    sns.set(style="whitegrid")
    pairplot = sns.pairplot(
        clean_data,
        diag_kind='kde',
        corner=False,
        plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'},
        diag_kws={'shade': True, 'color': 'blue'}
    )
    pairplot.fig.suptitle(f"{col1} vs {col2}", fontsize=16, y=1.02)

    encoded_image = encode_plot_to_base64(pairplot.fig)
    return {'image': encoded_image, 'title': "Pair Plot"}

# -------------------- Vues --------------------

def visualisations_view(request: HttpRequest):
    json_data = request.session.get(SESSION_KEYS["data"])
    if not json_data:
        return redirect("dashboard_view")

    df = pd.read_json(io.StringIO(json_data))
    preview = df.head().to_html(classes="table table-bordered table-striped", index=False)
    return render(request, "pages/visualisations.html", {"preview": preview})


def select_pairplot_columns_view(request: HttpRequest):
    json_data = request.session.get(SESSION_KEYS["data"])
    if not json_data:
        return redirect("dashboard_view")

    df = pd.read_json(io.StringIO(json_data))
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    return render(request, "pages/pair.html", {
        "columns": numeric_columns,
        "plot_image": None,
        "col1": None,
        "col2": None,
        "title": "Sélection des colonnes pour Pair Plot"
    })


def plot_result_view(request: HttpRequest, plot_type: str):
    json_data = request.session.get(SESSION_KEYS["data"])
    if not json_data:
        return HttpResponse("Aucun dataset disponible en session.", status=400)

    try:
        df = pd.read_json(io.StringIO(json_data))
        numeric_data = df.select_dtypes(include='number')

        if numeric_data.empty:
            return HttpResponse("Aucune colonne numérique détectée.", status=400)

        if plot_type == "histogram":
            plots = generate_histograms(numeric_data)
            return render(request, "pages/histogram.html", {"plot_images": plots, "title": "Histogrammes"})

        if plot_type == "pair1":
            col1 = request.GET.get('col1')
            col2 = request.GET.get('col2')
            if col1 not in numeric_data.columns or col2 not in numeric_data.columns:
                return HttpResponse("Les colonnes sélectionnées n'existent pas.", status=400)

            result = generate_pairplot(numeric_data, col1, col2)
            return render(request, "pages/pair.html", {
                "plot_image": result['image'],
                "columns": numeric_data.columns,
                "col1": col1,
                "col2": col2,
                "title": result['title']
            })

        else:
            result = generate_single_plot(numeric_data, plot_type)
            return render(request, "pages/histogram.html", {
                "plot_image": result['image'],
                "title": result['title']
            })

    except Exception as e:
        return HttpResponse(f"Erreur : {str(e)}", status=500)