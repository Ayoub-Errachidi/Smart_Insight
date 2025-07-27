import traceback
import pandas as pd
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

from .views_importation import load_dataset_from_session, logger


# -------------------- Statistiques du Dataset --------------------
def statistique_view(request: HttpRequest) -> HttpResponse:
    try:
        data = load_dataset_from_session(request)

        if data.empty:
            return render(request, "pages/statistiques.html", {"error": "Le dataset importé est vide."})

        # Initialisation
        stats_base, stats_adv, stats_text = None, None, None
        correlation_matrix_html, multicollinear_pairs = None, []
        missing_values, outliers, pca_html = None, None, None
        stats_base_dict, stats_adv_dict, stats_text_dict, top_values_dict = {}, {}, {}, {}

        # ------ Données numériques ------
        numeric_data = data.select_dtypes(include=['number'])
        if not numeric_data.empty:
            # Statistiques descriptives de base
            stats_base_df = numeric_data.describe()
            stats_base = stats_base_df.to_html(classes="table table-striped")

            stats_base_dict = stats_base_df.T.to_dict(orient='index')

            # Statistiques avancées
            stats_adv_df = pd.DataFrame({
                "Médiane": numeric_data.median(),
                "Variance": numeric_data.var(),
                "Écart-type": numeric_data.std(),
                "Coefficient de variation": numeric_data.std() / numeric_data.mean(),
                "Asymétrie (skewness)": numeric_data.skew(),
                "Aplatissement (kurtosis)": numeric_data.kurt()
            })
            stats_adv = stats_adv_df.to_html(classes="table table-striped")
            stats_adv_dict = stats_adv_df.to_dict(orient="index")

            # Valeurs aberrantes (outliers)
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers_detected = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
            outliers = outliers_detected.to_frame(name="Nombre de valeurs aberrantes").to_html(classes="table table-striped")

            # Heatmap de corrélation
            if numeric_data.shape[1] > 1:
                try:
                    correlation = numeric_data.corr()
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                    plt.title("Matrice de corrélation")
                    plt.tight_layout()

                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    heatmap_base64 = base64.b64encode(image_png).decode('utf-8')
                    correlation_matrix_html = f'<img src="data:image/png;base64,{heatmap_base64}" class="img-fluid"/>'
                    plt.close()
                except Exception as e:
                    logger.warning(f"Erreur heatmap corrélation : {e}")
                    correlation_matrix_html = "<p class='text-warning'>Erreur lors de la heatmap de corrélation.</p>"

                # Multicolinéarité (corrélation > 0.9)
                corr_matrix = correlation.abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                for col in upper_triangle.columns:
                    for idx in upper_triangle.index:
                        val = upper_triangle.loc[idx, col]
                        if pd.notna(val) and val >= 0.9:
                            multicollinear_pairs.append((idx, col, round(val, 3)))

        # ------ Données textuelles ------
        text_data = data.select_dtypes(include=['object'])
        if not text_data.empty:
            unique_values_dict = {}
            for col in text_data.columns:
                uniques = sorted(set(str(val) for val in text_data[col].dropna().unique()))
                unique_values_dict[col] = {"count": len(uniques), "values": uniques}

            unique_values_df = pd.DataFrame({
                col: {
                    "Nb valeurs uniques": unique_values_dict[col]["count"],
                    "Valeurs uniques": ", ".join(unique_values_dict[col]["values"][:10])
                } for col in unique_values_dict
            }).T

            stats_text = {
                "unique_values": unique_values_df.to_html(classes="table table-striped", escape=False),
                "top_values": text_data.describe().T.to_html(classes="table table-striped"),
                "avg_word_count": text_data.apply(lambda x: x.dropna().astype(str).apply(lambda y: len(y.split())).mean()).to_frame(name="Nombre moyen de mots").to_html(classes="table table-striped")
            }

            stats_text_dict = {
                col: {
                    "nb_uniques": unique_values_dict[col]["count"],
                    "values": unique_values_dict[col]["values"][:5]
                } for col in unique_values_dict
            }

            top_values_dict = text_data.describe().T.to_dict(orient='index')

        # ------ Valeurs manquantes ------
        null_counts = data.isnull().sum()
        missing_values = null_counts.to_frame(name="Valeurs manquantes").to_html(classes="table table-striped")

        # ------ PCA ------
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            try:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(numeric_data)

                pca = PCA()
                pca_result = pca.fit(data_scaled)

                pca_df = pd.DataFrame({
                    "Composante": [f"PC{i+1}" for i in range(len(pca.explained_variance_))],
                    "Valeur propre": pca.explained_variance_,
                    "Variance expliquée (%)": pca.explained_variance_ratio_ * 100,
                    "Variance cumulée (%)": pca.explained_variance_ratio_.cumsum() * 100
                })

                pca_html = pca_df.to_html(classes="table table-striped", index=False)

            except Exception as e:
                logger.warning(f"Erreur PCA : {e}")
                pca_html = "<p class='text-warning'>Erreur lors du calcul de la PCA.</p>"

        return render(request, "pages/statistiques.html", {
            "stats_base": stats_base,
            "stats_base_dict": stats_base_dict,
            "stats_adv": stats_adv,
            "stats_adv_dict": stats_adv_dict,
            "stats_text": stats_text,
            "stats_text_dict": stats_text_dict,
            "top_values_dict": top_values_dict,
            "correlation_matrix": correlation_matrix_html,
            "multicollinear_pairs": multicollinear_pairs,
            "missing_values": missing_values,
            "outliers": outliers,
            "pca_html": pca_html
        })
    
    except Exception as e:
        logger.error("Erreur lors du calcul des statistiques : %s", traceback.format_exc())
        return render(request, "pages/statistiques.html", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
