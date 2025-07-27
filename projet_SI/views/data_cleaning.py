from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
import logging
import traceback
import pandas as pd
from django.core.exceptions import ValidationError
from typing import Tuple
import io

# --- valeurs aberrantes ---
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore, skew
import numpy as np



from projet_SI.views.views_importation import (
    load_dataset_from_session,
    save_dataset_to_session,
    build_dataset_context,
    SESSION_KEYS,
)

logger = logging.getLogger(__name__)

# ----- Gestion des données manquantes -----
def clean_missing_values(data: pd.DataFrame, threshold_drop: float = 0.6) -> pd.DataFrame:
    cleaned_data = data.copy()
    n_rows = len(cleaned_data)

    # 1. Drop columns with too many missing values
    col_thresh = n_rows * threshold_drop
    cleaned_data = cleaned_data.dropna(axis=1, thresh=col_thresh)

    # 2. Clean column by column
    for col in cleaned_data.columns:
        missing_ratio = cleaned_data[col].isnull().sum() / n_rows

        if missing_ratio == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned_data[col]):
            if missing_ratio < 0.1:
                # Interpolate if very few values are missing
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear', limit_direction='forward', axis=0)
            elif missing_ratio < 0.4:
                # Fill with mean if moderate missing values
                cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
            else:
                # Drop rows with missing values in this column if too many
                cleaned_data = cleaned_data.dropna(subset=[col])
        
        elif pd.api.types.is_datetime64_any_dtype(cleaned_data[col]):
            try:
                # Fill with mode (most frequent date)
                cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0], inplace=True)
            except IndexError:
                cleaned_data[col].fillna(method='ffill', inplace=True)  # fallback

        else:  # Categorical or text columns
            if missing_ratio < 0.5:
                # Fill with mode or 'Inconnu'
                mode_val = cleaned_data[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Inconnu"
                cleaned_data[col].fillna(fill_val, inplace=True)
            else:
                # Drop rows with missing value in this categorical column
                cleaned_data = cleaned_data.dropna(subset=[col])

    return cleaned_data


def nettoyer_dataset_view(request: HttpRequest) -> HttpResponse:
    try:
        # Charger le dataset brut depuis la session
        data = load_dataset_from_session(request)
        rows_before = data.shape[0]
        cols_before = data.shape[1]
        columns_before = list(data.columns)

        # Nettoyage
        cleaned_data = clean_missing_values(data)
        rows_after = cleaned_data.shape[0]
        columns_after = list(cleaned_data.columns)
        cols_after = len(columns_after)

        # Colonnes supprimées
        removed_columns = list(set(columns_before) - set(columns_after))
        cols_removed = len(removed_columns)

        # Colonnes traitées (valeurs nulles remplies)
        cols_touched = []
        for col in columns_after:
            if col in columns_before:
                if data[col].isnull().sum() > 0 and cleaned_data[col].isnull().sum() == 0:
                    cols_touched.append(col)
        cols_filled = len(cols_touched)

        # Calcul des valeurs nulles après nettoyage
        null_counts = cleaned_data.isnull().sum()
        null_percentages = (null_counts / len(cleaned_data)) * 100

        # Convertir le DataFrame nettoyé en HTML pour affichage
        cleaned_data_html = cleaned_data.to_html(classes="table table-striped table-hover")

        # Sauvegarder le dataset nettoyé dans la session
        filename_with_ext = request.session.get(SESSION_KEYS["filename"], "dataset_nettoye") + "." + request.session.get(SESSION_KEYS["file_type"])

        save_dataset_to_session(request, cleaned_data, filename_with_ext)

        return render(request, "pages/resultat_nettoyage.html", {
            "table_nettoyer": cleaned_data_html,
            "colonnes_avec_valeurs_nulles": (null_counts > 0).sum(),
            "colonnes_sans_valeurs_nulles": (null_counts == 0).sum(),
            "null_values_list": [
                (col, int(null_counts[col]), round(null_percentages[col], 2))
                for col in cleaned_data.columns
            ],
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "cols_removed": cols_removed,
            "columns_after": columns_after,
            "removed_columns": removed_columns,
            "cols_filled": cols_filled,
            "cols_touched": cols_touched
        })

    except Exception as e:
        logger.error("Erreur nettoyage : %s\n%s", e, traceback.format_exc())
        return render(request, "pages/resultat_nettoyage.html", {
            "error": "Erreur lors du nettoyage des données."
        })

# -------------------- Supprimer les doublons --------------------
def remove_duplicates(data: pd.DataFrame) -> Tuple[
    pd.DataFrame, int, pd.DataFrame, list, list, list[dict]
]:
    """
    Retourne :
    - DataFrame nettoyé
    - Nombre de doublons supprimés
    - DataFrame des doublons supprimés
    - Liste des index supprimés
    - Liste des index des doublons uniques supprimés
    - Liste des groupes avec : ligne gardée + lignes supprimées (détails + index)
    """
    duplicated_mask = data.duplicated(keep='first')
    deleted_rows = data[duplicated_mask].copy()
    deleted_indices = deleted_rows.index.tolist()

    # Groupes de doublons (clé = tuple de ligne)
    grouped = data[data.duplicated(keep=False)].copy()
    grouped['__row_key__'] = grouped.apply(lambda row: tuple(row), axis=1)

    groups = []
    for key, group_df in grouped.groupby('__row_key__'):
        indices = group_df.index.tolist()
        if len(indices) > 1:
            kept_idx = indices[0]
            removed_idxs = indices[1:]

            kept_row = data.loc[kept_idx].to_dict()
            removed_rows = [data.loc[idx].to_dict() for idx in removed_idxs]

            groups.append({
                "ligne_conservee_index": kept_idx,
                "ligne_conservee_data": kept_row,
                "lignes_supprimees_index": removed_idxs,
                "lignes_supprimees_data": removed_rows
            })

    unique_deleted = deleted_rows.drop_duplicates()
    unique_deleted_indices = unique_deleted.index.tolist()

    cleaned_data = data.drop_duplicates(keep='first')
    duplicates_removed = len(deleted_rows)

    return cleaned_data, duplicates_removed, deleted_rows, deleted_indices, unique_deleted_indices, groups


def supprimer_doublons_view(request: HttpRequest) -> HttpResponse:
    try:
        data = load_dataset_from_session(request)
        rows_before = data.shape[0]

        # Suppression des doublons
        cleaned_data, duplicates_removed, deleted_rows, deleted_indices, unique_deleted_indices, groups = remove_duplicates(data)
        rows_after = cleaned_data.shape[0]

        table_cleaned_html = cleaned_data.to_html(classes="table table-striped table-hover")
        table_deleted_html = deleted_rows.to_html(classes="table table-bordered table-danger", index=True)

        # Sauvegarder le dataset nettoyé dans la session
        filename_with_ext = request.session.get(SESSION_KEYS["filename"], "dataset_sans_doublons") + "." + request.session.get(SESSION_KEYS["file_type"])
        save_dataset_to_session(request, cleaned_data, filename_with_ext)

        return render(request, "pages/resultat_doublons.html", {
            "table_sans_doublons": table_cleaned_html,
            "table_doublons_supprimes": table_deleted_html,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": duplicates_removed,
            "deleted_indices": deleted_indices,
            "unique_deleted_indices": unique_deleted_indices,
            "nombre_doublons_uniques": len(unique_deleted_indices),
            "groupes_doublons": groups
        })

    except Exception as e:
        logger.error("Erreur suppression doublons : %s\n%s", e, traceback.format_exc())
        return render(request, "pages/resultat_doublons.html", {
            "error": "Erreur lors de la suppression des doublons."
        })
    
# ----------- Nettoyage des valeurs aberrantes -----------
def traiter_outliers_auto(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, set, pd.DataFrame, str]:
    df_out = df.copy()
    traitement_log = {}
    deleted_indices = set()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        series = df_out[col]

        if series.isnull().all() or series.nunique() <= 1 or series.count() < 10:
            traitement_log[col] = "Colonne ignorée (vide, constante ou peu de données)"
            continue

        # Gérer les NaN
        if series.isna().sum() > 0:
            if series.isna().sum() / len(series) < 0.4:
                series = series.interpolate(method='linear', limit_direction='both')
                if series.isna().sum() > 0:
                    series.fillna(series.median(), inplace=True)
                log_na = "NaN interpolés / imputés"
            else:
                traitement_log[col] = "Trop de NaN – ignorée"
                continue
        else:
            log_na = "Aucun NaN"

        # Statistiques IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_iqr_mask = (series < lower) | (series > upper)
        n_outliers = outliers_iqr_mask.sum()
        outlier_ratio = n_outliers / len(series)

        try:
            col_skew = skew(series.dropna(), bias=False)
        except Exception:
            col_skew = series.skew()

        if n_outliers == 0:
            traitement_log[col] = f"{log_na} – Aucun outlier détecté"
            continue

        z_threshold = 3.0
        if len(series) < 100:
            z_threshold = 2.5
        elif abs(col_skew) > 1.5:
            z_threshold = 2.2

        # Méthode automatique
        if outlier_ratio < 0.01:
            lower_p = 0.01 + (0.05 * (IQR / (Q3 + 1e-5)))
            upper_p = 1 - lower_p
            lower_val, upper_val = series.quantile([lower_p, upper_p])
            df_out[col] = np.clip(series, lower_val, upper_val)
            traitement_log[col] = f"{log_na} – Winsorisation adaptative ({int(lower_p*100)}e–{int(upper_p*100)}e)"

        elif outlier_ratio < 0.08 and abs(col_skew) < 1.0:
            try:
                z_scores = zscore(series)
                mask_z = np.abs(z_scores) > z_threshold
                indices = series[mask_z].index
                deleted_indices.update(indices)
                traitement_log[col] = f"{log_na} – Z-score (>|z|>{z_threshold:.1f}) – {len(indices)} lignes supprimées"
            except Exception as e:
                traitement_log[col] = f"{log_na} – Erreur Z-score : {e}"

        elif outlier_ratio < 0.15:
            df_out.loc[outliers_iqr_mask, col] = series.median()
            traitement_log[col] = f"{log_na} – Remplacés par médiane – {n_outliers} valeurs"

        elif outlier_ratio > 0.3 or abs(col_skew) > 2.0:
            try:
                model = IsolationForest(contamination='auto', random_state=42)
                preds = model.fit_predict(series.values.reshape(-1, 1))
                anomalies_idx = series.index[preds == -1]
                deleted_indices.update(anomalies_idx)
                traitement_log[col] = f"{log_na} – Isolation Forest – {len(anomalies_idx)} lignes supprimées"
            except Exception as e:
                traitement_log[col] = f"{log_na} – Erreur Isolation Forest : {e}"

        else:
            indices = series[outliers_iqr_mask].index
            deleted_indices.update(indices)
            traitement_log[col] = f"{log_na} – Suppression IQR – {len(indices)} lignes supprimées"

    # Appliquer suppression et collecter les lignes supprimées
    indices_supprimes_total = list(deleted_indices)
    df_outliers = df.loc[indices_supprimes_total]
    df_out = df.drop(index=indices_supprimes_total)

    message_global = ""
    if len(indices_supprimes_total) / len(df) > 0.2:
        message_global = "⚠️ Plus de 20% des lignes ont été supprimées à cause des outliers."

    return df_out, traitement_log, set(indices_supprimes_total), df_outliers, message_global


def traiter_outliers_view(request: HttpRequest) -> HttpResponse:
    try:
        data = load_dataset_from_session(request)
        rows_before = data.shape[0]
        cols_before = data.shape[1]

        cleaned_df, traitement_log, indices_supprimes, df_outliers, message_global = traiter_outliers_auto(data)

        rows_after = cleaned_df.shape[0]
        cols_after = cleaned_df.shape[1]
        rows_removed = rows_before - rows_after

        filename_with_ext = request.session.get("filename", "dataset_nettoye") + "." + request.session.get("file_type", "csv")
        save_dataset_to_session(request, cleaned_df, filename_with_ext)

        return render(request, "pages/nettoyer_outliers.html", {
            "table_outliers": cleaned_df.to_html(classes="table table-bordered table-hover"),
            "table_outliers_removed": df_outliers.to_html(classes="table table-bordered table-sm", index=True) if not df_outliers.empty else None,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_removed,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "traitement_log": traitement_log.items(),
            "indices_supprimes": sorted(indices_supprimes),
            "message_global": message_global
        })

    except Exception as e:
        logger.error("Erreur traitement outliers : %s", traceback.format_exc())
        return render(request, "pages/nettoyer_outliers.html", {
            "error": f"Erreur lors du traitement : {str(e)}"
        })


# -----  Supprimer des colonnes -----
from django.shortcuts import redirect

def supprimer_colonnes_page(request: HttpRequest) -> HttpResponse:
    """
    Affiche la page avec la liste des colonnes à supprimer.
    """
    try:
        df = load_dataset_from_session(request)
        colonnes_list = df.columns.tolist()  # conversion en liste
        return render(request, "pages/supprimer_colonnes.html", {
            "colonnes": colonnes_list
        })
    
    except Exception as e:
        logger.error("Erreur chargement colonnes : %s", traceback.format_exc())
        return render(request, "pages/supprimer_colonnes.html", {
            "error": f"Impossible de charger les colonnes : {e}"
        })


def supprimer_colonnes_action(request: HttpRequest) -> HttpResponse:
    """
    Supprime les colonnes sélectionnées et renvoie vers Dataset.html.
    """
    if request.method == "POST":
        try:
            df = load_dataset_from_session(request)
            colonnes_a_supprimer = request.POST.getlist("colonnes")
            if colonnes_a_supprimer:
                df = df.drop(columns=[c for c in colonnes_a_supprimer if c in df.columns], errors="ignore")

                filename_with_ext = request.session.get(
                    SESSION_KEYS["filename"], "dataset_modifie"
                ) + "." + request.session.get(SESSION_KEYS["file_type"], "csv")

                save_dataset_to_session(request, df, filename_with_ext)

            return render(request, "pages/Dataset.html", build_dataset_context(df, request))
        except Exception as e:
            logger.error("Erreur suppression colonnes : %s", e)
            return render(request, "pages/supprimer_colonnes.html", {"error": "Erreur lors de la suppression des colonnes."})
    else:
        return redirect("supprimer_colonnes_page")
    
    
# ----- Téléchargement du dataset nettoyé -----
def download_dataset_view(request: HttpRequest) -> HttpResponse:
    try:
        data = load_dataset_from_session(request)
        filename = request.session.get(SESSION_KEYS["filename"], "dataset") + ".csv"
        
        buffer = io.StringIO()
        data.to_csv(buffer, index=False, encoding='utf-8-sig')  # UTF-8 avec BOM
        buffer.seek(0)

        response = HttpResponse(buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except Exception as e:
        logger.error(f"Téléchargement échoué : {e}")
        return HttpResponse("Erreur lors du téléchargement du dataset.", status=500)
