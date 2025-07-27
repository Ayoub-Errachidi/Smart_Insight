"""

import traceback
import io
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.core.exceptions import ValidationError
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from .views_importation import (
    load_dataset_from_session,
    load_transformed_dataset,
    save_dataset_to_session,
    SESSION_KEYS
)

# ====================================================
# Détection améliorée du type d'analyse ML
# ====================================================
def detect_ml_type(df: pd.DataFrame, target_column: str | None) -> str:
    
    if not target_column:
        return "clustering"

    y = df[target_column]
    n_samples = len(y)
    n_unique = y.nunique()

    # Classification si object ou category
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"

    # Classification si binaire (0/1 ou True/False)
    if n_unique <= 2:
        return "classification"

    # Si peu de classes par rapport au nombre d'échantillons
    if np.issubdtype(y.dtype, np.number):
        unique_ratio = n_unique / n_samples
        if unique_ratio < 0.05:  # moins de 5% de valeurs uniques
            return "classification"
        else:
            return "regression"

    # Par défaut
    return "classification"


# ====================================================
# 1. CHOIX DE LA TARGET
# ====================================================
def choix_cible_view(request: HttpRequest) -> HttpResponse:
    context = {}
    try:
        df = load_dataset_from_session(request)
        colonnes = df.columns.tolist()
        context["colonnes"] = colonnes

        previous_target = request.session.get("target_column")
        context["previous_target"] = previous_target

        if request.method == "POST":
            selected_target = request.POST.get("selected_target", "").strip()

            if not selected_target or selected_target == "aucune":
                request.session["target_column"] = None
                request.session["target_data"] = None

                ml_type = detect_ml_type(df, None)
                request.session["ml_type"] = ml_type

                context.update({
                    "selected_target": None,
                    "features_preview": colonnes,
                    "target_preview": [],
                    "ml_type": ml_type,
                    "success": "Aucune colonne cible sélectionnée. Mode clustering activé."
                })
            else:
                if selected_target not in colonnes:
                    raise ValidationError(f"La colonne '{selected_target}' n'existe pas dans le dataset.")

                y = df[selected_target]
                request.session["target_column"] = selected_target
                request.session["target_data"] = y.to_json()

                ml_type = detect_ml_type(df, selected_target)
                request.session["ml_type"] = ml_type

                context.update({
                    "selected_target": selected_target,
                    "features_preview": [col for col in colonnes if col != selected_target],
                    "target_preview": [selected_target],
                    "ml_type": ml_type,
                    "success": f"Colonne cible '{selected_target}' sélectionnée. Mode {ml_type} détecté."
                })

        return render(request, "pages/choix_cible.html", context)

    except ValidationError as ve:
        context["error"] = str(ve)
    except Exception as e:
        context["error"] = f"Erreur interne : {e}"
        context["traceback"] = traceback.format_exc()

    return render(request, "pages/choix_cible.html", context)

# ====================================================
# 2. FONCTION DE MATRICE DE CORRÉLATION
# ====================================================
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    if df.empty:
        return pd.DataFrame()
    return df.corr()

def generate_correlation_heatmap(corr_matrix: pd.DataFrame) -> str:
    
    if corr_matrix.empty:
        return ""

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Matrice de Corrélation", fontsize=14)
    plt.tight_layout()

    # Sauvegarder dans un buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Encoder en base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"



# ====================================================
# 3. FEATURE ENGINEERING
# ====================================================
def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_fe = df.copy()
    new_features = []

    numeric_cols = df_fe.select_dtypes(include=[np.number]).columns
    candidate_cols = df_fe.columns.difference(numeric_cols)

    date_cols = []
    for col in candidate_cols:
        try:
            convertible = pd.to_datetime(df_fe[col], errors='coerce')
            if convertible.notna().mean() >= 0.8:
                df_fe[col] = convertible
                date_cols.append(col)
        except Exception:
            continue

    for col in date_cols:
        df_fe[f"{col}_year"] = df_fe[col].dt.year
        df_fe[f"{col}_month"] = df_fe[col].dt.month
        df_fe[f"{col}_day"] = df_fe[col].dt.day
        new_features.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
        df_fe.drop(columns=[col], inplace=True)

    return df_fe, new_features


# ====================================================
# 4. REDUCTION DE DIMENSION
# ====================================================
def reduce_dimensionality(
    X: pd.DataFrame,
    y: pd.Series = None,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    pca_variance_ratio: float = 0.95
) -> tuple[pd.DataFrame, list[str]]:
    
    steps = []
    X_red = X.copy()

    # (1) Filtrage selon la corrélation avec la cible
    if y is not None and pd.api.types.is_numeric_dtype(y):
        correlations = X_red.corrwith(y).abs()
        low_corr_features = correlations[correlations < 0.05].index.tolist()
        if low_corr_features:
            X_red.drop(columns=low_corr_features, inplace=True)
            steps.append(f"Suppression des variables peu corrélées : {', '.join(low_corr_features)}")

    # (2) Supprimer les variables fortement corrélées entre elles
    corr_matrix = X_red.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    if to_drop:
        X_red.drop(columns=to_drop, inplace=True)
        steps.append(f"Suppression variables fortement corrélées : {', '.join(to_drop)}")

    # (3) Supprimer colonnes à faible variance
    selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = selector.fit_transform(X_red)
    selected_cols = X_red.columns[selector.get_support()]
    dropped_cols = list(set(X_red.columns) - set(selected_cols))
    X_red = pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
    if dropped_cols:
        steps.append(f"Suppression faible variance : {', '.join(dropped_cols)}")

    # (4) PCA
    if X_red.shape[1] > 2:
        pca = PCA(n_components=pca_variance_ratio, svd_solver='full')
        X_pca = pca.fit_transform(X_red)
        pca_columns = [f"PCA_{i+1}" for i in range(X_pca.shape[1])]
        X_red = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        steps.append(f"PCA appliqué : {len(pca_columns)} composantes retenues ({pca_variance_ratio*100}% variance).")

    return X_red, steps


# ====================================================
# 5. TRANSFORMATION DU DATASET
# ====================================================
def transform_dataset(
    df: pd.DataFrame,
    target_column: str = None,
    categorical_encoding: str = "auto",
    numeric_scaling: str = "standard"
) -> tuple[pd.DataFrame, pd.Series, list[str], str]:
    df_transformed = df.copy()
    transformed_columns = []

    y = None
    y_encoded = None
    if target_column and target_column in df_transformed.columns:
        y = df_transformed.pop(target_column)

        # Si y est non numérique, on l'encode pour la corrélation
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = pd.Series(le.fit_transform(y), name=target_column, index=y.index)
        else:
            y_encoded = y.copy()

    # Feature Engineering
    df_transformed, fe_cols = feature_engineering(df_transformed)
    if fe_cols:
        transformed_columns.append(f"Feature engineering : {', '.join(fe_cols)}")

    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_transformed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Imputation
    if numeric_cols:
        df_transformed[numeric_cols] = df_transformed[numeric_cols].fillna(df_transformed[numeric_cols].median())
    if categorical_cols:
        df_transformed[categorical_cols] = df_transformed[categorical_cols].fillna("Inconnu")

    # Encodage
    encoded_parts = []
    for col in categorical_cols:
        n_uniques = df_transformed[col].nunique()

        if categorical_encoding == "label" or (categorical_encoding == "auto" and n_uniques <= 5):
            le = LabelEncoder()
            encoded_parts.append(pd.Series(le.fit_transform(df_transformed[col]), name=col, index=df_transformed.index))
            transformed_columns.append(f"{col} (LabelEncoded)")
        else:
            onehot = pd.get_dummies(df_transformed[col], prefix=col)
            encoded_parts.append(onehot)
            transformed_columns.append(f"{col} (OneHotEncoded): {', '.join(onehot.columns)}")

    encoded_df = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df_transformed.index)

    # Scaling
    if numeric_cols:
        scaler = StandardScaler() if numeric_scaling == "standard" else MinMaxScaler()
        scaled_values = scaler.fit_transform(df_transformed[numeric_cols])
        scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df_transformed.index)
        transformed_columns.extend([f"{col} ({scaler.__class__.__name__})" for col in numeric_cols])
    else:
        scaled_df = pd.DataFrame(index=df_transformed.index)

    # Dataset des features
    X = pd.concat([scaled_df, encoded_df], axis=1)

    # ----- Ajout de la target encodée dans la matrice de corrélation -----
    if y_encoded is not None:
        X_corr = pd.concat([X, y_encoded], axis=1)
    else:
        X_corr = X.copy()

    # Calcul de la matrice de corrélation (features + target)
    corr_matrix = compute_correlation_matrix(X_corr)
    corr_image = generate_correlation_heatmap(corr_matrix)

    # Réduction de dimension
    X, reduction_steps = reduce_dimensionality(X, y)
    transformed_columns.extend(reduction_steps)

    return X, y, transformed_columns, corr_image


# ====================================================
# 6. VUE DE TRANSFORMATION
# ====================================================
def transformer_colonnes_view(request: HttpRequest) -> HttpResponse:
    context = {}
    try:
        data = load_dataset_from_session(request)
        target_column = request.session.get("target_column")

        X, y, colonnes_transformees, corr_image = transform_dataset(
            data,
            target_column=target_column,
            categorical_encoding="auto",
            numeric_scaling="standard"
        )

        final_df = pd.concat([X, y], axis=1) if y is not None else X
        request.session[SESSION_KEYS["data_transformed"]] = final_df.to_json()

        context.update({
            "rows": final_df.shape[0],
            "cols": final_df.shape[1],
            "columns": list(final_df.columns),
            "colonnes_transformees": colonnes_transformees,
            "correlation_image": corr_image,
            "success": "Transformation + Sélection et Réduction de dimension effectuées avec succès."
        })

    except ValidationError as ve:
        context["error"] = str(ve)
    except Exception as e:
        context["error"] = f"Erreur transformation : {e}"
        context["traceback"] = traceback.format_exc()

    return render(request, "pages/resultat_transformation.html", context)

# ====================================================
# 7. TÉLÉCHARGEMENT DU DATASET TRANSFORMÉ
# ====================================================
def telecharger_dataset_transforme(request: HttpRequest) -> HttpResponse:
    
    try:
        df = load_transformed_dataset(request)

        buffer = io.StringIO()
        df.to_csv(buffer, index=False, sep=';', encoding='utf-8')
        buffer.seek(0)

        response = HttpResponse(buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=dataset_transforme.csv'
        return response

    except ValidationError as ve:
        return HttpResponse(f"Erreur : {ve}", status=400)
    except Exception as e:
        return HttpResponse(f"Erreur interne : {e}", status=500)

"""

import traceback
import io
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.core.exceptions import ValidationError
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from .views_importation import (
    load_dataset_from_session,
    load_transformed_dataset,
    save_dataset_to_session,
    SESSION_KEYS
)

# ====================================================
# Détection améliorée du type d'analyse ML
# ====================================================
def detect_ml_type(df: pd.DataFrame, target_column: str | None) -> str:
    """
    Détecte automatiquement le type d'analyse ML avec des heuristiques robustes :
    - classification
    - regression
    - clustering (si pas de target)
    """
    if not target_column:
        return "clustering"

    y = df[target_column]
    n_samples = len(y)
    n_unique = y.nunique()

    # Classification si object ou category
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"

    # Classification si binaire (0/1 ou True/False)
    if n_unique <= 2:
        return "classification"

    # Si peu de classes par rapport au nombre d'échantillons
    if np.issubdtype(y.dtype, np.number):
        unique_ratio = n_unique / n_samples
        if unique_ratio < 0.05:  # moins de 5% de valeurs uniques
            return "classification"
        else:
            return "regression"

    # Par défaut
    return "classification"


# ====================================================
# 1. CHOIX DE LA TARGET
# ====================================================
def choix_cible_view(request: HttpRequest) -> HttpResponse:
    context = {}
    try:
        df = load_dataset_from_session(request)
        colonnes = df.columns.tolist()
        context["colonnes"] = colonnes

        previous_target = request.session.get("target_column")
        context["previous_target"] = previous_target

        if request.method == "POST":
            selected_target = request.POST.get("selected_target", "").strip()

            if not selected_target or selected_target == "aucune":
                request.session["target_column"] = None
                request.session["target_data"] = None

                ml_type = detect_ml_type(df, None)
                request.session["ml_type"] = ml_type

                context.update({
                    "selected_target": None,
                    "features_preview": colonnes,
                    "target_preview": [],
                    "ml_type": ml_type,
                    "success": "Aucune colonne cible sélectionnée. Mode clustering activé."
                })
            else:
                if selected_target not in colonnes:
                    raise ValidationError(f"La colonne '{selected_target}' n'existe pas dans le dataset.")

                y = df[selected_target]
                request.session["target_column"] = selected_target
                request.session["target_data"] = y.to_json()

                ml_type = detect_ml_type(df, selected_target)
                request.session["ml_type"] = ml_type

                context.update({
                    "selected_target": selected_target,
                    "features_preview": [col for col in colonnes if col != selected_target],
                    "target_preview": [selected_target],
                    "ml_type": ml_type,
                    "success": f"Colonne cible '{selected_target}' sélectionnée. Mode {ml_type} détecté."
                })

        return render(request, "pages/choix_cible.html", context)

    except ValidationError as ve:
        context["error"] = str(ve)
    except Exception as e:
        context["error"] = f"Erreur interne : {e}"
        context["traceback"] = traceback.format_exc()

    return render(request, "pages/choix_cible.html", context)

# ====================================================
# 2. FONCTION DE MATRICE DE CORRÉLATION
# ====================================================
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation pour les colonnes numériques.
    """
    if df.empty:
        return pd.DataFrame()
    return df.corr()

def generate_correlation_heatmap(corr_matrix: pd.DataFrame) -> str:
    """
    Génère une image de heatmap de la matrice de corrélation au format base64.
    """
    if corr_matrix.empty:
        return ""

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Matrice de Corrélation", fontsize=14)
    plt.tight_layout()

    # Sauvegarder dans un buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Encoder en base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


# ====================================================
# 3. REDUCTION DE DIMENSION
# ====================================================
def reduce_dimensionality(
    X: pd.DataFrame,
    y: pd.Series = None,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    pca_variance_ratio: float = 0.95
) -> tuple[pd.DataFrame, list[str]]:
    """
    Réduction de dimension :
    1. Supprime les variables peu corrélées avec la cible (si y est numérique).
    2. Supprime les variables fortement corrélées entre elles.
    3. Supprime les variables à faible variance.
    4. Applique PCA pour réduire la dimension tout en conservant pca_variance_ratio de variance cumulée.
    """
    steps = []
    X_red = X.copy()

    # (1) Filtrage selon la corrélation avec la cible
    if y is not None and pd.api.types.is_numeric_dtype(y):
        correlations = X_red.corrwith(y).abs()
        low_corr_features = correlations[correlations < 0.05].index.tolist()
        if low_corr_features:
            X_red.drop(columns=low_corr_features, inplace=True)
            steps.append(f"Suppression des variables peu corrélées : {', '.join(low_corr_features)}")

    # (2) Supprimer les variables fortement corrélées entre elles
    corr_matrix = X_red.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    if to_drop:
        X_red.drop(columns=to_drop, inplace=True)
        steps.append(f"Suppression variables fortement corrélées : {', '.join(to_drop)}")

    # (3) Supprimer colonnes à faible variance
    selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = selector.fit_transform(X_red)
    selected_cols = X_red.columns[selector.get_support()]
    dropped_cols = list(set(X_red.columns) - set(selected_cols))
    X_red = pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
    if dropped_cols:
        steps.append(f"Suppression faible variance : {', '.join(dropped_cols)}")

    # (4) PCA
    if X_red.shape[1] > 2:
        pca = PCA(n_components=pca_variance_ratio, svd_solver='full')
        X_pca = pca.fit_transform(X_red)
        pca_columns = [f"PCA_{i+1}" for i in range(X_pca.shape[1])]
        X_red = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        steps.append(f"PCA appliqué : {len(pca_columns)} composantes retenues ({pca_variance_ratio*100}% variance).")

    return X_red, steps


# ====================================================
# 4. TRANSFORMATION DU DATASET
# ====================================================
def transform_dataset(
    df: pd.DataFrame,
    target_column: str = None,
    categorical_encoding: str = "auto",
    numeric_scaling: str = "standard"
) -> tuple[pd.DataFrame, pd.Series, list[str], str]:
    df_transformed = df.copy()
    transformed_columns = []

    y = None
    y_encoded = None
    if target_column and target_column in df_transformed.columns:
        y = df_transformed.pop(target_column)

        # Si y est non numérique, on l'encode pour la corrélation
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = pd.Series(le.fit_transform(y), name=target_column, index=y.index)
        else:
            y_encoded = y.copy()

    # Imputation
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_transformed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if numeric_cols:
        df_transformed[numeric_cols] = df_transformed[numeric_cols].fillna(df_transformed[numeric_cols].median())
    if categorical_cols:
        df_transformed[categorical_cols] = df_transformed[categorical_cols].fillna("Inconnu")

    # Encodage
    encoded_parts = []
    for col in categorical_cols:
        n_uniques = df_transformed[col].nunique()

        if categorical_encoding == "label" or (categorical_encoding == "auto" and n_uniques <= 5):
            le = LabelEncoder()
            encoded_parts.append(pd.Series(le.fit_transform(df_transformed[col]), name=col, index=df_transformed.index))
            transformed_columns.append(f"{col} (LabelEncoded)")
        else:
            onehot = pd.get_dummies(df_transformed[col], prefix=col)
            encoded_parts.append(onehot)
            transformed_columns.append(f"{col} (OneHotEncoded): {', '.join(onehot.columns)}")

    encoded_df = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df_transformed.index)

    # Scaling
    if numeric_cols:
        scaler = StandardScaler() if numeric_scaling == "standard" else MinMaxScaler()
        scaled_values = scaler.fit_transform(df_transformed[numeric_cols])
        scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df_transformed.index)
        transformed_columns.extend([f"{col} ({scaler.__class__.__name__})" for col in numeric_cols])
    else:
        scaled_df = pd.DataFrame(index=df_transformed.index)

    # Dataset des features
    X = pd.concat([scaled_df, encoded_df], axis=1)

    # ----- Ajout de la target encodée dans la matrice de corrélation -----
    if y_encoded is not None:
        X_corr = pd.concat([X, y_encoded], axis=1)
    else:
        X_corr = X.copy()

    # Calcul de la matrice de corrélation (features + target)
    corr_matrix = compute_correlation_matrix(X_corr)
    corr_image = generate_correlation_heatmap(corr_matrix)

    # Réduction de dimension
    X, reduction_steps = reduce_dimensionality(X, y)
    transformed_columns.extend(reduction_steps)

    return X, y, transformed_columns, corr_image


# ====================================================
# 5. VUE DE TRANSFORMATION
# ====================================================
def transformer_colonnes_view(request: HttpRequest) -> HttpResponse:
    context = {}
    try:
        data = load_dataset_from_session(request)
        target_column = request.session.get("target_column")

        X, y, colonnes_transformees, corr_image = transform_dataset(
            data,
            target_column=target_column,
            categorical_encoding="auto",
            numeric_scaling="standard"
        )

        final_df = pd.concat([X, y], axis=1) if y is not None else X
        request.session[SESSION_KEYS["data_transformed"]] = final_df.to_json()

        context.update({
            "rows": final_df.shape[0],
            "cols": final_df.shape[1],
            "columns": list(final_df.columns),
            "colonnes_transformees": colonnes_transformees,
            "correlation_image": corr_image,
            "success": "Transformation + Sélection et Réduction de dimension effectuées avec succès."
        })

    except ValidationError as ve:
        context["error"] = str(ve)
    except Exception as e:
        context["error"] = f"Erreur transformation : {e}"
        context["traceback"] = traceback.format_exc()

    return render(request, "pages/resultat_transformation.html", context)

# ====================================================
# 6. TÉLÉCHARGEMENT DU DATASET TRANSFORMÉ
# ====================================================
def telecharger_dataset_transforme(request: HttpRequest) -> HttpResponse:
    """
    Télécharge le dataset transformé stocké en session, au format CSV.
    """
    try:
        df = load_transformed_dataset(request)

        buffer = io.StringIO()
        df.to_csv(buffer, index=False, sep=';', encoding='utf-8')
        buffer.seek(0)

        response = HttpResponse(buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=dataset_transforme.csv'
        return response

    except ValidationError as ve:
        return HttpResponse(f"Erreur : {ve}", status=400)
    except Exception as e:
        return HttpResponse(f"Erreur interne : {e}", status=500)