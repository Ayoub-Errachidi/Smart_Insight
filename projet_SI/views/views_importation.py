import logging
import io
import pandas as pd
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.http import HttpRequest, HttpResponse
import re
import chardet


# -------------------- Configuration --------------------

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.json')
FORBIDDEN_EXTENSIONS = ('.xlsm',)
MAX_FILE_SIZE_MB = 40  # Taille maximale des fichiers (40 Mo)

SESSION_KEYS = {
    "data": "dataset",  # Dataset original
    "data_transformed": "dataset_transformed",  # Dataset transformé
    "filename": "filename",
    "file_type": "file_type",
    "encoding": "file_encoding"
}

# -------------------- Validation et Analyse --------------------

def validate_uploaded_file(file) -> None:
    '''
    Vérifie que le fichier est valide en termes de format et de taille.
    '''
    if not file:
        raise ValidationError("Aucun fichier sélectionné.")
    
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValidationError(f"Le fichier dépasse la taille limite de {MAX_FILE_SIZE_MB} Mo.")

    if file.name.endswith(FORBIDDEN_EXTENSIONS):
        raise ValidationError("Les fichiers Excel contenant des macros (.xlsm) ne sont pas autorisés.")
    
    if not file.name.endswith(SUPPORTED_EXTENSIONS):
        raise ValidationError("Format non supporté. Veuillez importer un fichier CSV, Excel ou JSON.")

# -------------------- Clean Data --------------------
def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les chaînes considérées comme nulles par des NaN réels.
    """
    return df.replace(
        to_replace=["", " ", "NA", "NaN", "nan", "NULL", "null", "None"],
        value=pd.NA
    )

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Met les noms de colonnes en minuscules et en snake_case.
    """
    def to_snake_case(name):
        name = str(name).strip().lower()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name

    df.columns = [to_snake_case(col) for col in df.columns]
    return df

# -------------------- Normalisation des mots --------------------
import unicodedata

def normalize_word_variants(value: str) -> str:
    """
    Corrige les variantes d'un mot :
    - supprime les accents
    - met en minuscules
    """
    if not isinstance(value, str):
        return value
    
    # Supprimer les accents
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(c for c in value if not unicodedata.combining(c))
    
    return value.lower().strip()

def uniformize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Met toutes les valeurs textuelles (catégorielles) en minuscules
    et applique normalize_word_variants.
    """
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str).apply(normalize_word_variants)
    return df

# -------------------- Pipeline de nettoyage --------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de nettoyage global :
    1. Standardiser valeurs manquantes
    2. Normaliser les noms de colonnes
    3. Uniformiser les catégories
    """
    df = standardize_missing_values(df)
    df = standardize_column_names(df)
    df = uniformize_categories(df)
    return df

# -------------------- Helpers encodage --------------------

def detect_encoding(file) -> tuple[str, bytes]:
    """
    Détecte l'encodage d'un fichier et renvoie (encoding, raw_data).
    """
    raw_data = file.read()
    detected = chardet.detect(raw_data)
    encoding_used = detected['encoding'] or 'utf-8'
    file.seek(0)  # Reset du curseur
    return encoding_used, raw_data

def parse_uploaded_file(file) -> pd.DataFrame:
    '''
    Lit le fichier en un DataFrame selon son type, et nettoie les faux NaN.
    '''
    try:
        encoding_used = 'utf-8'
        file_name = file.name.lower()
        
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8', na_values=["", " ", "NA", "NaN", "null", "None"])
                encoding_used = 'utf-8'
            except UnicodeDecodeError:
                file.seek(0)
                encoding_used, raw_data = detect_encoding(file)
                df = pd.read_csv(io.StringIO(raw_data.decode(encoding_used)), na_values=["", " ", "NA", "NaN", "null", "None"])
        
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=["", " ", "NA", "NaN", "null", "None"])
            encoding_used = 'binary'
        
        elif file_name.endswith('.json'):
            df = pd.read_json(file)
            encoding_used = 'utf-8'
            df = standardize_missing_values(df)  # nettoyage manuel des fausses valeurs nulles
            if not isinstance(df, pd.DataFrame):
                raise ValidationError("Le JSON ne contient pas de tableau valide.")
            return df, encoding_used
        
        else:
            raise ValidationError("Type de fichier non supporté.")
        
        # Nettoyage commun à tous les formats
        #df = standardize_missing_values(df)
        #df = standardize_column_names(df)

        #df = uniformize_categories(df)

        df = clean_dataset(df)

        return df, encoding_used
    
    except Exception as e:
        logger.exception("Erreur lecture : %s", file.name)
        raise ValidationError("Le fichier est corrompu ou mal structuré.")

# -------------------- Session helpers --------------------

def save_dataset_to_session(request: HttpRequest, data: pd.DataFrame, file_name: str, encoding: str = None) -> None:
    
    ''' Sauvegarde le dataset et ses métadonnées dans la session '''

     # Sauvegarde le dataset
    request.session[SESSION_KEYS["data"]] = data.to_json()

    # Vérifie s’il y a une extension
    if '.' in file_name:
        name, ext = file_name.rsplit('.', 1)
        request.session[SESSION_KEYS["filename"]] = name
        request.session[SESSION_KEYS["file_type"]] = ext.lower()
    else:
        # Aucun point => pas d'extension détectée
        request.session[SESSION_KEYS["filename"]] = file_name
        request.session[SESSION_KEYS["file_type"]] = "inconnu"

    if encoding:
        request.session[SESSION_KEYS["encoding"]] = encoding

def load_dataset_from_session(request: HttpRequest) -> pd.DataFrame:
    '''
    Charge le DataFrame original depuis la session.
    '''
    json_data = request.session.get(SESSION_KEYS["data"])
    if not json_data:
        raise ValidationError("Aucun dataset original disponible dans la session.")
    return pd.read_json(io.StringIO(json_data))

def load_transformed_dataset(request: HttpRequest) -> pd.DataFrame:
    json_data = request.session.get(SESSION_KEYS["data_transformed"])
    if not json_data:
        raise ValidationError("Aucun dataset transformé disponible dans la session. Avez-vous effectué une transformation ?")
    return pd.read_json(io.StringIO(json_data))

# -------------------- Contexte HTML --------------------

def build_dataset_context(data: pd.DataFrame, request: HttpRequest) -> dict:
    
    ''' Construit le contexte à passer au template Dataset.html '''

    return {
        "data": data.to_html(classes="table table-striped table-hover"),
        "head_data": data.head().to_json(),
        "columns": data.shape[1],
        "lignes": data.shape[0],
        "filename": request.session.get(SESSION_KEYS["filename"], "Fichier inconnu"),
        "file_type": request.session.get(SESSION_KEYS["file_type"], "inconnu"),
        "encoding": request.session.get(SESSION_KEYS["encoding"], "inconnu")
    }

# -------------------- Vues --------------------
def dashboard_view(request: HttpRequest) -> HttpResponse:
    
    ''' Vue d'accueil : permet d'uploader un fichier et de visualiser son contenu '''

    if request.method == "POST":
        uploaded_file = request.FILES.get("type_file")
        try:
            validate_uploaded_file(uploaded_file)
            df, encoding = parse_uploaded_file(uploaded_file)
            
            save_dataset_to_session(request, df, uploaded_file.name, encoding)
            return render(request, "pages/Dataset.html", build_dataset_context(df, request))
        except ValidationError as ve:
            logger.warning("Fichier invalide : %s", ve)
            return render(request, "pages/dashboard.html", {"error": str(ve)})
        except Exception as e:
            logger.exception("Erreur inattendue upload")
            return render(request, "pages/dashboard.html", {"error": "Erreur interne lors de l’upload du fichier."})
    return render(request, "pages/dashboard.html")


def dataset_view(request: HttpRequest) -> HttpResponse:
    
    '''Vue dédiée à l'affichage du dernier dataset importé '''

    try:
        df = load_dataset_from_session(request)
        return render(request, "pages/Dataset.html", build_dataset_context(df, request))
    except ValidationError as ve:
        return render(request, "pages/Dataset.html", {"error": str(ve)})
    except Exception:
        logger.exception("Erreur interne : dataset_view")
        return render(request, "pages/Dataset.html", {"error": "Erreur interne lors du chargement du dataset."})
    
