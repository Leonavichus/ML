import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

# Пути к моделям сегментации
SEG_KMEANS_PATH = "models/segmentation/kmeans_segmentation.pkl"
SEG_HIER_PATH = "models/segmentation/hierarchical_segmentation.pkl"
SEG_GMM_PATH = "models/segmentation/gmm_segmentation.pkl"
SEG_NN_PATH = "models/segmentation/nn_segmentation.pkl"

def compute_age(dob_series):
    """
    Парсит даты рождения в форматах 'DD/MM/YY' или 'DD/MM/YYYY' и возвращает возраст в годах.
    """
    # Попытка парсинга формата 'DD/MM/YY'
    parsed = pd.to_datetime(dob_series, format='%d/%m/%y', errors='coerce', dayfirst=True)
    # Для элементов с ошибками парсинга, пробуем 'DD/MM/YYYY'
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            dob_series[mask], format='%d/%m/%Y', errors='coerce', dayfirst=True
        )
    current_year = pd.Timestamp.today().year
    return current_year - parsed.dt.year

# Общий препроцессор для сегментации
def build_segmentation_preprocessor():
    num_cols = ['CustAccountBalance', 'TransactionAmount', 'Age']
    cat_cols = ['CustGender', 'CustLocation']
    num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')

# === Обучение различных моделей сегментации ===

def train_kmeans(df: pd.DataFrame, n_clusters: int = 4, save: bool = True):
    df_work = df.copy()
    df_work['Age'] = compute_age(df_work['CustomerDOB'])
    X = df_work[['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']]
    pipeline = ImbPipeline([
        ('preprocessor', build_segmentation_preprocessor()),
        ('clusterer', KMeans(n_clusters=n_clusters, random_state=42))
    ])
    pipeline.fit(X)
    if save:
        joblib.dump(pipeline, SEG_KMEANS_PATH)
    return pipeline


def train_hierarchical(df: pd.DataFrame, n_clusters: int = 4, save: bool = True):
    df_work = df.copy()
    df_work['Age'] = compute_age(df_work['CustomerDOB'])
    X = df_work[['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']]
    pipeline = ImbPipeline([
        ('preprocessor', build_segmentation_preprocessor()),
        ('clusterer', AgglomerativeClustering(n_clusters=n_clusters))
    ])
    pipeline.fit(X)
    if save:
        joblib.dump(pipeline, SEG_HIER_PATH)
    return pipeline


def train_gmm(df: pd.DataFrame, n_clusters: int = 4, save: bool = True):
    df_work = df.copy()
    df_work['Age'] = compute_age(df_work['CustomerDOB'])
    X = df_work[['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']]
    pipeline = ImbPipeline([
        ('preprocessor', build_segmentation_preprocessor()),
        ('clusterer', GaussianMixture(n_components=n_clusters, random_state=42))
    ])
    pipeline.fit(X)
    if save:
        joblib.dump(pipeline, SEG_GMM_PATH)
    return pipeline


def train_nn_segmentation(df: pd.DataFrame, n_clusters: int = 4, save: bool = True):
    # Генерируем метки KMeans, затем обучаем классификатор
    kmeans_pipe = train_kmeans(df, n_clusters=n_clusters, save=False)
    df_work = df.copy()
    df_work['Age'] = compute_age(df_work['CustomerDOB'])
    X = df_work[['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']]
    labels = kmeans_pipe.named_steps['clusterer'].labels_

    pre = build_segmentation_preprocessor()
    X_trans = pre.fit_transform(X)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_trans, labels)

    pipeline = ImbPipeline([
        ('preprocessor', pre),
        ('classifier', mlp)
    ])
    if save:
        joblib.dump(pipeline, SEG_NN_PATH)
    return pipeline

# === Загрузка и обогащение ===

def load_segmentation_model(method: str):
    path_map = {
        'kmeans': SEG_KMEANS_PATH,
        'hier': SEG_HIER_PATH,
        'gmm': SEG_GMM_PATH,
        'nn': SEG_NN_PATH
    }
    try:
        return joblib.load(path_map[method])
    except FileNotFoundError:
        raise FileNotFoundError(f"Модель сегментации '{method}' не найдена. Сначала обучите метод.")

def enrich_segmentation(df: pd.DataFrame, method: str, model=None) -> pd.DataFrame:
    if model is None:
        model = load_segmentation_model(method)
    df_out = df.copy()
    df_out['Age'] = compute_age(df_out['CustomerDOB'])
    X = df_out[['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']]

    if hasattr(model, 'predict'):
        labels = model.predict(X)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['clusterer'], 'labels_'):
        labels = model.named_steps['clusterer'].labels_
    else:
        labels = model.predict(X)

    df_out['Segment'] = labels
    return df_out