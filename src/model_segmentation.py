import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

# Пути для сохранения
SEG_KMEANS_PATH = "models/segmentation/kmeans_segmentation.pkl"
SEG_GMM_PATH    = "models/segmentation/gmm_segmentation.pkl"
SEG_NN_PATH     = "models/segmentation/nn_segmentation.pkl"


def compute_age(dob_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dob_series, format='%d/%m/%y', errors='coerce', dayfirst=True)
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            dob_series[mask], format='%d/%m/%Y', errors='coerce', dayfirst=True
        )
    return pd.Timestamp.today().year - parsed.dt.year


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    num_cols = [f for f in ['CustAccountBalance', 'TransactionAmount', 'Age'] if f in features]
    cat_small = [f for f in ['CustGender'] if f in features]
    cat_large = [f for f in ['CustLocation'] if f in features]

    num_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    cat_small_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    )
    cat_large_pipe = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=''),
        FeatureHasher(n_features=256, input_type='string')
    )

    return ColumnTransformer([
        ('num',       num_pipe,       num_cols),
        ('cat_small', cat_small_pipe, cat_small),
        ('cat_hash',  cat_large_pipe, cat_large),
    ], remainder='drop')


def train_kmeans(df: pd.DataFrame, n_clusters: int = 4, features: list[str] = None, save: bool = True):
    """Обучает preprocessor + KMeans и сохраняет их."""
    df2 = df.copy()
    df2['Age'] = compute_age(df2['CustomerDOB'])
    features = features or ['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']
    X = df2[features]

    pre = build_preprocessor(features)
    X_proc = pre.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X_proc)

    if save:
        joblib.dump((pre, km, features), SEG_KMEANS_PATH)
    return pre, km, features


def train_gmm(df: pd.DataFrame, n_clusters: int = 4, features: list[str] = None, save: bool = True):
    """Обучает preprocessor + GaussianMixture и сохраняет."""
    df2 = df.copy()
    df2['Age'] = compute_age(df2['CustomerDOB'])
    features = features or ['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']
    X = df2[features]

    pre = build_preprocessor(features)
    X_proc = pre.fit_transform(X).toarray()  # гарантированно dense

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X_proc)

    if save:
        joblib.dump((pre, gmm, features), SEG_GMM_PATH)
    return pre, gmm, features


def train_nn_segmentation(df: pd.DataFrame, n_clusters: int = 4, features: list[str] = None, save: bool = True):
    """
    Имитация сегментации нейросетью:
      1) Берём метки KMeans
      2) Обучаем MLPClassifier на этих метках
    """
    df2 = df.copy()
    df2['Age'] = compute_age(df2['CustomerDOB'])

    # Используем либо указанные признаки, либо дефолтные
    features = features or ['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']
    X = df2[features]

    # Обучаем KMeans как teacher
    pre = build_preprocessor(features)
    X_proc = pre.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X_proc)
    labels = km.predict(X_proc)

    # Обучаем MLPClassifier на тех же признаках
    X_dense = X_proc.toarray() if hasattr(X_proc, 'toarray') else X_proc
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_dense, labels)

    if save:
        joblib.dump((pre, mlp, features), SEG_NN_PATH)

    return pre, mlp, features

def load_segmentation_model(method: str):
    """Загружает (preprocessor, model) по ключу."""
    path_map = {
        'kmeans': SEG_KMEANS_PATH,
        'gmm':    SEG_GMM_PATH,
        'nn':     SEG_NN_PATH
    }
    if method not in path_map:
        raise ValueError(f"Неизвестный метод '{method}'")
    try:
        return joblib.load(path_map[method])
    except FileNotFoundError:
        raise FileNotFoundError("Сначала обучите модель методом " + method)


def enrich_segmentation(df: pd.DataFrame, method: str, model_tuple=None) -> pd.DataFrame:
    """"
    Обогащает df колонками Segment (число) и SegmentName (читаемая метка).
    """
    if model_tuple is None:
        pre, model, features = load_segmentation_model(method)
    else:
        pre, model, features = model_tuple

    df2 = df.copy()
    if "CustomerDOB" in df2.columns and "Age" in features:
        df2['Age'] = compute_age(df2['CustomerDOB'])

    X = df2[features]
    X_proc = pre.transform(X)
    X_arr = X_proc.toarray() if hasattr(X_proc, 'toarray') else X_proc

    labels = model.predict(X_arr)
    df2['Segment'] = labels

    segment_cols = [col for col in features if df2[col].dtype.kind in "iuf"]
    profile = df2.groupby('Segment')[segment_cols].mean().round(1).reset_index()

    def describe_segment(row):
        return f"#{int(row.Segment)}: " + ", ".join([f"{col}~{row[col]}" for col in segment_cols])

    profile['SegmentName'] = profile.apply(describe_segment, axis=1)
    df2['SegmentName'] = df2['Segment'].map(dict(zip(profile['Segment'], profile['SegmentName'])))

    return df2
