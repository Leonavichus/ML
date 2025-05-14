import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Dict, Union, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from logging import getLogger
import os
from datetime import datetime

logger = getLogger(__name__)

# Пути для сохранения моделей
MODELS_DIR = "models/segmentation"
SEG_KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_segmentation.pkl")
SEG_GMM_PATH = os.path.join(MODELS_DIR, "gmm_segmentation.pkl")
SEG_NN_PATH = os.path.join(MODELS_DIR, "nn_segmentation.pkl")

# Конфигурация признаков по умолчанию
DEFAULT_FEATURES = [
    'CustAccountBalance',
    'TransactionAmount',
    'Age',
    'CustGender',
    'CustLocation'
]

def ensure_model_dir() -> None:
    """Создает директорию для моделей, если она не существует."""
    os.makedirs(MODELS_DIR, exist_ok=True)

def compute_age(dob_series: pd.Series) -> pd.Series:
    """
    Вычисляет возраст на основе даты рождения.
    
    Аргументы:
        dob_series: Серия с датами рождения
        
    Возвращает:
        pd.Series: Серия с вычисленным возрастом
        
    Примечание:
        Поддерживает форматы дат 'dd/mm/yy' и 'dd/mm/yyyy'
    """
    try:
    parsed = pd.to_datetime(dob_series, format='%d/%m/%y', errors='coerce', dayfirst=True)
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            dob_series[mask], format='%d/%m/%Y', errors='coerce', dayfirst=True
        )
    return pd.Timestamp.today().year - parsed.dt.year
    except Exception as e:
        logger.error(f"Ошибка при вычислении возраста: {str(e)}")
        raise

def build_preprocessor(features: List[str]) -> ColumnTransformer:
    """
    Создает препроцессор для числовых и категориальных признаков.
    
    Аргументы:
        features: Список используемых признаков
        
    Возвращает:
        ColumnTransformer: Подготовленный препроцессор
    """
    try:
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
            ('num', num_pipe, num_cols),
        ('cat_small', cat_small_pipe, cat_small),
            ('cat_hash', cat_large_pipe, cat_large),
    ], remainder='drop')
    except Exception as e:
        logger.error(f"Ошибка при создании препроцессора: {str(e)}")
        raise

def validate_input_data(df: pd.DataFrame, features: List[str]) -> None:
    """
    Проверяет наличие необходимых колонок во входных данных.
    
    Аргументы:
        df: Входной датафрейм
        features: Список необходимых признаков
        
    Вызывает:
        ValueError: Если отсутствуют необходимые колонки
    """
    missing = set(features) - set(df.columns)
    if missing:
        msg = f"Отсутствуют колонки: {', '.join(missing)}"
        logger.error(msg)
        raise ValueError(msg)

def train_model(
    df: pd.DataFrame,
    model: BaseEstimator,
    n_clusters: int,
    features: Optional[List[str]] = None,
    save_path: str = "",
    save: bool = True
) -> Tuple[ColumnTransformer, BaseEstimator, List[str]]:
    """
    Общая функция для обучения моделей сегментации.
    
    Аргументы:
        df: Входной датафрейм
        model: Модель для обучения
        n_clusters: Количество кластеров
        features: Список признаков (опционально)
        save_path: Путь для сохранения
        save: Флаг сохранения модели
        
    Возвращает:
        Tuple[ColumnTransformer, BaseEstimator, List[str]]: 
            Препроцессор, обученная модель и список признаков
    """
    try:
    df2 = df.copy()
    df2['Age'] = compute_age(df2['CustomerDOB'])
        
        features = features or DEFAULT_FEATURES
        validate_input_data(df2, features)
    X = df2[features]

    pre = build_preprocessor(features)
    X_proc = pre.fit_transform(X)

        # Преобразуем в dense если нужно
        if hasattr(X_proc, 'toarray'):
            X_proc = X_proc.toarray()
            
        model.fit(X_proc)

    if save:
            ensure_model_dir()
            joblib.dump((pre, model, features), save_path)
            logger.info(f"Модель сохранена в {save_path}")
            
        return pre, model, features
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

def train_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    features: Optional[List[str]] = None,
    save: bool = True
) -> Tuple[ColumnTransformer, KMeans, List[str]]:
    """Обучает KMeans для сегментации клиентов."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    return train_model(df, model, n_clusters, features, SEG_KMEANS_PATH, save)

def train_gmm(
    df: pd.DataFrame,
    n_clusters: int = 4,
    features: Optional[List[str]] = None,
    save: bool = True
) -> Tuple[ColumnTransformer, GaussianMixture, List[str]]:
    """Обучает Gaussian Mixture Model для сегментации клиентов."""
    model = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        n_init=5
    )
    return train_model(df, model, n_clusters, features, SEG_GMM_PATH, save)

def train_nn_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 4,
    features: Optional[List[str]] = None,
    save: bool = True
) -> Tuple[ColumnTransformer, MLPClassifier, List[str]]:
    """
    Обучает нейросетевую модель для сегментации клиентов.
    Использует KMeans как учителя для создания меток.
    """
    try:
        # Обучаем KMeans как учителя
        pre, km, features = train_kmeans(df, n_clusters, features, save=False)
        X = df[features]
        X_proc = pre.transform(X)
        if hasattr(X_proc, 'toarray'):
            X_proc = X_proc.toarray()
        
    labels = km.predict(X_proc)

        # Обучаем нейросеть на метках KMeans
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        mlp.fit(X_proc, labels)

    if save:
            ensure_model_dir()
        joblib.dump((pre, mlp, features), SEG_NN_PATH)
            logger.info(f"Модель сохранена в {SEG_NN_PATH}")

    return pre, mlp, features
    except Exception as e:
        logger.error(f"Ошибка при обучении нейросетевой модели: {str(e)}")
        raise

def load_segmentation_model(method: str) -> Tuple[ColumnTransformer, BaseEstimator, List[str]]:
    """
    Загружает сохраненную модель по названию метода.
    
    Аргументы:
        method: Название метода ('kmeans', 'gmm' или 'nn')
        
    Возвращает:
        Tuple[ColumnTransformer, BaseEstimator, List[str]]:
            Препроцессор, модель и список признаков
            
    Вызывает:
        ValueError: Если метод неизвестен
        FileNotFoundError: Если модель не найдена
    """
    try:
    path_map = {
        'kmeans': SEG_KMEANS_PATH,
            'gmm': SEG_GMM_PATH,
            'nn': SEG_NN_PATH
    }
    if method not in path_map:
        raise ValueError(f"Неизвестный метод '{method}'")
            
        path = path_map[method]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена: {path}")
            
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {method}: {str(e)}")
        raise

def enrich_segmentation(
    df: pd.DataFrame,
    method: str,
    model_tuple: Optional[Tuple[ColumnTransformer, BaseEstimator, List[str]]] = None
) -> pd.DataFrame:
    """
    Обогащает датафрейм результатами сегментации.
    
    Аргументы:
        df: Входной датафрейм
        method: Метод сегментации
        model_tuple: Опциональный кортеж (препроцессор, модель, признаки)
        
    Возвращает:
        pd.DataFrame: Датафрейм с добавленными колонками Segment и SegmentName
    """
    try:
    if model_tuple is None:
        pre, model, features = load_segmentation_model(method)
    else:
        pre, model, features = model_tuple

    df2 = df.copy()
    if "CustomerDOB" in df2.columns and "Age" in features:
        df2['Age'] = compute_age(df2['CustomerDOB'])

        validate_input_data(df2, features)
    X = df2[features]
    X_proc = pre.transform(X)
    X_arr = X_proc.toarray() if hasattr(X_proc, 'toarray') else X_proc

        # Получаем метки сегментов
    labels = model.predict(X_arr)
    df2['Segment'] = labels

        # Создаем описательные названия сегментов
    segment_cols = [col for col in features if df2[col].dtype.kind in "iuf"]
    profile = df2.groupby('Segment')[segment_cols].mean().round(1).reset_index()

    def describe_segment(row):
            return f"Сегмент #{int(row.Segment)}: " + ", ".join(
                [f"{col}~{row[col]}" for col in segment_cols]
            )

    profile['SegmentName'] = profile.apply(describe_segment, axis=1)
        df2['SegmentName'] = df2['Segment'].map(
            dict(zip(profile['Segment'], profile['SegmentName']))
        )
        
        # Добавляем метаданные
        df2.attrs['segmentation_method'] = method
        df2.attrs['segmentation_timestamp'] = pd.Timestamp.now()
        df2.attrs['features_used'] = features

    return df2
    except Exception as e:
        logger.error(f"Ошибка при обогащении данных: {str(e)}")
        raise
