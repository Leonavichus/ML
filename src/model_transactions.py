import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Union, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator
from logging import getLogger
import os

logger = getLogger(__name__)

# Пути для сохранения моделей
MODELS_DIR = "models/anomaly"
IFOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
OCSVM_PATH = os.path.join(MODELS_DIR, "ocsvm.pkl")
LOF_PATH = os.path.join(MODELS_DIR, "lof.pkl")

# Конфигурация полей
NUMERICAL = [
    "amount", "hour_of_day", "session_duration",
    "login_frequency", "risk_score"
]
CATEGORICAL = [
    "transaction_type", "location_region", "ip_prefix",
    "purchase_pattern", "age_group"
]

def ensure_model_dir() -> None:
    """Создает директорию для моделей, если она не существует."""
    os.makedirs(MODELS_DIR, exist_ok=True)

def build_preprocessor() -> ColumnTransformer:
    """
    Создает препроцессор данных для числовых и категориальных признаков.
    
    Возвращает:
        ColumnTransformer: Подготовленный препроцессор с StandardScaler для числовых
                         и OneHotEncoder для категориальных признаков
    """
    try:
        num_pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )
        cat_pipe = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )
    return ColumnTransformer([
        ("num", num_pipe, NUMERICAL),
        ("cat", cat_pipe, CATEGORICAL),
    ], remainder="drop")
    except Exception as e:
        logger.error(f"Ошибка при создании препроцессора: {str(e)}")
        raise

def validate_input_data(df: pd.DataFrame) -> None:
    """
    Проверяет входные данные на наличие необходимых колонок.
    
    Аргументы:
        df: Входной датафрейм
        
    Вызывает:
        ValueError: Если отсутствуют необходимые колонки
    """
    missing_num = set(NUMERICAL) - set(df.columns)
    missing_cat = set(CATEGORICAL) - set(df.columns)
    if missing_num or missing_cat:
        msg = "Отсутствуют колонки: " + ", ".join(missing_num | missing_cat)
        logger.error(msg)
        raise ValueError(msg)

def train_model(df: pd.DataFrame,
                model: BaseEstimator,
                save_path: str,
                save: bool = True) -> Tuple[ColumnTransformer, BaseEstimator]:
    """
    Общая функция для обучения моделей обнаружения аномалий.
    
    Аргументы:
        df: Входной датафрейм
        model: Модель для обучения
        save_path: Путь для сохранения
        save: Флаг сохранения модели
        
    Возвращает:
        Tuple[ColumnTransformer, BaseEstimator]: Препроцессор и обученная модель
    """
    try:
        validate_input_data(df)
    pre = build_preprocessor()
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.fit_transform(X)

        if isinstance(model, LocalOutlierFactor):
            model.fit_predict(Xp)  # LOF требует fit_predict
        else:
    model.fit(Xp)

    if save:
            ensure_model_dir()
            joblib.dump((pre, model), save_path)
            logger.info(f"Модель сохранена в {save_path}")
            
    return pre, model
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

def train_iforest(df: pd.DataFrame, save: bool = True) -> Tuple[ColumnTransformer, IsolationForest]:
    """Обучает Isolation Forest для обнаружения аномалий."""
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    return train_model(df, model, IFOREST_PATH, save)

def train_ocsvm(df: pd.DataFrame, save: bool = True) -> Tuple[ColumnTransformer, OneClassSVM]:
    """Обучает One-Class SVM для обнаружения аномалий."""
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.01,
        cache_size=500
    )
    return train_model(df, model, OCSVM_PATH, save)

def train_lof(df: pd.DataFrame, save: bool = True) -> Tuple[ColumnTransformer, LocalOutlierFactor]:
    """Обучает Local Outlier Factor для обнаружения аномалий."""
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01,
        n_jobs=-1
    )
    return train_model(df, model, LOF_PATH, save)

def load_model(name: str) -> Tuple[ColumnTransformer, BaseEstimator]:
    """
    Загружает сохраненную модель по имени.
    
    Аргументы:
        name: Название модели
        
    Возвращает:
        Tuple[ColumnTransformer, BaseEstimator]: Препроцессор и модель
        
    Вызывает:
        ValueError: Если модель не найдена или имя неизвестно
    """
    try:
    if name == "Isolation Forest":
            path = IFOREST_PATH
        elif name == "One-Class SVM":
            path = OCSVM_PATH
        elif name == "Local Outlier Factor":
            path = LOF_PATH
        else:
    raise ValueError(f"Неизвестный метод {name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена: {path}")
            
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {name}: {str(e)}")
        raise

def enrich(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Обогащает датафрейм предсказаниями аномалий.
    
    Аргументы:
        df: Входной датафрейм
        name: Название модели
        
    Возвращает:
        pd.DataFrame: Датафрейм с добавленными колонками AnomalyScore и AnomalyPred
    """
    try:
        validate_input_data(df)
    pre, model = load_model(name)
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.transform(X)

        if name == "Local Outlier Factor":
            scores = model.negative_outlier_factor_
            preds = model.fit_predict(Xp)
        else:
            scores = model.decision_function(Xp) if name == "Isolation Forest" else model.score_samples(Xp)
            preds = model.predict(Xp)

        df_enriched = df.copy()
        df_enriched["AnomalyScore"] = scores
        df_enriched["AnomalyPred"] = preds
        
        # Добавляем метаданные
        df_enriched.attrs["anomaly_model"] = name
        df_enriched.attrs["anomaly_timestamp"] = pd.Timestamp.now()
        
        return df_enriched
    except Exception as e:
        logger.error(f"Ошибка при обогащении данных: {str(e)}")
        raise
