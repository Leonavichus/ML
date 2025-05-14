"""
Модуль для обнаружения аномальных транзакций с использованием различных методов машинного обучения.
Включает три основных алгоритма:
- Isolation Forest (изоляционный лес)
- One-Class SVM (одноклассовый метод опорных векторов)
- Local Outlier Factor (локальный уровень выброса)
"""

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from typing import Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пути для сохранения обученных моделей
IFOREST_PATH = "models/anomaly/isolation_forest.pkl"
OCSVM_PATH   = "models/anomaly/ocsvm.pkl"
LOF_PATH     = "models/anomaly/lof.pkl"

# Определение признаков для обработки
NUMERICAL   = ["amount", "hour_of_day", "session_duration", "login_frequency", "risk_score"]
CATEGORICAL = ["transaction_type", "location_region", "ip_prefix", "purchase_pattern", "age_group"]

def build_preprocessor() -> ColumnTransformer:
    """
    Создает пайплайн предобработки данных.
    
    Числовые признаки: заполнение медианой и стандартизация
    Категориальные признаки: заполнение модой и one-hot кодирование
    
    Returns:
        ColumnTransformer: сконфигурированный трансформер данных
    """
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

def train_iforest(df: pd.DataFrame, save: bool = True) -> Tuple[ColumnTransformer, IsolationForest]:
    """
    Обучает модель Isolation Forest для поиска аномалий.
    
    Args:
        df: DataFrame с данными для обучения
        save: сохранить ли модель на диск
    
    Returns:
        Tuple[preprocessor, model]: предобработчик и обученная модель
    """
    logger.info("Начало обучения Isolation Forest...")
    pre = build_preprocessor()
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1  # Используем все доступные ядра
    )
    model.fit(Xp)

    if save:
        logger.info(f"Сохранение модели в {IFOREST_PATH}")
        joblib.dump((pre, model), IFOREST_PATH)
    return pre, model

# One-Class SVM
def train_ocsvm(df: pd.DataFrame, save: bool = True):
    pre = build_preprocessor()
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.fit_transform(X)

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    model.fit(Xp)

    if save:
        joblib.dump((pre, model), OCSVM_PATH)
    return pre, model

# Local Outlier Factor
def train_lof(df: pd.DataFrame, save: bool = True):
    pre = build_preprocessor()
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.fit_transform(X)

    # LOF не имеет отдельного fit/predict: fit_predict сразу
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    labels = model.fit_predict(Xp)
    # Для единообразия сохраним предикты и скоринг
    # decision_function: чем ниже, тем более аномалия
    scores = model.negative_outlier_factor_

    if save:
        joblib.dump((pre, model), LOF_PATH)
    return pre, model

# Загрузка
def load_model(name: str):
    if name == "Isolation Forest":
        return joblib.load(IFOREST_PATH)
    if name == "One-Class SVM":
        return joblib.load(OCSVM_PATH)
    if name == "Local Outlier Factor":
        return joblib.load(LOF_PATH)
    raise ValueError(f"Неизвестный метод {name}")

def enrich(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Обогащает датафрейм результатами определения аномалий.
    
    Args:
        df: исходный DataFrame
        name: название метода ('Isolation Forest', 'One-Class SVM', 'Local Outlier Factor')
    
    Returns:
        DataFrame с добавленными колонками:
        - AnomalyScore: числовая оценка аномальности
        - AnomalyPred: бинарная метка (1 - норма, -1 - аномалия)
    
    Raises:
        ValueError: если указано неизвестное название метода
    """
    try:
        pre, model = load_model(name)
        X = df[NUMERICAL + CATEGORICAL]
        Xp = pre.transform(X)
        
        df2 = df.copy()
        
        if name == "Isolation Forest":
            scores = model.decision_function(Xp)
            preds = model.predict(Xp)
        elif name == "One-Class SVM":
            scores = model.score_samples(Xp)
            preds = model.predict(Xp)
        else:  # Local Outlier Factor
            lof = model
            scores = lof.negative_outlier_factor_
            preds = lof.fit_predict(Xp)

        df2["AnomalyScore"] = scores
        df2["AnomalyPred"] = preds
        
        # Добавляем понятную метку аномалии
        df2["IsAnomaly"] = df2["AnomalyPred"].map({1: "Норма", -1: "Аномалия"})
        
        return df2
        
    except Exception as e:
        logger.error(f"Ошибка при обогащении данных: {str(e)}")
        raise