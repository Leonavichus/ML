"""
Модуль оценки риска дефолта по кредиту с использованием методов машинного обучения.

Реализованы следующие модели:
1. Логистическая регрессия (базовая модель)
2. Нейронная сеть (расширенная модель)

Особенности реализации:
- Автоматическая предобработка числовых и категориальных признаков
- Балансировка классов с помощью SMOTE
- Подбор гиперпараметров через GridSearchCV
- Оценка качества по ROC-AUC
"""

import pandas as pd
import joblib
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from typing import Tuple, Dict, Any, Optional
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Пути для сохранения моделей
DEFAULT_MODEL_PATH = "models/default_risk/logreg_default.pkl"
DEFAULT_NN_MODEL_PATH = "models/default_risk/nn_default.pkl"

def build_default_preprocessor() -> ColumnTransformer:
    """
    Создает пайплайн предобработки данных для моделей оценки риска дефолта.
    
    Обрабатывает следующие признаки:
    
    Числовые:
    - person_age: возраст заемщика
    - person_income: годовой доход
    - person_emp_length: стаж работы
    - loan_amnt: сумма кредита
    - loan_int_rate: процентная ставка
    - loan_percent_income: отношение платежа к доходу
    - cb_person_cred_hist_length: длина кредитной истории
    
    Категориальные:
    - person_home_ownership: тип владения жильем
    - loan_intent: цель кредита
    - loan_grade: рейтинг кредита
    - cb_person_default_on_file: наличие дефолтов в прошлом
    
    Returns:
        ColumnTransformer: сконфигурированный трансформер данных
    """
    # Определение признаков по типам
    num_cols = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"
    ]
    cat_cols = [
        "person_home_ownership", "loan_intent", "loan_grade",
        "cb_person_default_on_file"
    ]
    
    # Пайплайны обработки
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler()
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def prepare_data(df: pd.DataFrame) -> tuple[DataFrame, Any] | tuple[DataFrame, None]:
    """
    Подготавливает данные для обучения или предсказания.
    
    Args:
        df: исходный датафрейм
    
    Returns:
        Tuple[X, y]: признаки и целевая переменная (если есть)
    
    Raises:
        ValueError: если для обучения нет целевой переменной
    """
    if "loan_status" in df.columns:
        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]
        return X, y
    return df.copy(), None

def train_model(
    df: pd.DataFrame,
    model_type: str,
    param_grid: Dict[str, Any],
    save: bool = True,
    model_path: str = None
) -> Tuple[ImbPipeline, GridSearchCV]:
    """
    Общая функция для обучения моделей оценки риска дефолта.
    
    Args:
        df: датафрейм с данными
        model_type: тип модели ('logreg' или 'nn')
        param_grid: сетка параметров для GridSearchCV
        save: сохранить ли модель
        model_path: путь для сохранения
    
    Returns:
        Tuple[model, grid]: обученная модель и результаты поиска параметров
    """
    logger.info(f"Начало обучения модели {model_type.upper()}")
    
    X, y = prepare_data(df)
    if y is None:
        raise ValueError("Целевая переменная 'loan_status' не найдена в данных")
    
    # Выбор базовой модели
    if model_type == 'logreg':
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'nn':
        classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
    
    # Создание пайплайна
    pipeline = ImbPipeline([
        ("preprocessor", build_default_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier)
    ])
    
    # Поиск лучших параметров
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Начало поиска параметров...")
    grid.fit(X, y)
    logger.info(f"Лучшие параметры: {grid.best_params_}")
    logger.info(f"Лучший ROC-AUC: {grid.best_score_:.4f}")
    
    if save and model_path:
        logger.info(f"Сохранение модели в {model_path}")
        joblib.dump(grid.best_estimator_, model_path)
    
    return grid.best_estimator_, grid

def train_default_model(
    df: pd.DataFrame,
    save: bool = True,
    model_path: str = DEFAULT_MODEL_PATH
) -> Tuple[ImbPipeline, GridSearchCV]:
    """
    Обучает логистическую регрессию для оценки риска дефолта.
    
    Args:
        df: датафрейм с данными
        save: сохранить ли модель
        model_path: путь для сохранения
    
    Returns:
        Tuple[model, grid]: обученная модель и результаты поиска параметров
    """
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    }
    return train_model(df, 'logreg', param_grid, save, model_path)

def train_default_nn_model(
    df: pd.DataFrame,
    save: bool = True,
    model_path: str = DEFAULT_NN_MODEL_PATH
) -> Tuple[ImbPipeline, GridSearchCV]:
    """
    Обучает нейронную сеть для оценки риска дефолта.
    
    Args:
        df: датафрейм с данными
        save: сохранить ли модель
        model_path: путь для сохранения
    
    Returns:
        Tuple[model, grid]: обученная модель и результаты поиска параметров
    """
    param_grid = {
        "classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__learning_rate_init": [0.001, 0.01],
        "classifier__hidden_layer_sizes": [(64, 32), (128, 64)]
    }
    return train_model(df, 'nn', param_grid, save, model_path)

def load_model(model_type: str = 'logreg') -> ImbPipeline:
    """
    Загружает сохраненную модель.
    
    Args:
        model_type: тип модели ('logreg' или 'nn')
    
    Returns:
        Загруженная модель
    
    Raises:
        FileNotFoundError: если модель не найдена
        ValueError: если указан неверный тип модели
    """
    try:
        if model_type == 'logreg':
            return joblib.load(DEFAULT_MODEL_PATH)
        elif model_type == 'nn':
            return joblib.load(DEFAULT_NN_MODEL_PATH)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Модель типа {model_type} не найдена. Сначала обучите её.")

def enrich_predictions(
    df: pd.DataFrame,
    model: Optional[ImbPipeline] = None,
    model_type: str = 'logreg'
) -> pd.DataFrame:
    """
    Обогащает датафрейм предсказаниями риска дефолта.
    
    Args:
        df: исходный датафрейм
        model: предварительно загруженная модель (опционально)
        model_type: тип модели для загрузки, если model=None
    
    Returns:
        DataFrame с добавленными колонками:
        - Default_Prob: вероятность дефолта
        - Default_Pred: бинарный прогноз
        - RiskLevel: уровень риска
        - PredictionDate: дата прогноза
    """
    try:
        if model is None:
            model = load_model(model_type)
        
        X, _ = prepare_data(df)
        
        # Получение предсказаний
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        
        # Обогащение датафрейма
        df_out = df.copy()
        df_out["Default_Prob"] = probs
        df_out["Default_Pred"] = preds
        
        # Добавление уровня риска
        df_out["RiskLevel"] = pd.cut(
            probs,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
            labels=["Очень низкий", "Низкий", "Средний", "Высокий", "Очень высокий"]
        )
        
        # Добавление даты прогноза
        df_out["PredictionDate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return df_out
        
    except Exception as e:
        logger.error(f"Ошибка при обогащении данных: {str(e)}")
        raise
