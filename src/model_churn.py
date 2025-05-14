"""
Модуль прогнозирования оттока клиентов с использованием различных методов машинного обучения.

Реализованы следующие модели:
1. Random Forest (случайный лес)
2. Neural Network (нейронная сеть)
3. XGBoost (градиентный бустинг)
4. LightGBM (легковесный градиентный бустинг)

Особенности реализации:
- Автоматическая предобработка числовых и категориальных признаков
- Балансировка классов с помощью SMOTE
- Подбор гиперпараметров через GridSearchCV
- Кросс-валидация для оценки качества
"""

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.exceptions import NotFittedError
from typing import Tuple, Dict, Any, Optional
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Пути к сохраненным моделям
CHURN_RF_MODEL_PATH = "models/churn/churn_rf.pkl"
CHURN_NN_MODEL_PATH = "models/churn/churn_nn.pkl"
CHURN_XGB_MODEL_PATH = "models/churn/churn_xgb.pkl"
CHURN_LGBM_MODEL_PATH = "models/churn/churn_lgbm.pkl"

def build_churn_preprocessor() -> ColumnTransformer:
    """
    Создает пайплайн предобработки данных для моделей оттока.
    
    Обрабатывает:
    Числовые признаки:
    - CreditScore: кредитный рейтинг
    - Age: возраст
    - Tenure: срок обслуживания
    - Balance: баланс
    - NumOfProducts: количество продуктов
    - EstimatedSalary: предполагаемая зарплата
    
    Категориальные признаки:
    - Geography: регион
    - Gender: пол
    - HasCrCard: наличие кредитной карты
    - IsActiveMember: активность клиента
    
    Returns:
        ColumnTransformer: сконфигурированный трансформер данных
    """
    # Определение признаков по типам
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
    
    # Пайплайны обработки
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
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

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Подготавливает данные для обучения или предсказания.
    
    Args:
        df: исходный датафрейм
    
    Returns:
        Tuple[X, y]: признаки и целевая переменная (если есть)
    
    Raises:
        ValueError: если для обучения нет целевой переменной
    """
    X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"], errors="ignore")
    y = df.get("Exited")
    return X, y

def train_model(
    df: pd.DataFrame,
    model_type: str,
    param_grid: Dict[str, Any],
    save: bool = True,
    model_path: str = None
) -> Tuple[ImbPipeline, GridSearchCV]:
    """
    Общая функция для обучения моделей оттока.
    
    Args:
        df: датафрейм с данными
        model_type: тип модели ('rf', 'nn', 'xgb', 'lgbm')
        param_grid: сетка параметров для GridSearchCV
        save: сохранить ли модель
        model_path: путь для сохранения
    
    Returns:
        Tuple[model, grid]: обученная модель и результаты поиска параметров
    """
    logger.info(f"Начало обучения модели {model_type.upper()}")
    
    X, y = prepare_data(df)
    if y is None:
        raise ValueError("Целевая переменная 'Exited' не найдена в данных")
    
    # Выбор базовой модели
    if model_type == 'rf':
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_type == 'nn':
        classifier = MLPClassifier(max_iter=500, random_state=42)
    elif model_type == 'xgb':
        classifier = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    elif model_type == 'lgbm':
        classifier = LGBMClassifier(random_state=42)
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
    
    # Создание пайплайна
    pipeline = ImbPipeline([
        ("preprocessor", build_churn_preprocessor()),
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

def train_churn_rf(df: pd.DataFrame, save: bool = True) -> Tuple[ImbPipeline, GridSearchCV]:
    """
    Обучает модель Random Forest для прогнозирования оттока.
    
    Args:
        df: датафрейм с данными
        save: сохранить ли модель
    
    Returns:
        Tuple[model, grid]: обученная модель и результаты поиска параметров
    """
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }
    return train_model(df, 'rf', param_grid, save, CHURN_RF_MODEL_PATH)

def load_churn_rf(model_path=CHURN_RF_MODEL_PATH):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError("RF-модель оттока не найдена.")

def enrich_churn_rf(df: pd.DataFrame, model=None):
    if model is None:
        model = load_churn_rf()
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    try:
        probs = model.predict_proba(X)[:,1]
        preds = model.predict(X)
    except NotFittedError:
        raise NotFittedError("RF-модель не обучена.")
    out = df.copy()
    out["Exited_Prob"] = probs; out["Exited_Pred"] = preds
    return out

# === Neural Network ===
def train_churn_nn(df: pd.DataFrame, save=True):
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    y = df.get("Exited")
    if y is None:
        raise ValueError("Целевая 'Exited' не найдена.")
    pipeline = ImbPipeline([
        ("preprocessor", build_churn_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", MLPClassifier(max_iter=500,random_state=42))
    ])
    param_grid = {"classifier__hidden_layer_sizes": [(50,),(100,50)], "classifier__alpha":[1e-4,1e-3]}
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=StratifiedKFold(5,shuffle=True,random_state=42), n_jobs=-1)
    grid.fit(X,y)
    if save:
        joblib.dump(grid.best_estimator_, CHURN_NN_MODEL_PATH)
    return grid.best_estimator_, grid

def load_churn_nn(path=CHURN_NN_MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("NN-модель оттока не найдена.")

def enrich_churn_nn(df: pd.DataFrame, model=None):
    if model is None:
        model = load_churn_nn()
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    try:
        probs = model.predict_proba(X)[:,1]
        preds = model.predict(X)
    except NotFittedError:
        raise NotFittedError("NN-модель не обучена.")
    out = df.copy()
    out["Exited_Prob"] = probs; out["Exited_Pred"] = preds
    return out

# === XGBoost ===
def train_churn_xgb(df: pd.DataFrame, save=True):
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    y = df.get("Exited")
    if y is None:
        raise ValueError("Целевая 'Exited' не найдена.")
    pipeline = ImbPipeline([
        ("preprocessor", build_churn_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=42))
    ])
    param_grid = {"classifier__n_estimators":[100,200], "classifier__max_depth":[3,5]}
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=StratifiedKFold(5,shuffle=True,random_state=42), n_jobs=-1)
    grid.fit(X,y)
    if save:
        joblib.dump(grid.best_estimator_, CHURN_XGB_MODEL_PATH)
    return grid.best_estimator_, grid

def load_churn_xgb(path=CHURN_XGB_MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("XGB-модель оттока не найдена.")

def enrich_churn_xgb(df: pd.DataFrame, model=None):
    if model is None:
        model = load_churn_xgb()
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    try:
        probs = model.predict_proba(X)[:,1]
        preds = model.predict(X)
    except NotFittedError:
        raise NotFittedError("XGB-модель не обучена.")
    out = df.copy()
    out["Exited_Prob"] = probs; out["Exited_Pred"] = preds
    return out

# === LightGBM ===
def train_churn_lgbm(df: pd.DataFrame, save=True):
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    y = df.get("Exited")
    if y is None:
        raise ValueError("Целевая 'Exited' не найдена.")
    pipeline = ImbPipeline([
        ("preprocessor", build_churn_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LGBMClassifier(random_state=42))
    ])
    param_grid = {"classifier__n_estimators":[100,200], "classifier__num_leaves":[31,63]}
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=StratifiedKFold(5,shuffle=True,random_state=42), n_jobs=-1)
    grid.fit(X,y)
    if save:
        joblib.dump(grid.best_estimator_, CHURN_LGBM_MODEL_PATH)
    return grid.best_estimator_, grid

def load_churn_lgbm(path=CHURN_LGBM_MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("LGBM-модель оттока не найдена.")

def enrich_churn_lgbm(df: pd.DataFrame, model=None):
    if model is None:
        model = load_churn_lgbm()
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    try:
        probs = model.predict_proba(X)[:,1]
        preds = model.predict(X)
    except NotFittedError:
        raise NotFittedError("LGBM-модель не обучена.")
    out = df.copy()
    out["Exited_Prob"] = probs; out["Exited_Pred"] = preds
    return out

def enrich_predictions(
    df: pd.DataFrame,
    model: Optional[ImbPipeline] = None,
    model_type: str = 'rf'
) -> pd.DataFrame:
    """
    Обогащает датафрейм предсказаниями модели оттока.
    
    Args:
        df: исходный датафрейм
        model: предварительно загруженная модель (опционально)
        model_type: тип модели для загрузки, если model=None
    
    Returns:
        DataFrame с добавленными колонками:
        - Exited_Prob: вероятность оттока
        - Exited_Pred: бинарный прогноз
        - ChurnRiskLevel: уровень риска оттока
        - PredictionDate: дата прогноза
    """
    try:
        if model is None:
            if model_type == 'rf':
                model = load_churn_rf()
            elif model_type == 'nn':
                model = load_churn_nn()
            elif model_type == 'xgb':
                model = load_churn_xgb()
            elif model_type == 'lgbm':
                model = load_churn_lgbm()
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        X, _ = prepare_data(df)
        
        # Получение предсказаний
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        
        # Обогащение датафрейма
        out = df.copy()
        out["Exited_Prob"] = probs
        out["Exited_Pred"] = preds
        
        # Добавление уровня риска
        out["ChurnRiskLevel"] = pd.cut(
            probs,
            bins=[0, 0.3, 0.6, 1],
            labels=["Низкий", "Средний", "Высокий"]
        )
        
        # Добавление даты прогноза
        out["PredictionDate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return out
        
    except Exception as e:
        logger.error(f"Ошибка при обогащении данных: {str(e)}")
        raise
