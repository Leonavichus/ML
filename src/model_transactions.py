import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Пути для сохранения моделей
IFOREST_PATH = "models/anomaly/isolation_forest.pkl"
OCSVM_PATH   = "models/anomaly/ocsvm.pkl"
LOF_PATH     = "models/anomaly/lof.pkl"

# Наши поля
NUMERICAL   = ["amount", "hour_of_day", "session_duration", "login_frequency", "risk_score"]
CATEGORICAL = ["transaction_type", "location_region", "ip_prefix", "purchase_pattern", "age_group"]

def build_preprocessor():
    """ColumnTransformer: числовые → StandardScaler, категориальные → OneHotEncoder."""
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([
        ("num", num_pipe, NUMERICAL),
        ("cat", cat_pipe, CATEGORICAL),
    ], remainder="drop")

# Isolation Forest
def train_iforest(df: pd.DataFrame, save: bool = True):
    pre = build_preprocessor()
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(Xp)

    if save:
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

# Обогащение
def enrich(df: pd.DataFrame, name: str) -> pd.DataFrame:
    pre, model = load_model(name)
    X = df[NUMERICAL + CATEGORICAL]
    Xp = pre.transform(X)

    if name == "Isolation Forest":
        scores = model.decision_function(Xp)
        preds  = model.predict(Xp)
    elif name == "One-Class SVM":
        scores = model.score_samples(Xp)
        preds  = model.predict(Xp)
    else:  # Local Outlier Factor
        # LOF: отрицательный outlier_factor_, все >1 considered normal
        lof = model
        # при загрузке LOF можно переиспользовать .negative_outlier_factor_
        scores = lof.negative_outlier_factor_
        # повторим fit_predict, чтобы получить метки
        preds = lof.fit_predict(Xp)

    # нормальные → 1, аномалии → -1
    df2 = df.copy()
    df2["AnomalyScore"] = scores
    df2["AnomalyPred"]  = preds
    return df2