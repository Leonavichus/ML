import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError

# Пути для моделей
DEFAULT_MODEL_PATH = "models/default_risk/logreg_default.pkl"
DEFAULT_NN_MODEL_PATH = "models/default_risk/nn_default.pkl"

# === Препроцессор ===

def build_default_preprocessor():
    num_cols = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"
    ]
    cat_cols = [
        "person_home_ownership", "loan_intent", "loan_grade",
        "cb_person_default_on_file"
    ]
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler()
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )

# === Логистическая регрессия ===

def train_default_model(df: pd.DataFrame, save=True, model_path=DEFAULT_MODEL_PATH):
    if "loan_status" not in df.columns:
        raise ValueError("Отсутствует целевая переменная 'loan_status'")
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    preprocessor = build_default_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model)
    ])

    param_grid = {
        "classifier__C": [0.1, 1, 10]
    }
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=StratifiedKFold(5), n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_

    if save:
        joblib.dump(best_model, model_path)
    return best_model, grid

def load_default_model(model_path=DEFAULT_MODEL_PATH):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Модель не найдена по пути {model_path}. Сначала обучите её.")

def enrich_default(df: pd.DataFrame, model=None):
    if model is None:
        model = load_default_model()
    X = df.copy()
    try:
        prob = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
    except NotFittedError:
        raise NotFittedError("Модель не обучена.")
    df_out = df.copy()
    df_out["Default_Prob"] = prob
    df_out["Default_Pred"] = pred
    return df_out

# === Нейросеть ===

def train_default_nn_model(df: pd.DataFrame, save=True, model_path=DEFAULT_NN_MODEL_PATH):
    if "loan_status" not in df.columns:
        raise ValueError("Отсутствует целевая переменная 'loan_status'")
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    preprocessor = build_default_preprocessor()
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model)
    ])

    param_grid = {
        "classifier__alpha": [0.0001, 0.001],
        "classifier__learning_rate_init": [0.001, 0.01]
    }
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=StratifiedKFold(5), n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_

    if save:
        joblib.dump(best_model, model_path)
    return best_model, grid

def load_default_nn_model(model_path=DEFAULT_NN_MODEL_PATH):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Нейросетевая модель не найдена по пути {model_path}. Сначала обучите её.")

def enrich_default_nn(df: pd.DataFrame, model=None):
    if model is None:
        model = load_default_nn_model()
    X = df.copy()
    try:
        prob = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
    except NotFittedError:
        raise NotFittedError("Модель не обучена.")
    df_out = df.copy()
    df_out["Default_Prob"] = prob
    df_out["Default_Pred"] = pred
    return df_out
