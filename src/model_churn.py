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

# Пути к моделям
CHURN_RF_MODEL_PATH = "models/churn/churn_rf.pkl"
CHURN_NN_MODEL_PATH = "models/churn/churn_nn.pkl"
CHURN_XGB_MODEL_PATH = "models/churn/churn_xgb.pkl"
CHURN_LGBM_MODEL_PATH = "models/churn/churn_lgbm.pkl"

# === Общий препроцессор для оттока ===
def build_churn_preprocessor():
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop")

# === RandomForest ===
def train_churn_rf(df: pd.DataFrame, save=True):
    X = df.drop(columns=["RowNumber","CustomerId","Surname","Exited"], errors="ignore")
    y = df.get("Exited")
    if y is None:
        raise ValueError("Целевая 'Exited' не найдена.")
    pipeline = ImbPipeline([
        ("preprocessor", build_churn_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    param_grid = {"classifier__n_estimators": [100,200], "classifier__max_depth": [None,10,20]}
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=StratifiedKFold(5,shuffle=True,random_state=42), n_jobs=-1)
    grid.fit(X,y)
    if save:
        joblib.dump(grid.best_estimator_, CHURN_RF_MODEL_PATH)
    return grid.best_estimator_, grid

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
