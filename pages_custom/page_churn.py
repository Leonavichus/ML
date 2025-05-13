import streamlit as st
import pandas as pd
import altair as alt
from pandas.errors import EmptyDataError, ParserError
from sklearn.exceptions import NotFittedError
from src.model_churn import (
    train_churn_rf, load_churn_rf, enrich_churn_rf,
    train_churn_nn, load_churn_nn, enrich_churn_nn,
    train_churn_xgb, load_churn_xgb, enrich_churn_xgb,
    train_churn_lgbm, load_churn_lgbm, enrich_churn_lgbm
)

# Возможные модели
MODEL_OPTIONS = {
    'Random Forest': ('rf', train_churn_rf, load_churn_rf, enrich_churn_rf),
    'Neural Network': ('nn', train_churn_nn, load_churn_nn, enrich_churn_nn),
    'XGBoost': ('xgb', train_churn_xgb, load_churn_xgb, enrich_churn_xgb),
    'LightGBM': ('lgbm', train_churn_lgbm, load_churn_lgbm, enrich_churn_lgbm),
}

REQUIRED_FIELDS = [
    ("CreditScore", "Кредитный скоринг"),
    ("Geography", "География"),
    ("Gender", "Пол"),
    ("Age", "Возраст"),
    ("Tenure", "Стаж обслуживания"),
    ("Balance", "Баланс"),
    ("NumOfProducts", "Число продуктов"),
    ("HasCrCard", "Есть кредитная карта"),
    ("IsActiveMember", "Активный клиент"),
    ("EstimatedSalary", "Оценочная зарплата"),
]


def app():
    st.title("Отток клиентов")

    # Выбор модели на странице (не в сайдбаре)
    st.markdown("#### Выберите модель для обучения и обработки")
    model_name = st.selectbox(
        "Модель:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )
    key_prefix, train_fn, load_fn, enrich_fn = MODEL_OPTIONS[model_name]

    # Кнопка обучения на странице
    if st.button(f"Обучить {model_name} на встроенных данных"):
        try:
            df_raw = pd.read_csv("data/raw/churn.csv")
            with st.spinner(f"Обучаем модель {model_name}..."):
                model, grid = train_fn(df_raw, save=True)
            st.success(f"{model_name} обучена (ROC-AUC: {grid.best_score_:.3f})")
        except Exception as e:
            st.error(f"Ошибка при обучении: {e}")

    st.markdown("---")

    uploaded = st.file_uploader("Загрузить CSV/Excel для обогащения", type=["csv", "xlsx"])
    if not uploaded:
        return

    # Чтение пользовательского файла
    try:
        df_user = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    except (EmptyDataError, ParserError) as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return
    except Exception as e:
        st.error(f"Неизвестная ошибка при чтении файла: {e}")
        return

    st.write("**Исходные колонки:**", df_user.columns.tolist())

    # Маппинг колонок
    st.markdown("#### Настройка соответствия полей")
    mapping = {}
    cols = df_user.columns.tolist()
    for field, label in REQUIRED_FIELDS:
        mapping[field] = st.selectbox(
            f"{label} →", options=["<отсутствует>"] + cols,
            key=f"map_{key_prefix}_{field}"
        )
    missing = [label for field, label in REQUIRED_FIELDS if mapping[field] == "<отсутствует>"]
    if missing:
        st.error("Укажите соответствие для полей: " + ", ".join(missing))
        return

    # Переименование
    df_mapped = df_user.rename(columns={mapping[field]: field for field, _ in REQUIRED_FIELDS})
    st.write("**Пример после маппинга:**", df_mapped.head())

    # Обогащение данных выбранной моделью
    try:
        model = load_fn()
        df_enriched = enrich_fn(df_mapped, model)
    except FileNotFoundError as e:
        st.error(e)
        return
    except NotFittedError as e:
        st.error(e)
        return
    except Exception as e:
        st.error(f"Ошибка при обогащении: {e}")
        return

    st.write("**Обогащённый датасет:**", df_enriched.head())

    # Визуализация
    pred_col = "Exited_Pred"
    prob_col = "Exited_Prob"
    chart1 = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X(f"{pred_col}:O", title="Прогноз оттока"),
        y=alt.Y("count()", title="Количество"),
        color=alt.Color(f"{pred_col}:O", legend=None)
    ).properties(width=600, height=300)
    st.altair_chart(chart1)

    chart2 = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X(f"{prob_col}", bin=alt.Bin(maxbins=30), title="Вероятность оттока"),
        y=alt.Y("count()", title="Количество")
    ).properties(width=600, height=300)
    st.altair_chart(chart2)

    # Скачивание
    csv_data = df_enriched.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать обогащённый CSV",
        data=csv_data,
        file_name=f"churn_enriched_{key_prefix}.csv",
        mime="text/csv"
    )
