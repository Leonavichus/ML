import streamlit as st
import pandas as pd
import altair as alt
from pandas.errors import EmptyDataError, ParserError
from sklearn.exceptions import NotFittedError
from src.model_default_risk import (
    train_default_model, load_default_model, enrich_default,
    train_default_nn_model, load_default_nn_model, enrich_default_nn
)

# Опции моделей для риска дефолта
MODEL_OPTIONS = {
    'Logistic Regression': ('logreg', train_default_model, load_default_model, enrich_default),
    'Neural Network': ('nn', train_default_nn_model, load_default_nn_model, enrich_default_nn)
}

# Обязательные поля для маппинга входных признаков
REQUIRED_FIELDS = [
    ('person_age', 'Возраст клиента'),
    ('person_income', 'Доход клиента'),
    ('person_home_ownership', 'Тип собственности'),
    ('person_emp_length', 'Стаж работы (лет)'),
    ('loan_intent', 'Цель кредита'),
    ('loan_grade', 'Кредитный рейтинг'),
    ('loan_amnt', 'Сумма кредита'),
    ('loan_int_rate', 'Процентная ставка (%)'),
    ('loan_percent_income', 'Отношение кредита к доходу'),
    ('cb_person_default_on_file', 'Дефолт в истории (Y/N)'),
    ('cb_person_cred_hist_length', 'Длина кредитной истории (лет)'),
]


def app():
    st.title("Риск дефолта кредита")

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
            df_raw = pd.read_csv("data/raw/default_risk.csv")
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
            f"{label} →",
            options=["<отсутствует>"] + cols,
            key=f"map_{key_prefix}_{field}"
        )
    missing = [label for field, label in REQUIRED_FIELDS if mapping[field] == "<отсутствует>"]
    if missing:
        st.error("Укажите соответствие для полей: " + ", ".join(missing))
        return

    # Переименование полей
    df_mapped = df_user.rename(columns={mapping[field]: field for field, _ in REQUIRED_FIELDS})
    st.write("**Пример после маппинга:**", df_mapped.head())

    # Обогащение данных
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

    # Визуализация прогнозов
    pred_col = "Default_Pred"
    prob_col = "Default_Prob"
    chart1 = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X(f"{pred_col}:O", title="Прогноз дефолта"),
        y=alt.Y("count()", title="Количество"),
        color=alt.Color(f"{pred_col}:O", legend=None)
    ).properties(width=600, height=300)
    st.altair_chart(chart1)

    chart2 = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X(f"{prob_col}", bin=alt.Bin(maxbins=30), title="Вероятность дефолта"),
        y=alt.Y("count()", title="Количество")
    ).properties(width=600, height=300)
    st.altair_chart(chart2)

    # Скачивание обогащённого CSV
    csv_data = df_enriched.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать обогащённый CSV",
        data=csv_data,
        file_name=f"default_risk_enriched_{key_prefix}.csv",
        mime="text/csv"
    )
