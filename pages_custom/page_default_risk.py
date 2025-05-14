import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Tuple, Callable, Any
from pandas.errors import EmptyDataError, ParserError
from sklearn.exceptions import NotFittedError
from src.model_default_risk import (
    train_default_model, load_default_model, enrich_default,
    train_default_nn_model, load_default_nn_model, enrich_default_nn
)
from logging import getLogger

logger = getLogger(__name__)

# Конфигурация моделей
MODEL_OPTIONS: Dict[str, Tuple[str, Callable, Callable, Callable]] = {
    'Logistic Regression': ('logreg', train_default_model, load_default_model, enrich_default),
    'Neural Network': ('nn', train_default_nn_model, load_default_nn_model, enrich_default_nn)
}

# Обязательные поля для ввода в модель
REQUIRED_FIELDS: List[Tuple[str, str]] = [
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

def create_visualizations(df: pd.DataFrame, pred_col: str, prob_col: str) -> List[alt.Chart]:
    """
    Создание визуализаций для прогнозов риска дефолта.
    
    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с прогнозами
        pred_col (str): Название колонки с прогнозами
        prob_col (str): Название колонки с вероятностями
        
    Возвращает:
        List[alt.Chart]: Список графиков визуализации
    """
    # Распределение прогнозов дефолта
    pred_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{pred_col}:O", title="Прогноз дефолта"),
        y=alt.Y("count()", title="Количество"),
        color=alt.Color(f"{pred_col}:O", legend=None)
    ).properties(width=600, height=300)

    # Распределение вероятностей дефолта
    prob_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{prob_col}", bin=alt.Bin(maxbins=30), title="Вероятность дефолта"),
        y=alt.Y("count()", title="Количество")
    ).properties(width=600, height=300)

    return [pred_chart, prob_chart]

def read_user_data(uploaded_file: Any) -> pd.DataFrame:
    """
    Чтение и валидация загруженных пользователем данных.
    
    Аргументы:
        uploaded_file: Объект загруженного файла Streamlit
        
    Возвращает:
        pd.DataFrame: Загруженный датафрейм
        
    Вызывает:
        ValueError: Если не удалось прочитать файл
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("Неподдерживаемый формат файла")
    except (EmptyDataError, ParserError) as e:
        raise ValueError(f"Ошибка при чтении файла: {str(e)}")
    except Exception as e:
        raise ValueError(f"Неизвестная ошибка при чтении файла: {str(e)}")

def app():
    """Main application interface for default risk prediction."""
    st.title("Риск дефолта кредита")

    # Model selection
    st.markdown("#### Выберите модель для обучения и обработки")
    model_name = st.selectbox(
        "Модель:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )
    key_prefix, train_fn, load_fn, enrich_fn = MODEL_OPTIONS[model_name]

    # Model training
    if st.button(f"Обучить {model_name} на встроенных данных"):
        try:
            df_raw = pd.read_csv("data/raw/default_risk.csv")
            with st.spinner(f"Обучаем модель {model_name}..."):
                model, grid = train_fn(df_raw, save=True)
            st.success(f"{model_name} обучена (ROC-AUC: {grid.best_score_:.3f})")
        except Exception as e:
            logger.exception("Error during model training")
            st.error(f"Ошибка при обучении: {str(e)}")

    st.markdown("---")

    # Data upload and processing
    uploaded = st.file_uploader("Загрузить CSV/Excel для обогащения", type=["csv", "xlsx"])
    if not uploaded:
        return

    try:
        # Read and process data
        df_user = read_user_data(uploaded)
    st.write("**Исходные колонки:**", df_user.columns.tolist())

        # Field mapping
    st.markdown("#### Настройка соответствия полей")
    mapping = {}
    cols = df_user.columns.tolist()
    for field, label in REQUIRED_FIELDS:
        mapping[field] = st.selectbox(
            f"{label} →",
            options=["<отсутствует>"] + cols,
            key=f"map_{key_prefix}_{field}"
        )

        # Validation
    missing = [label for field, label in REQUIRED_FIELDS if mapping[field] == "<отсутствует>"]
    if missing:
        st.error("Укажите соответствие для полей: " + ", ".join(missing))
        return

        # Data processing
    df_mapped = df_user.rename(columns={mapping[field]: field for field, _ in REQUIRED_FIELDS})
    st.write("**Пример после маппинга:**", df_mapped.head())

        # Model prediction
    try:
        model = load_fn()
        df_enriched = enrich_fn(df_mapped, model)
    st.write("**Обогащённый датасет:**", df_enriched.head())

            # Visualizations
            charts = create_visualizations(
                df_enriched,
                pred_col="Default_Pred",
                prob_col="Default_Prob"
            )
            for chart in charts:
                st.altair_chart(chart)

            # Download results
    csv_data = df_enriched.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать обогащённый CSV",
        data=csv_data,
        file_name=f"default_risk_enriched_{key_prefix}.csv",
        mime="text/csv"
    )

        except FileNotFoundError:
            st.error("Модель не найдена. Пожалуйста, обучите модель сначала.")
        except NotFittedError:
            st.error("Модель не обучена. Пожалуйста, обучите модель сначала.")
        except Exception as e:
            logger.exception("Error during prediction")
            st.error(f"Ошибка при обогащении: {str(e)}")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        logger.exception("Unexpected error in default risk prediction")
        st.error(f"Неожиданная ошибка: {str(e)}")
