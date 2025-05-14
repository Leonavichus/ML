import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Tuple, Callable, Any
from pandas.errors import EmptyDataError, ParserError
from sklearn.exceptions import NotFittedError
from src.model_churn import (
    train_churn_rf, load_churn_rf, enrich_churn_rf,
    train_churn_nn, load_churn_nn, enrich_churn_nn,
    train_churn_xgb, load_churn_xgb, enrich_churn_xgb,
    train_churn_lgbm, load_churn_lgbm, enrich_churn_lgbm
)
from logging import getLogger

logger = getLogger(__name__)

# Конфигурация моделей
MODEL_OPTIONS: Dict[str, Tuple[str, Callable, Callable, Callable]] = {
    'Random Forest': ('rf', train_churn_rf, load_churn_rf, enrich_churn_rf),
    'Neural Network': ('nn', train_churn_nn, load_churn_nn, enrich_churn_nn),
    'XGBoost': ('xgb', train_churn_xgb, load_churn_xgb, enrich_churn_xgb),
    'LightGBM': ('lgbm', train_churn_lgbm, load_churn_lgbm, enrich_churn_lgbm),
}

# Обязательные поля для ввода в модель
REQUIRED_FIELDS: List[Tuple[str, str]] = [
    ("CreditScore", "Кредитный скоринг"),
    ("Geography", "Местоположение"),
    ("Gender", "Пол"),
    ("Age", "Возраст"),
    ("Tenure", "Стаж обслуживания"),
    ("Balance", "Баланс"),
    ("NumOfProducts", "Число продуктов"),
    ("HasCrCard", "Есть кредитная карта"),
    ("IsActiveMember", "Активный клиент"),
    ("EstimatedSalary", "Оценочная зарплата"),
]

def create_visualization(df: pd.DataFrame, pred_col: str, prob_col: str) -> Tuple[alt.Chart, alt.Chart]:
    """
    Создание визуализаций для прогнозов оттока.
    
    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с прогнозами
        pred_col (str): Название колонки с прогнозами
        prob_col (str): Название колонки с вероятностями
        
    Возвращает:
        Tuple[alt.Chart, alt.Chart]: Графики распределения прогнозов и вероятностей
    """
    pred_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{pred_col}:O", title="Прогноз оттока"),
        y=alt.Y("count()", title="Количество"),
        color=alt.Color(f"{pred_col}:O", legend=None)
    ).properties(width=600, height=300)

    prob_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{prob_col}", bin=alt.Bin(maxbins=30), title="Вероятность оттока"),
        y=alt.Y("count()", title="Количество")
    ).properties(width=600, height=300)

    return pred_chart, prob_chart

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
    """Основной интерфейс приложения прогнозирования оттока."""
    st.title("Отток клиентов")

    # Выбор модели
    st.markdown("#### Выберите модель для обучения и обработки")
    model_name = st.selectbox(
        "Модель:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )
    key_prefix, train_fn, load_fn, enrich_fn = MODEL_OPTIONS[model_name]

    # Обучение модели
    if st.button(f"Обучить {model_name} на встроенных данных"):
        try:
            df_raw = pd.read_csv("data/raw/churn.csv")
            with st.spinner(f"Обучаем модель {model_name}..."):
                model, grid = train_fn(df_raw, save=True)
            st.success(f"{model_name} обучена (ROC-AUC: {grid.best_score_:.3f})")
        except Exception as e:
            logger.exception("Ошибка при обучении модели")
            st.error(f"Ошибка при обучении: {str(e)}")

    st.markdown("---")

    # Загрузка и обработка данных
    uploaded = st.file_uploader("Загрузить CSV/Excel для обогащения", type=["csv", "xlsx"])
    if not uploaded:
        return

    try:
        df_user = read_user_data(uploaded)
        st.write("**Исходные колонки:**", df_user.columns.tolist())

        # Сопоставление полей
        st.markdown("#### Настройка соответствия полей")
        mapping = {}
        cols = df_user.columns.tolist()
        for field, label in REQUIRED_FIELDS:
            mapping[field] = st.selectbox(
                f"{label} →", 
                options=["<отсутствует>"] + cols,
                key=f"map_{key_prefix}_{field}"
            )

        # Валидация
        missing = [label for field, label in REQUIRED_FIELDS if mapping[field] == "<отсутствует>"]
        if missing:
            st.error("Укажите соответствие для полей: " + ", ".join(missing))
            return

        # Обработка данных
        df_mapped = df_user.rename(columns={mapping[field]: field for field, _ in REQUIRED_FIELDS})
        st.write("**Пример после маппинга:**", df_mapped.head())

        # Прогнозирование
        try:
            model = load_fn()
            df_enriched = enrich_fn(df_mapped, model)
        except FileNotFoundError:
            st.error("Модель не найдена. Пожалуйста, обучите модель сначала.")
            return
        except NotFittedError:
            st.error("Модель не обучена. Пожалуйста, обучите модель сначала.")
            return
        except Exception as e:
            logger.exception("Ошибка при прогнозировании")
            st.error(f"Ошибка при обогащении: {str(e)}")
            return

        st.write("**Обогащённый датасет:**", df_enriched.head())

        # Визуализация
        pred_chart, prob_chart = create_visualization(
            df_enriched, 
            pred_col="Exited_Pred",
            prob_col="Exited_Prob"
        )
        st.altair_chart(pred_chart)
        st.altair_chart(prob_chart)

        # Скачивание результатов
        csv_data = df_enriched.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать обогащённый CSV",
            data=csv_data,
            file_name=f"churn_enriched_{key_prefix}.csv",
            mime="text/csv"
        )

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении прогнозирования оттока")
        st.error(f"Неожиданная ошибка: {str(e)}")
