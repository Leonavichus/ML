import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Callable, List, Tuple
from pandas.errors import EmptyDataError, ParserError
from streamlit.runtime.uploaded_file_manager import UploadedFile
from src.model_transactions import (
    train_iforest, train_ocsvm, train_lof,
    enrich
)
from logging import getLogger

logger = getLogger(__name__)

# Конфигурация моделей
MODEL_OPTIONS: Dict[str, Callable] = {
    "Isolation Forest": train_iforest,
    "One-Class SVM": train_ocsvm,
    "Local Outlier Factor": train_lof
}

# Обязательные поля для анализа транзакций
REQUIRED_FIELDS = [
    ("Timestamp",        "Дата и время транзакции"),
    ("Hour of Day",      "Час совершения транзакции"),
    ("Amount",           "Сумма транзакции"),
    ("Transaction Type", "Тип транзакции"),
    ("Location Region",  "Регион местоположения"),
    ("IP Prefix",        "Префикс IP-адреса"),
    ("Login Frequency",  "Частота входов в систему"),
    ("Session Duration", "Длительность сессии"),
    ("Purchase Pattern", "Шаблон покупок"),
    ("Age Group",        "Возрастная группа"),
    ("Risk Score",       "Оценка риска")
]

def create_visualizations(df: pd.DataFrame) -> List[alt.Chart]:
    """
    Создание визуализаций для аномалий в транзакциях.
    
    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с предсказанными аномалиями
        
    Возвращает:
        List[alt.Chart]: Список графиков визуализации
    """
    # Гистограмма скоринга аномалий
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("AnomalyScore:Q", bin=alt.Bin(maxbins=50), title="Anomaly Score"),
        y=alt.Y("count()", title="Количество"),
        color=alt.Color(
            "AnomalyPred:N",
            scale=alt.Scale(domain=[1,-1], range=["#4caf50","#e91e63"]),
            legend=alt.Legend(title="Нормальные / Аномальные")
        )
    ).properties(width=600, height=300)

    # Круговая диаграмма распределения аномалий
    pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta="count()",
        color=alt.Color(
            "AnomalyPred:N",
            scale=alt.Scale(domain=[1,-1], range=["#4caf50","#e91e63"])
        ),
        tooltip=["AnomalyPred:N","count()"]
    ).properties(width=300, height=300)

    # График аномалий по часам
    df_hours = df.groupby(df["Timestamp"].dt.hour)["AnomalyPred"]\
                 .apply(lambda s: (s == -1).sum())\
                 .reset_index(name="Anomalies")
    line = alt.Chart(df_hours).mark_line(point=True).encode(
        x=alt.X("Timestamp:O", title="Час дня"),
        y=alt.Y("Anomalies:Q", title="Количество аномалий")
    ).properties(width=600, height=300)

    # Точечная диаграмма Сумма vs Скоринг
    sample = df.sample(min(1000, len(df)), random_state=42)
    scatter = alt.Chart(sample).mark_circle().encode(
        x="Amount:Q",
        y="AnomalyScore:Q",
        color=alt.Color("AnomalyPred:N", legend=None),
        tooltip=["Amount","AnomalyScore","AnomalyPred"]
    ).properties(width=600, height=300)

    return [hist, pie, line, scatter]

def read_user_data(uploaded_file: UploadedFile) -> pd.DataFrame:
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
    """Основной интерфейс приложения обнаружения аномалий."""
    st.title("Обнаружение аномалий в транзакциях")

    # Выбор модели
    st.markdown("#### Выберите модель для обучения и обработки")
    method = st.selectbox("Метод обнаружения аномалий", list(MODEL_OPTIONS.keys()))

    # Обучение модели
    if st.button(f"Обучить {method}"):
        try:
            df_raw = pd.read_csv("data/raw/metaverse_transactions_dataset.csv")
            with st.spinner(f"Обучаем {method}..."):
                MODEL_OPTIONS[method](df_raw, save=True)
            st.success(f"{method} обучен!")
        except Exception as e:
            logger.exception("Ошибка при обучении модели")
            st.error(f"Ошибка при обучении: {str(e)}")

    st.markdown("---")

    # Загрузка и обработка данных
    uploaded = st.file_uploader("Загрузить CSV/Excel с транзакциями", type=["csv","xlsx"])
    if not uploaded:
        return

    try:
        # Чтение и обработка данных
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
                key=f"map_trans_{field}"
            )

        # Валидация
        missing = [label for field, label in REQUIRED_FIELDS if mapping[field] == "<отсутствует>"]
        if missing:
            st.error("Укажите соответствие для полей: " + ", ".join(missing))
            return

        # Обработка данных
        df_mapped = df_user.rename(columns={mapping[field]: field for field, _ in REQUIRED_FIELDS})
        st.write("**Пример после маппинга:**", df_mapped.head())
        
        # Обогащение данных предсказаниями
        try:
            df_enriched = enrich(df_mapped, method)
            df_enriched["timestamp"] = pd.to_datetime(df_enriched["timestamp"])
            
            st.write("**Пример обогащённых данных:**", df_enriched.head())

            # Создание и отображение визуализаций
            charts = create_visualizations(df_enriched)
            for chart in charts:
                st.altair_chart(chart)

            # Скачивание результатов
            csv = df_enriched.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать обогащённый CSV",
                csv,
                "anomaly_enriched.csv",
                mime="text/csv"
            )

        except FileNotFoundError:
            st.error("Модель не найдена. Пожалуйста, обучите модель сначала.")
        except Exception as e:
            logger.exception("Ошибка при обогащении данных")
            st.error(f"Ошибка при обогащении данных: {str(e)}")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении обнаружения аномалий")
        st.error(f"Неожиданная ошибка: {str(e)}")
