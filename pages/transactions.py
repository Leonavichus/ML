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

st.set_page_config(
    page_title="Транзакции | Analytics App",
    page_icon="💱",
    layout="wide"
)

# Словарь доступных моделей для обнаружения аномалий
MODEL_OPTIONS: Dict[str, Callable] = {
    "Isolation Forest": train_iforest,
    "One-Class SVM": train_ocsvm,
    "Local Outlier Factor": train_lof
}

# Обязательные поля для анализа транзакций с описаниями
REQUIRED_FIELDS = [
    ("timestamp", "Дата и время транзакции"),
    ("hour_of_day", "Час совершения транзакции"),
    ("amount", "Сумма транзакции"),
    ("transaction_type", "Тип транзакции"),
    ("location_region", "Регион местоположения"),
    ("ip_prefix", "Префикс IP-адреса"),
    ("login_frequency", "Частота входов в систему"),
    ("session_duration", "Длительность сессии"),
    ("purchase_pattern", "Шаблон покупок"),
    ("age_group", "Возрастная группа"),
]


def create_visualizations(df: pd.DataFrame) -> List[alt.Chart]:
    """
    Создание расширенных визуализаций для анализа аномалий в транзакциях.

    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с предсказанными аномалиями

    Возвращает:
        List[alt.Chart]: Список интерактивных графиков визуализации
    """
    # Гистограмма распределения скоринга аномалий
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("AnomalyScore:Q", bin=alt.Bin(maxbins=50), title="Оценка аномальности"),
        y=alt.Y("count()", title="Количество транзакций"),
        color=alt.Color(
            "AnomalyPred:N",
            scale=alt.Scale(domain=[1, -1], range=["#4caf50", "#e91e63"]),
            legend=alt.Legend(title="Тип транзакции")
        ),
        tooltip=[
            alt.Tooltip("AnomalyScore:Q", title="Оценка"),
            alt.Tooltip("count()", title="Количество")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение оценок аномальности"
    )

    # Круговая диаграмма соотношения нормальных и аномальных транзакций
    pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta="count()",
        color=alt.Color(
            "AnomalyPred:N",
            scale=alt.Scale(domain=[1, -1], range=["#4caf50", "#e91e63"])
        ),
        tooltip=[
            alt.Tooltip("AnomalyPred:N", title="Тип"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("count():Q", title="Процент", format=".1%")
        ]
    ).properties(
        width=300,
        height=300,
        title="Соотношение нормальных и аномальных транзакций"
    )

    # График аномалий по часам
    df_hours = df.groupby([
        df["timestamp"].dt.hour,
        "AnomalyPred"
    ]).size().reset_index(name="count")

    line = alt.Chart(df_hours).mark_line(point=True).encode(
        x=alt.X("timestamp:O", title="Час дня"),
        y=alt.Y("count:Q", title="Количество транзакций"),
        color=alt.Color(
            "AnomalyPred:N",
            scale=alt.Scale(domain=[1, -1], range=["#4caf50", "#e91e63"])
        ),
        tooltip=[
            alt.Tooltip("timestamp:O", title="Час"),
            alt.Tooltip("count:Q", title="Количество"),
            alt.Tooltip("AnomalyPred:N", title="Тип")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение транзакций по часам"
    )

    # Тепловая карта аномалий по дням недели и часам
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour

    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X("hour:O", title="Час"),
        y=alt.Y("day_of_week:O", title="День недели",
                sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        color=alt.Color("count(AnomalyPred)", title="Количество аномалий"),
        tooltip=[
            alt.Tooltip("day_of_week:O", title="День"),
            alt.Tooltip("hour:O", title="Час"),
            alt.Tooltip("count(AnomalyPred):Q", title="Аномалии")
        ]
    ).transform_filter(
        alt.datum.AnomalyPred == -1
    ).properties(
        width=600,
        height=300,
        title="Тепловая карта аномалий"
    )

    # Точечная диаграмма Сумма vs Скоринг с интерактивностью
    sample = df.sample(min(1000, len(df)), random_state=42)
    scatter = alt.Chart(sample).mark_circle().encode(
        x=alt.X("amount:Q", title="Сумма транзакции"),
        y=alt.Y("AnomalyScore:Q", title="Оценка аномальности"),
        color=alt.Color(
            "AnomalyPred:N",
            scale=alt.Scale(domain=[1, -1], range=["#4caf50", "#e91e63"])
        ),
        size=alt.Size("risk_score:Q", title="Оценка риска"),
        tooltip=[
            alt.Tooltip("amount:Q", title="Сумма"),
            alt.Tooltip("AnomalyScore:Q", title="Оценка аномальности"),
            alt.Tooltip("risk_score:Q", title="Оценка риска"),
            alt.Tooltip("transaction_type:N", title="Тип транзакции"),
            alt.Tooltip("location_region:N", title="Регион")
        ]
    ).properties(
        width=600,
        height=300,
        title="Зависимость аномальности от суммы транзакции"
    )

    # Заменяем боксплот на более информативную визуализацию распределения сумм по типам транзакций
    amount_distribution = alt.Chart(df).transform_density(
        'amount',
        as_=['amount', 'density'],
        groupby=['transaction_type', 'AnomalyPred'],
        steps=100
    ).mark_area(
        opacity=0.5
    ).encode(
        x=alt.X('amount:Q', title='Сумма транзакции'),
        y=alt.Y('density:Q', title='Плотность распределения', stack=None),
        color=alt.Color(
            'AnomalyPred:N',
            scale=alt.Scale(domain=[1, -1], range=['#4caf50', '#e91e63']),
            title='Тип транзакции'
        ),
        row=alt.Row('transaction_type:N', title='Тип транзакции'),
        tooltip=[
            alt.Tooltip('amount:Q', title='Сумма', format=',.2f'),
            alt.Tooltip('density:Q', title='Плотность'),
            alt.Tooltip('transaction_type:N', title='Тип транзакции'),
            alt.Tooltip('AnomalyPred:N', title='Статус аномалии')
        ]
    ).properties(
        height=100,
        title={
            "text": "Распределение сумм по типам транзакций",
            "fontSize": 16
        }
    ).configure_facet(
        spacing=10
    ).configure_view(
        stroke=None
    )

    return [hist, pie, line, heatmap, scatter, amount_distribution]


def read_user_data(uploaded_file: UploadedFile) -> pd.DataFrame:
    """
    Чтение и валидация загруженных пользователем данных.

    Аргументы:
        uploaded_file: Объект загруженного файла Streamlit

    Возвращает:
        pd.DataFrame: Загруженный и валидированный датафрейм

    Вызывает:
        ValueError: Если формат файла не поддерживается или возникла ошибка при чтении
    """
    try:
        # Определяем формат файла и читаем данные
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Поддерживаются только файлы форматов CSV и Excel (.xlsx)")

        # Проверяем, что датафрейм не пустой
        if df.empty:
            raise ValueError("Загруженный файл не содержит данных")

        # Проверяем базовые требования к данным
        if len(df.columns) < 5:  # Минимальное количество колонок
            raise ValueError("Файл должен содержать не менее 5 колонок с данными")

        return df

    except EmptyDataError:
        raise ValueError("Файл пуст или имеет неверный формат")
    except ParserError:
        raise ValueError("Ошибка при разборе файла. Проверьте корректность формата")
    except Exception as e:
        logger.exception("Неожиданная ошибка при чтении файла")
        raise ValueError(f"Ошибка при чтении файла: {str(e)}")


def app():
    """
    Основной интерфейс приложения обнаружения аномалий в транзакциях.

    Функционал:
    - Выбор и обучение модели обнаружения аномалий
    - Загрузка и анализ пользовательских данных
    - Визуализация результатов анализа
    - Выгрузка обогащенных данных
    """
    st.title("Обнаружение аномалий в транзакциях")

    # Добавляем описание приложения
    st.markdown("""
    Это приложение помогает обнаруживать аномальные транзакции в ваших данных.
    Вы можете:
    - Выбрать и обучить модель обнаружения аномалий
    - Загрузить свои данные для анализа
    - Получить визуальный анализ аномалий
    - Выгрузить результаты анализа
    """)

    # Секция выбора и обучения модели
    st.markdown("### 1. Выбор модели")
    st.markdown("Выберите алгоритм для обнаружения аномалий в данных:")

    method = st.selectbox(
        "Метод обнаружения аномалий",
        list(MODEL_OPTIONS.keys()),
        help="Каждый метод имеет свои особенности в обнаружении аномалий"
    )

    # Обучение модели
    if st.button(f"Обучить модель {method}", help="Нажмите для обучения модели на тестовых данных"):
        try:
            with st.spinner(f"Обучаем модель {method}..."):
                df_raw = pd.read_csv("data/raw/metaverse_transactions_dataset.csv")
                MODEL_OPTIONS[method](df_raw, save=True)
            st.success(f"Модель {method} успешно обучена! 🎉")
        except FileNotFoundError:
            st.error("❌ Не найден файл с тестовыми данными")
        except Exception as e:
            logger.exception("Ошибка при обучении модели")
            st.error(f"❌ Ошибка при обучении модели: {str(e)}")

    st.markdown("---")

    # Секция загрузки и обработки данных
    st.markdown("### 2. Анализ данных")
    st.markdown("Загрузите файл с транзакциями для анализа:")

    uploaded = st.file_uploader(
        "Загрузить CSV/Excel с транзакциями",
        type=["csv", "xlsx"],
        help="Поддерживаются форматы CSV и Excel (.xlsx)"
    )

    if not uploaded:
        st.info("👆 Пожалуйста, загрузите файл с данными для анализа")
        return

    try:
        # Чтение и обработка данных
        df_user = read_user_data(uploaded)

        st.markdown("### 3. Сопоставление полей")
        st.markdown("Укажите соответствие между полями в вашем файле и требуемыми полями:")

        cols = df_user.columns.tolist()
        mapping = {}

        # Создаем две колонки для более компактного отображения маппинга
        col1, col2 = st.columns(2)
        for i, (field, label) in enumerate(REQUIRED_FIELDS):
            with col1 if i % 2 == 0 else col2:
                mapping[field] = st.selectbox(
                    f"{label} →",
                    options=["<отсутствует>"] + cols,
                    key=f"map_trans_{field}",
                    help=f"Выберите поле, соответствующее {label}"
                )

        # Валидация маппинга
        missing = [label for field, label in REQUIRED_FIELDS if mapping[field] == "<отсутствует>"]
        if missing:
            st.warning("⚠️ Пожалуйста, укажите соответствие для следующих полей:\n- " + "\n- ".join(missing))
            return

        # Обработка данных
        with st.spinner("Обрабатываем данные..."):
            df_mapped = df_user.rename(columns={mapping[field]: field for field, _ in REQUIRED_FIELDS})

            if st.checkbox("Показать пример данных после сопоставления"):
                st.write("Пример данных после сопоставления полей:", df_mapped.head())

            # Обогащение данных предсказаниями
            try:
                df_enriched = enrich(df_mapped, method)
                df_enriched["timestamp"] = pd.to_datetime(df_enriched["timestamp"])

                if st.checkbox("Показать пример обогащённых данных"):
                    st.write("Пример данных после обогащения:", df_enriched.head())

                # Статистика по аномалиям
                total = len(df_enriched)
                anomalies = (df_enriched["AnomalyPred"] == -1).sum()
                normal = total - anomalies

                st.markdown("### 4. Результаты анализа")

                # Показываем основную статистику
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего транзакций", total)
                with col2:
                    st.metric("Нормальные", normal)
                with col3:
                    st.metric("Аномальные", anomalies)

                # Создание и отображение визуализаций
                st.markdown("### 5. Визуализация результатов")
                charts = create_visualizations(df_enriched)
                for chart in charts:
                    st.altair_chart(chart)

                # Скачивание результатов
                st.markdown("### 6. Выгрузка результатов")
                csv = df_enriched.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Скачать результаты анализа (CSV)",
                    csv,
                    "anomaly_enriched.csv",
                    mime="text/csv",
                    help="Скачать данные с результатами анализа аномалий"
                )

            except FileNotFoundError:
                st.error("❌ Модель не найдена. Пожалуйста, сначала обучите модель.")
            except Exception as e:
                logger.exception("Ошибка при обогащении данных")
                st.error(f"❌ Ошибка при обработке данных: {str(e)}")

    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении")
        st.error(f"❌ Произошла неожиданная ошибка: {str(e)}")

app() 