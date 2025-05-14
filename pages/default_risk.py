import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from pandas.errors import EmptyDataError, ParserError
from sklearn.exceptions import NotFittedError
from src.model_default_risk import (
    train_default_model, load_default_model, enrich_default,
    train_default_nn_model, load_default_nn_model, enrich_default_nn
)
from logging import getLogger

logger = getLogger(__name__)

st.set_page_config(
    page_title="Риск дефолта | Analytics App",
    page_icon="⚠️",
    layout="wide"
)

# Словарь доступных моделей для оценки риска дефолта с их описаниями
MODEL_OPTIONS: Dict[str, Tuple[str, Callable, Callable, Callable]] = {
    'Logistic Regression': ('logreg', train_default_model, load_default_model, enrich_default),
    'Neural Network': ('nn', train_default_nn_model, load_default_nn_model, enrich_default_nn)
}

# Обязательные поля для анализа риска дефолта с подробными описаниями
REQUIRED_FIELDS: List[Tuple[str, str]] = [
    ('person_age', 'Возраст клиента (полных лет)'),
    ('person_income', 'Годовой доход клиента'),
    ('person_home_ownership', 'Тип собственности (Own/Rent/Mortgage)'),
    ('person_emp_length', 'Стаж работы на текущем месте (лет)'),
    ('loan_intent', 'Цель кредита (Personal/Education/Medical/Venture/Home/Debt)'),
    ('loan_grade', 'Кредитный рейтинг (A-G)'),
    ('loan_amnt', 'Запрашиваемая сумма кредита'),
    ('loan_int_rate', 'Процентная ставка по кредиту (%)'),
    ('loan_percent_income', 'Отношение суммы кредита к годовому доходу (%)'),
    ('cb_person_default_on_file', 'Наличие дефолта в кредитной истории (Y/N)'),
    ('cb_person_cred_hist_length', 'Длительность кредитной истории (лет)'),
]


def create_visualizations(df: pd.DataFrame, pred_col: str, prob_col: str) -> List[alt.Chart]:
    """
    Создание расширенных интерактивных визуализаций для анализа риска дефолта.

    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с предсказаниями
        pred_col (str): Название колонки с прогнозами дефолта
        prob_col (str): Название колонки с вероятностями дефолта

    Возвращает:
        List[alt.Chart]: Список интерактивных графиков визуализации
    """
    # Круговая диаграмма распределения прогнозов
    pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta="count()",
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"]),
            legend=alt.Legend(title="Прогноз дефолта")
        ),
        tooltip=[
            alt.Tooltip(f"{pred_col}:N", title="Прогноз"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("count():Q", title="Доля", format=".1%")
        ]
    ).properties(
        width=300,
        height=300,
        title="Распределение прогнозов дефолта"
    )

    # Гистограмма вероятностей дефолта
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            f"{prob_col}:Q",
            bin=alt.Bin(maxbins=30),
            title="Вероятность дефолта"
        ),
        y=alt.Y(
            "count()",
            title="Количество заявок"
        ),
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"])
        ),
        tooltip=[
            alt.Tooltip(f"{prob_col}:Q", title="Вероятность", format=".2f"),
            alt.Tooltip("count()", title="Количество")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение вероятностей дефолта"
    )

    # Точечная диаграмма: доход vs сумма кредита
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("person_income:Q", title="Годовой доход"),
        y=alt.Y("loan_amnt:Q", title="Сумма кредита"),
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"])
        ),
        tooltip=[
            alt.Tooltip("person_income:Q", title="Доход", format=",.0f"),
            alt.Tooltip("loan_amnt:Q", title="Сумма кредита", format=",.0f"),
            alt.Tooltip(f"{prob_col}:Q", title="Вероятность дефолта", format=".2%"),
            alt.Tooltip("loan_intent:N", title="Цель кредита"),
            alt.Tooltip("loan_grade:N", title="Рейтинг")
        ]
    ).properties(
        width=600,
        height=400,
        title="Соотношение дохода и суммы кредита"
    )

    # Тепловая карта риска по целям кредита и рейтингу
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X("loan_grade:N", title="Кредитный рейтинг"),
        y=alt.Y("loan_intent:N", title="Цель кредита"),
        color=alt.Color(
            f"mean({prob_col}):Q",
            title="Средняя вероятность",
            scale=alt.Scale(scheme="redyellowgreen", reverse=True)
        ),
        tooltip=[
            alt.Tooltip("loan_grade:N", title="Рейтинг"),
            alt.Tooltip("loan_intent:N", title="Цель"),
            alt.Tooltip(f"mean({prob_col}):Q", title="Ср. вероятность", format=".2%"),
            alt.Tooltip("count()", title="Количество заявок")
        ]
    ).properties(
        width=500,
        height=300,
        title="Тепловая карта риска дефолта"
    )

    # Боксплот возраста по прогнозу дефолта
    boxplot = alt.Chart(df).mark_boxplot().encode(
        x=alt.X(f"{pred_col}:N", title="Прогноз дефолта"),
        y=alt.Y("person_age:Q", title="Возраст заявителя"),
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"])
        )
    ).properties(
        width=300,
        height=300,
        title="Распределение возраста по прогнозу"
    )

    return [pie, hist, scatter, heatmap, boxplot]


def read_user_data(uploaded_file: Any) -> pd.DataFrame:
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
    Основной интерфейс приложения оценки риска дефолта по кредиту.

    Функционал:
    - Выбор и обучение модели оценки риска
    - Загрузка и анализ данных заявок
    - Визуализация результатов оценки
    - Выгрузка обогащенных данных
    """
    st.title("Оценка риска дефолта по кредиту")

    # Добавляем описание приложения
    st.markdown("""
    Это приложение помогает оценить риск дефолта по кредитным заявкам.
    Вы можете:
    - Выбрать и обучить модель оценки риска
    - Загрузить данные заявок для анализа
    - Получить визуальный анализ рисков
    - Выгрузить результаты оценки
    """)

    # Секция выбора и обучения модели
    st.markdown("### 1. Выбор модели")
    st.markdown("Выберите алгоритм для оценки риска дефолта:")

    model_name = st.selectbox(
        "Модель оценки риска",
        list(MODEL_OPTIONS.keys()),
        help="Каждая модель имеет свои особенности в оценке риска"
    )
    key_prefix, train_fn, load_fn, enrich_fn = MODEL_OPTIONS[model_name]

    # Обучение модели
    if st.button(
            f"Обучить модель {model_name}",
            help="Нажмите для обучения модели на тестовых данных"
    ):
        try:
            with st.spinner(f"Обучаем модель {model_name}..."):
                df_raw = pd.read_csv("data/raw/default_risk.csv")
                model, grid = train_fn(df_raw, save=True)
            st.success(f"Модель {model_name} успешно обучена! 🎉 (ROC-AUC: {grid.best_score_:.3f})")
        except FileNotFoundError:
            st.error("❌ Не найден файл с тестовыми данными")
        except Exception as e:
            logger.exception("Ошибка при обучении модели")
            st.error(f"❌ Ошибка при обучении модели: {str(e)}")

    st.markdown("---")

    # Секция загрузки и обработки данных
    st.markdown("### 2. Анализ данных")
    st.markdown("Загрузите файл с кредитными заявками для анализа:")

    uploaded = st.file_uploader(
        "Загрузить CSV/Excel с заявками",
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
                    key=f"map_{key_prefix}_{field}",
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

            # Прогнозирование рисков
            try:
                model = load_fn()
                df_enriched = enrich_fn(df_mapped, model)

                if st.checkbox("Показать пример обогащённых данных"):
                    st.write("Пример данных после оценки риска:", df_enriched.head())

                # Статистика по рискам
                total = len(df_enriched)
                high_risk = (df_enriched["Default_Pred"] == 1).sum()
                low_risk = total - high_risk

                st.markdown("### 4. Результаты анализа")

                # Показываем основную статистику
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего заявок", total)
                with col2:
                    st.metric("Низкий риск", low_risk)
                with col3:
                    st.metric("Высокий риск", high_risk)

                # Создание и отображение визуализаций
                st.markdown("### 5. Визуализация результатов")
                charts = create_visualizations(
                    df_enriched,
                    pred_col="Default_Pred",
                    prob_col="Default_Prob"
                )
                for chart in charts:
                    st.altair_chart(chart)

                # Скачивание результатов
                st.markdown("### 6. Выгрузка результатов")
                csv = df_enriched.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Скачать результаты анализа (CSV)",
                    csv,
                    f"default_risk_enriched_{key_prefix}.csv",
                    mime="text/csv",
                    help="Скачать данные с результатами оценки риска"
                )

            except FileNotFoundError:
                st.error("❌ Модель не найдена. Пожалуйста, сначала обучите модель.")
            except NotFittedError:
                st.error("❌ Модель не обучена. Пожалуйста, сначала обучите модель.")
            except Exception as e:
                logger.exception("Ошибка при прогнозировании")
                st.error(f"❌ Ошибка при обработке данных: {str(e)}")

    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении")
        st.error(f"❌ Произошла неожиданная ошибка: {str(e)}")

app() 