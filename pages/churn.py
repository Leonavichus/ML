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

st.set_page_config(
    page_title="Отток клиентов | Analytics App",
    page_icon="📉",
    layout="wide"
)

# Словарь доступных моделей для прогнозирования оттока
MODEL_OPTIONS: Dict[str, Tuple[str, Callable, Callable, Callable]] = {
    'Random Forest': ('rf', train_churn_rf, load_churn_rf, enrich_churn_rf),
    'Neural Network': ('nn', train_churn_nn, load_churn_nn, enrich_churn_nn),
    'XGBoost': ('xgb', train_churn_xgb, load_churn_xgb, enrich_churn_xgb),
    'LightGBM': ('lgbm', train_churn_lgbm, load_churn_lgbm, enrich_churn_lgbm),
}

# Обязательные поля для анализа оттока с описаниями
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


def create_visualization(df: pd.DataFrame, pred_col: str, prob_col: str) -> List[alt.Chart]:
    """
    Создание расширенных визуализаций для анализа оттока клиентов.

    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с прогнозами
        pred_col (str): Название колонки с прогнозами
        prob_col (str): Название колонки с вероятностями

    Возвращает:
        List[alt.Chart]: Список интерактивных графиков визуализации
    """
    # Круговая диаграмма распределения прогнозов
    pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta="count()",
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"]),
            legend=alt.Legend(title="Прогноз оттока")
        ),
        tooltip=[
            alt.Tooltip(f"{pred_col}:N", title="Отток"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("count():Q", title="Процент", format=".1%")
        ]
    ).properties(
        width=300,
        height=300,
        title="Распределение прогнозов оттока"
    )

    # Гистограмма вероятностей
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            f"{prob_col}:Q",
            bin=alt.Bin(maxbins=30),
            title="Вероятность оттока"
        ),
        y=alt.Y(
            "count()",
            title="Количество клиентов"
        ),
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"]),
            legend=alt.Legend(title="Прогноз оттока")
        ),
        tooltip=[
            alt.Tooltip(f"{prob_col}:Q", title="Вероятность", format=".2f"),
            alt.Tooltip("count()", title="Количество")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение вероятностей оттока"
    )

    # Зависимость оттока от возраста и баланса
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("Age:Q", title="Возраст"),
        y=alt.Y("Balance:Q", title="Баланс"),
        color=alt.Color(
            f"{pred_col}:N",
            scale=alt.Scale(domain=[0, 1], range=["#4caf50", "#e91e63"]),
            legend=alt.Legend(title="Прогноз оттока")
        ),
        tooltip=[
            alt.Tooltip("Age:Q", title="Возраст"),
            alt.Tooltip("Balance:Q", title="Баланс", format=",.2f"),
            alt.Tooltip(f"{prob_col}:Q", title="Вероятность оттока", format=".2%"),
            alt.Tooltip("Geography:N", title="Регион"),
            alt.Tooltip("Gender:N", title="Пол")
        ]
    ).properties(
        width=600,
        height=400,
        title="Отток в зависимости от возраста и баланса"
    )

    # Комплексный анализ по регионам
    region_analysis = alt.vconcat(
        # Верхний график: Распределение вероятностей оттока по регионам
        alt.Chart(df).transform_density(
            f'{prob_col}',
            as_=[f'{prob_col}', 'density'],
            groupby=['Geography'],
            steps=100
        ).mark_area(
            opacity=0.6,
            interpolate='monotone'
        ).encode(
            x=alt.X(f'{prob_col}:Q', title='Вероятность оттока'),
            y=alt.Y('density:Q', title='Плотность', stack=None),
            color=alt.Color('Geography:N', 
                          title='Регион',
                          scale=alt.Scale(scheme='category10')),
            tooltip=[
                alt.Tooltip(f'{prob_col}:Q', title='Вероятность', format='.2%'),
                alt.Tooltip('density:Q', title='Плотность'),
                alt.Tooltip('Geography:N', title='Регион')
            ]
        ).properties(
            width=600,
            height=200,
            title='Распределение вероятностей оттока по регионам'
        ),
        # Нижний график: Статистика по регионам
        alt.Chart(df).transform_aggregate(
            total='count()',
            churned=f'sum(({pred_col} == 1)*1)',
            avg_prob=f'mean({prob_col})',
            avg_balance='mean(Balance)',
            groupby=['Geography']
        ).transform_calculate(
            churn_rate='datum.churned / datum.total'
        ).mark_bar().encode(
            x=alt.X('Geography:N', title='Регион'),
            y=alt.Y('total:Q', title='Количество клиентов'),
            color=alt.Color('churn_rate:Q', 
                          title='Доля оттока',
                          scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
            tooltip=[
                alt.Tooltip('Geography:N', title='Регион'),
                alt.Tooltip('total:Q', title='Всего клиентов'),
                alt.Tooltip('churned:Q', title='Ушедшие клиенты'),
                alt.Tooltip('churn_rate:Q', title='Доля оттока', format='.2%'),
                alt.Tooltip('avg_prob:Q', title='Ср. вероятность', format='.2%'),
                alt.Tooltip('avg_balance:Q', title='Ср. баланс', format=',.2f')
            ]
        ).properties(
            width=600,
            height=200,
            title='Статистика оттока по регионам'
        )
    ).properties(
        title={
            "text": "Анализ оттока по регионам",
            "fontSize": 16
        }
    )

    # Анализ продуктовой корзины
    product_analysis = alt.vconcat(
        # Верхний график: Распределение клиентов по количеству продуктов
        alt.Chart(df).transform_aggregate(
            count='count()',
            churned=f'sum(({pred_col} == 1)*1)',
            groupby=['NumOfProducts', 'IsActiveMember']
        ).transform_calculate(
            churn_rate='datum.churned / datum.count'
        ).mark_bar().encode(
            x=alt.X('NumOfProducts:O', title='Количество продуктов'),
            y=alt.Y('count:Q', title='Количество клиентов'),
            color=alt.Color('IsActiveMember:N',
                          scale=alt.Scale(domain=[0, 1], range=['#ff9800', '#2196f3']),
                          title='Активность'),
            tooltip=[
                alt.Tooltip('NumOfProducts:O', title='Продуктов'),
                alt.Tooltip('count:Q', title='Клиентов'),
                alt.Tooltip('churned:Q', title='Ушли'),
                alt.Tooltip('churn_rate:Q', title='Доля оттока', format='.2%'),
                alt.Tooltip('IsActiveMember:N', title='Активный клиент')
            ]
        ).properties(
            width=600,
            height=200,
            title='Распределение клиентов по количеству продуктов'
        ),
        # Нижний график: Средняя вероятность оттока
        alt.Chart(df).transform_aggregate(
            avg_prob=f'mean({prob_col})',
            avg_balance='mean(Balance)',
            avg_tenure='mean(Tenure)',
            groupby=['NumOfProducts', 'IsActiveMember']
        ).mark_line(point=True).encode(
            x=alt.X('NumOfProducts:O', title='Количество продуктов'),
            y=alt.Y('avg_prob:Q', title='Средняя вероятность оттока', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('IsActiveMember:N',
                          scale=alt.Scale(domain=[0, 1], range=['#ff9800', '#2196f3']),
                          title='Активность'),
            tooltip=[
                alt.Tooltip('NumOfProducts:O', title='Продуктов'),
                alt.Tooltip('avg_prob:Q', title='Ср. вероятность', format='.2%'),
                alt.Tooltip('avg_balance:Q', title='Ср. баланс', format=',.2f'),
                alt.Tooltip('avg_tenure:Q', title='Ср. стаж', format='.1f'),
                alt.Tooltip('IsActiveMember:N', title='Активный клиент')
            ]
        ).properties(
            width=600,
            height=200,
            title='Зависимость вероятности оттока от количества продуктов'
        )
    ).properties(
        title={
            "text": "Анализ продуктовой корзины и активности",
            "fontSize": 16
        }
    )

    return [pie, hist, scatter, region_analysis, product_analysis]


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
    Основной интерфейс приложения прогнозирования оттока клиентов.

    Функционал:
    - Выбор и обучение модели прогнозирования оттока
    - Загрузка и анализ пользовательских данных
    - Визуализация результатов прогнозирования
    - Выгрузка обогащенных данных
    """
    st.title("Прогнозирование оттока клиентов")

    # Добавляем описание приложения
    st.markdown("""
    Это приложение помогает прогнозировать отток клиентов на основе их характеристик.
    Вы можете:
    - Выбрать и обучить модель прогнозирования
    - Загрузить данные клиентов для анализа
    - Получить визуальный анализ прогнозов
    - Выгрузить результаты прогнозирования
    """)

    # Секция выбора и обучения модели
    st.markdown("### 1. Выбор модели")
    st.markdown("Выберите алгоритм для прогнозирования оттока:")

    model_name = st.selectbox(
        "Модель прогнозирования:",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Каждая модель имеет свои особенности в прогнозировании"
    )
    key_prefix, train_fn, load_fn, enrich_fn = MODEL_OPTIONS[model_name]

    # Обучение модели
    if st.button(f"Обучить модель {model_name}", help="Нажмите для обучения модели на тестовых данных"):
        try:
            with st.spinner(f"Обучаем модель {model_name}..."):
                df_raw = pd.read_csv("data/raw/churn.csv")
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
    st.markdown("Загрузите файл с данными клиентов для анализа:")

    uploaded = st.file_uploader(
        "Загрузить CSV/Excel с данными",
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

            # Прогнозирование
            try:
                model = load_fn()
                df_enriched = enrich_fn(df_mapped, model)

                if st.checkbox("Показать пример обогащённых данных"):
                    st.write("Пример данных после обогащения:", df_enriched.head())

                # Статистика по оттоку
                total = len(df_enriched)
                churn = (df_enriched["Exited_Pred"] == 1).sum()
                loyal = total - churn

                st.markdown("### 4. Результаты анализа")

                # Показываем основную статистику
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего клиентов", total)
                with col2:
                    st.metric("Лояльные", loyal)
                with col3:
                    st.metric("Склонные к оттоку", churn)

                # Создание и отображение визуализаций
                st.markdown("### 5. Визуализация результатов")
                charts = create_visualization(
                    df_enriched,
                    pred_col="Exited_Pred",
                    prob_col="Exited_Prob"
                )
                for chart in charts:
                    st.altair_chart(chart)

                # Скачивание результатов
                st.markdown("### 6. Выгрузка результатов")
                csv_data = df_enriched.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Скачать результаты анализа (CSV)",
                    data=csv_data,
                    file_name=f"churn_enriched_{key_prefix}.csv",
                    mime="text/csv",
                    help="Скачать данные с результатами прогнозирования"
                )

            except FileNotFoundError:
                st.error("❌ Модель не найдена. Пожалуйста, сначала обучите модель.")
            except NotFittedError:
                st.error("❌ Модель не обучена. Пожалуйста, сначала обучите модель.")
            except Exception as e:
                logger.exception("Ошибка при прогнозировании")
                st.error(f"❌ Ошибка при обогащении данных: {str(e)}")

    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении")
        st.error(f"❌ Произошла неожиданная ошибка: {str(e)}")

app() 