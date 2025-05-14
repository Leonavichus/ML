import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Tuple, Callable, Union
from pandas.errors import EmptyDataError, ParserError
from streamlit.runtime.uploaded_file_manager import UploadedFile
from src.model_segmentation import (
    train_kmeans, train_gmm, train_nn_segmentation,
    enrich_segmentation
)
from logging import getLogger

logger = getLogger(__name__)

st.set_page_config(
    page_title="Сегментация клиентов | Analytics App",
    page_icon="👥",
    layout="wide"
)

# Словарь доступных методов сегментации с их описаниями
METHODS: Dict[str, Tuple[str, Callable]] = {
    'KMeans': ('kmeans', train_kmeans),  # Метод k-средних для базовой сегментации
    'Gaussian Mixture': ('gmm', train_gmm),  # Гауссовы смеси для сложных распределений
    'Neural Network': ('nn', train_nn_segmentation)  # Нейросеть для нелинейной сегментации
}

# Обязательные поля для сегментации с описаниями
REQUIRED_FIELDS_SEG: List[Tuple[str, str]] = [
    ('CustomerDOB', 'Дата рождения клиента'),
    ('CustGender', 'Пол клиента'),
    ('CustLocation', 'Местоположение клиента'),
    ('CustAccountBalance', 'Баланс счёта клиента'),
    ('TransactionAmount', 'Сумма транзакции клиента')
]


def create_visualizations(df: pd.DataFrame) -> List[alt.Chart]:
    """
    Создание расширенных визуализаций для анализа клиентских сегментов.

    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с предсказанными сегментами

    Возвращает:
        List[alt.Chart]: Список интерактивных графиков визуализации
    """
    # Круговая диаграмма распределения сегментов
    pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("count()", title="Количество клиентов"),
        color=alt.Color(
            "SegmentName:N",
            title="Сегмент",
            scale=alt.Scale(scheme="category10")
        ),
        tooltip=[
            alt.Tooltip("SegmentName:N", title="Сегмент"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("count():Q", title="Доля", format=".1%")
        ]
    ).properties(
        width=300,
        height=300,
        title="Распределение клиентов по сегментам"
    )

    # Точечная диаграмма возраст/транзакции
    scatter = alt.Chart(df.sample(min(1000, len(df)))).mark_circle(
        size=60,
        opacity=0.6
    ).encode(
        x=alt.X("Age:Q", title="Возраст"),
        y=alt.Y("TransactionAmount:Q", title="Сумма транзакции"),
        color=alt.Color(
            "SegmentName:N",
            title="Сегмент",
            scale=alt.Scale(scheme="category10")
        ),
        tooltip=[
            alt.Tooltip("SegmentName:N", title="Сегмент"),
            alt.Tooltip("Age:Q", title="Возраст"),
            alt.Tooltip("TransactionAmount:Q", title="Сумма", format=",.2f"),
            alt.Tooltip("CustLocation:N", title="Регион"),
            alt.Tooltip("CustGender:N", title="Пол")
        ]
    ).properties(
        width=600,
        height=400,
        title="Распределение клиентов по возрасту и суммам транзакций"
    )

    # Гистограмма распределения баланса по сегментам
    hist = alt.Chart(df).mark_bar(opacity=0.6).encode(
        x=alt.X(
            "CustAccountBalance:Q",
            bin=alt.Bin(maxbins=30),
            title="Баланс счета"
        ),
        y=alt.Y(
            "count()",
            stack=None,
            title="Количество клиентов"
        ),
        color=alt.Color(
            "SegmentName:N",
            title="Сегмент",
            scale=alt.Scale(scheme="category10")
        ),
        tooltip=[
            alt.Tooltip("SegmentName:N", title="Сегмент"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("CustAccountBalance:Q", title="Баланс", format=",.2f")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение баланса по сегментам"
    )

    # Тепловая карта сегментов по регионам
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X("CustLocation:N", title="Регион"),
        y=alt.Y("SegmentName:N", title="Сегмент"),
        color=alt.Color(
            "count()",
            title="Количество",
            scale=alt.Scale(scheme="viridis")
        ),
        tooltip=[
            alt.Tooltip("CustLocation:N", title="Регион"),
            alt.Tooltip("SegmentName:N", title="Сегмент"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("count():Q", title="Доля в регионе", format=".1%")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение сегментов по регионам"
    )

    # Распределение по полу в сегментах
    gender = alt.Chart(df).mark_bar().encode(
        x=alt.X("SegmentName:N", title="Сегмент"),
        y=alt.Y("count()", title="Количество клиентов"),
        color=alt.Color("CustGender:N", title="Пол"),
        tooltip=[
            alt.Tooltip("SegmentName:N", title="Сегмент"),
            alt.Tooltip("CustGender:N", title="Пол"),
            alt.Tooltip("count()", title="Количество"),
            alt.Tooltip("count():Q", title="Доля", format=".1%")
        ]
    ).properties(
        width=600,
        height=300,
        title="Гендерное распределение по сегментам"
    )

    # Боксплот возраста по сегментам
    boxplot = alt.Chart(df).mark_boxplot().encode(
        x=alt.X("SegmentName:N", title="Сегмент"),
        y=alt.Y("Age:Q", title="Возраст"),
        color=alt.Color(
            "SegmentName:N",
            title="Сегмент",
            scale=alt.Scale(scheme="category10")
        ),
        tooltip=[
            alt.Tooltip("SegmentName:N", title="Сегмент"),
            alt.Tooltip("Age:Q", title="Возраст"),
            alt.Tooltip("count()", title="Количество")
        ]
    ).properties(
        width=600,
        height=300,
        title="Распределение возраста по сегментам"
    )

    return [pie, scatter, hist, heatmap, gender, boxplot]


def read_user_data(uploaded_file: Union[UploadedFile, str]) -> pd.DataFrame:
    """
    Чтение и валидация загруженных пользователем данных.

    Аргументы:
        uploaded_file: Объект загруженного файла Streamlit или путь к файлу

    Возвращает:
        pd.DataFrame: Загруженный и валидированный датафрейм

    Вызывает:
        ValueError: Если формат файла не поддерживается или возникла ошибка при чтении
    """
    try:
        # Определяем тип входных данных и читаем файл
        if isinstance(uploaded_file, str):
            if uploaded_file.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Поддерживаются только файлы форматов CSV и Excel (.xlsx)")
        else:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
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
    Основной интерфейс приложения сегментации клиентов.

    Функционал:
    - Выбор и настройка метода сегментации
    - Загрузка и анализ клиентских данных
    - Визуализация результатов сегментации
    - Выгрузка обогащенных данных
    """
    st.title('Сегментация клиентов')

    # Добавляем описание приложения
    st.markdown("""
    Это приложение помогает сегментировать клиентов на основе их характеристик.
    Вы можете:
    - Выбрать метод сегментации и настроить его параметры
    - Загрузить данные клиентов для анализа
    - Получить визуальный анализ сегментов
    - Выгрузить результаты сегментации
    """)

    # Секция выбора метода
    st.markdown("### 1. Выбор метода сегментации")
    st.markdown("Выберите алгоритм для сегментации клиентов:")

    method_name = st.selectbox(
        'Метод сегментации',
        list(METHODS.keys()),
        help="Каждый метод имеет свои особенности в сегментации"
    )
    method_key, train_fn = METHODS[method_name]

    # Настройка параметров сегментации
    st.markdown("### 2. Настройка параметров")

    n_clusters = st.slider(
        'Количество сегментов',
        min_value=2,
        max_value=10,
        value=4,
        help="Количество групп, на которые будут разделены клиенты"
    )

    st.markdown("### 3. Выбор признаков")
    st.markdown("Выберите характеристики клиентов для сегментации:")

    available_features = [
        'CustAccountBalance',
        'TransactionAmount',
        'Age',
        'CustGender',
        'CustLocation'
    ]
    selected_features = st.multiselect(
        'Признаки для кластеризации',
        available_features,
        default=['CustAccountBalance', 'TransactionAmount', 'Age'],
        help="Характеристики, на основе которых будет производиться сегментация"
    )

    if not selected_features:
        st.warning('⚠️ Необходимо выбрать хотя бы один признак для сегментации')
        return

    # Обучение модели
    if st.button(
            f'Обучить модель {method_name}',
            help="Нажмите для обучения модели на тестовых данных"
    ):
        try:
            with st.spinner(f'Обучаем модель {method_name} с {n_clusters} сегментами...'):
                df_raw = pd.read_csv('data/raw/transactions.csv')
                train_fn(
                    df_raw,
                    n_clusters=n_clusters,
                    features=selected_features,
                    save=True
                )
            st.success(f'Модель {method_name} успешно обучена! 🎉')
        except FileNotFoundError:
            st.error("❌ Не найден файл с тестовыми данными")
        except Exception as e:
            logger.exception("Ошибка при обучении модели")
            st.error(f'❌ Ошибка при обучении модели: {str(e)}')

    st.markdown('---')

    # Секция загрузки и обработки данных
    st.markdown("### 4. Анализ данных")
    st.markdown("Загрузите файл с данными клиентов для сегментации:")

    uploaded = st.file_uploader(
        'Загрузить CSV/Excel с данными',
        type=['csv', 'xlsx'],
        help="Поддерживаются форматы CSV и Excel (.xlsx)"
    )

    if not uploaded:
        st.info("👆 Пожалуйста, загрузите файл с данными для анализа")
        return

    try:
        # Чтение и обработка данных
        df_user = read_user_data(uploaded)

        st.markdown("### 5. Сопоставление полей")
        st.markdown("Укажите соответствие между полями в вашем файле и требуемыми полями:")

        # Создаем две колонки для более компактного отображения маппинга
        cols = df_user.columns.tolist()
        mapping = {}
        col1, col2 = st.columns(2)

        for i, (field, label) in enumerate(REQUIRED_FIELDS_SEG):
            with col1 if i % 2 == 0 else col2:
                mapping[field] = st.selectbox(
                    f'{label} →',
                    ['<отсутствует>'] + cols,
                    key=f'map_seg_{field}',
                    help=f"Выберите поле, соответствующее {label}"
                )

        # Валидация маппинга
        missing = [lbl for fld, lbl in REQUIRED_FIELDS_SEG if mapping[fld] == '<отсутствует>']
        if missing:
            st.warning("⚠️ Пожалуйста, укажите соответствие для следующих полей:\n- " + "\n- ".join(missing))
            return

        # Обработка и обогащение данных
        with st.spinner("Обрабатываем данные..."):
            df_mapped = df_user.rename(columns={mapping[f]: f for f, _ in REQUIRED_FIELDS_SEG})

            if st.checkbox("Показать пример данных после сопоставления"):
                st.write("Пример данных после сопоставления полей:", df_mapped.head())

            try:
                df_enriched = enrich_segmentation(df_mapped, method_key)

                if st.checkbox("Показать пример обогащённых данных"):
                    st.write("Пример данных после сегментации:", df_enriched.head())

                # Статистика по сегментам
                total = len(df_enriched)
                segments = df_enriched["SegmentName"].value_counts()

                st.markdown("### 6. Результаты анализа")

                # Показываем основную статистику
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Всего клиентов", total)
                with col2:
                    st.metric("Количество сегментов", len(segments))

                # Показываем статистику по сегментам
                st.markdown("#### Размеры сегментов:")
                for segment, count in segments.items():
                    st.metric(
                        segment,
                        f"{count} ({count / total * 100:.1f}%)"
                    )

                # Создание и отображение визуализаций
                st.markdown("### 7. Визуализация результатов")
                charts = create_visualizations(df_enriched)
                for chart in charts:
                    st.altair_chart(chart)

                # Скачивание результатов
                st.markdown("### 8. Выгрузка результатов")
                csv = df_enriched.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Скачать результаты сегментации (CSV)",
                    csv,
                    f"segmented_{method_key}.csv",
                    mime="text/csv",
                    help="Скачать данные с результатами сегментации"
                )

            except FileNotFoundError:
                st.error("❌ Модель не найдена. Пожалуйста, сначала обучите модель.")
            except Exception as e:
                logger.exception("Ошибка при обогащении данных")
                st.error(f'❌ Ошибка при обработке данных: {str(e)}')

    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении")
        st.error(f'❌ Произошла неожиданная ошибка: {str(e)}')

app() 