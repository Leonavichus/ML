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

# Конфигурация методов сегментации
METHODS: Dict[str, Tuple[str, Callable]] = {
    'KMeans': ('kmeans', train_kmeans),
    'Gaussian Mixture': ('gmm', train_gmm),
    'Neural Network': ('nn', train_nn_segmentation)
}

# Обязательные поля для сопоставления
REQUIRED_FIELDS_SEG: List[Tuple[str, str]] = [
    ('CustomerDOB', 'Дата рождения'),
    ('CustGender', 'Пол'),
    ('CustLocation', 'Локация'),
    ('CustAccountBalance', 'Баланс счёта'),
    ('TransactionAmount', 'Сумма транзакции')
]

def create_visualizations(df: pd.DataFrame) -> List[alt.Chart]:
    """
    Создание визуализаций для клиентских сегментов.
    
    Аргументы:
        df (pd.DataFrame): Обогащенный датафрейм с предсказанными сегментами
        
    Возвращает:
        List[alt.Chart]: Список графиков визуализации
    """
    # Круговая диаграмма
    pie = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("count()", title="Клиенты"),
        color=alt.Color("SegmentName:N", title="Сегмент"),
        tooltip=["SegmentName", "count()"]
    ).properties(width=300, height=300)

    # Точечная диаграмма
    scatter = alt.Chart(df.sample(min(1000, len(df)))).mark_circle(size=60, opacity=0.6).encode(
        x="Age:Q",
        y="TransactionAmount:Q",
        color="SegmentName:N",
        tooltip=["SegmentName", "Age", "TransactionAmount"]
    ).properties(width=600, height=300)

    # Гистограмма баланса
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("CustAccountBalance:Q", bin=alt.Bin(maxbins=30)),
        y="count()",
        color="SegmentName:N"
    ).properties(width=600, height=300)

    # Столбчатая диаграмма количества клиентов
    bar = alt.Chart(df).mark_bar().encode(
        x=alt.X("SegmentName:N", sort=None),
        y="count()",
        color="SegmentName:N"
    ).properties(width=600, height=300)

    return [pie, scatter, hist, bar]

def read_user_data(uploaded_file: Union[UploadedFile, str]) -> pd.DataFrame:
    """
    Чтение и валидация загруженных пользователем данных.
    
    Аргументы:
        uploaded_file: Объект загруженного файла Streamlit или путь к файлу
        
    Возвращает:
        pd.DataFrame: Загруженный датафрейм
        
    Вызывает:
        ValueError: Если не удалось прочитать файл или неподдерживаемый формат
    """
    try:
        if isinstance(uploaded_file, str):
            if uploaded_file.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        else:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        raise ValueError("Неподдерживаемый формат файла")
    except (EmptyDataError, ParserError) as e:
        raise ValueError(f"Ошибка при чтении файла: {str(e)}")
    except Exception as e:
        raise ValueError(f"Неизвестная ошибка при чтении файла: {str(e)}")

def app():
    """Основной интерфейс приложения сегментации клиентов."""
    st.title('Сегментация клиентов')

    # Выбор метода
    st.markdown('#### Выберите метод сегментации')
    method_name = st.selectbox('Метод', list(METHODS.keys()))
    method_key, train_fn = METHODS[method_name]

    # Параметры сегментации
    n_clusters = st.slider('Количество сегментов', 2, 10, 4)

    st.markdown('#### Выберите признаки для сегментации')
    available_features = ['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation']
    selected_features = st.multiselect(
        'Выберите колонки для кластеризации',
        available_features,
        default=['CustAccountBalance', 'TransactionAmount', 'Age']
    )

    if not selected_features:
        st.error('Нужно выбрать хотя бы один признак для сегментации')
        return

    # Обучение модели
    if st.button(f'Обучить модель ({method_name}) на встроенных данных'):
        try:
            df_raw = pd.read_csv('data/raw/transactions.csv')
            with st.spinner(f'Обучаем модель {method_name} с {n_clusters} сегментами...'):
                train_fn(
                    df_raw,
                    n_clusters=n_clusters,
                    features=selected_features,
                    save=True
                )
            st.success(f'Модель сегментации {method_name} обучена с {n_clusters} сегментами.')
        except Exception as e:
            logger.exception("Ошибка при обучении модели")
            st.error(f'Ошибка при обучении: {str(e)}')

    st.markdown('---')

    # Загрузка и обработка данных
    uploaded = st.file_uploader('Загрузить CSV/Excel для сегментации', type=['csv','xlsx'])
    if not uploaded:
        return

    try:
        df_user = read_user_data(uploaded)
        st.write('Исходные колонки:', df_user.columns.tolist())

        # Сопоставление полей
        mapping = {}
        cols = df_user.columns.tolist()
        for field, label in REQUIRED_FIELDS_SEG:
            mapping[field] = st.selectbox(
                f'{label} →',
                ['<отсутствует>'] + cols,
                key=f'map_seg_{field}'
            )

        # Валидация
        missing = [lbl for fld, lbl in REQUIRED_FIELDS_SEG if mapping[fld] == '<отсутствует>']
        if missing:
            st.error('Укажите соответствие для полей: ' + ', '.join(missing))
            return

        # Обработка и обогащение данных
        df_mapped = df_user.rename(columns={mapping[f]: f for f, _ in REQUIRED_FIELDS_SEG})
        try:
            df_enriched = enrich_segmentation(df_mapped, method_key)
            st.write("Пример:", df_enriched.head())

            # Визуализации
            charts = create_visualizations(df_enriched)
            for chart in charts:
                st.altair_chart(chart)

            # Скачивание результатов
            csv = df_enriched.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать результат",
                csv,
                f"segmented_{method_key}.csv",
                mime="text/csv"
            )

        except FileNotFoundError:
            st.error("Модель не найдена. Пожалуйста, обучите модель сначала.")
        except Exception as e:
            logger.exception("Ошибка при обогащении данных")
            st.error(f'Ошибка при обогащении данными: {str(e)}')

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        logger.exception("Неожиданная ошибка в приложении сегментации")
        st.error(f'Неожиданная ошибка: {str(e)}')