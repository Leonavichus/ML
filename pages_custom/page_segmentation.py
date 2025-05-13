import streamlit as st
import pandas as pd
import altair as alt
from pandas.errors import EmptyDataError, ParserError
from src.model_segmentation import (
    train_kmeans, train_gmm, train_nn_segmentation,
    enrich_segmentation
)

# Методы сегментации
METHODS = {
    'KMeans': ('kmeans', train_kmeans),
    'Gaussian Mixture': ('gmm', train_gmm),
    'Neural Network': ('nn', train_nn_segmentation)
}

# Обязательные поля для маппинга
REQUIRED_FIELDS_SEG = [
    ('CustomerDOB', 'Дата рождения'),
    ('CustGender', 'Пол'),
    ('CustLocation', 'Локация'),
    ('CustAccountBalance', 'Баланс счёта'),
    ('TransactionAmount', 'Сумма транзакции')
]

def app():
    st.title('Сегментация клиентов')

    # Выбор метода сегментации
    st.markdown('#### Выберите метод сегментации')
    method_name = st.selectbox('Метод', list(METHODS.keys()))
    method_key, train_fn = METHODS[method_name]

    # Параметр числа сегментов
    n_clusters = st.slider('Количество сегментов', 2, 10, 4)

    st.markdown('#### Выберите признаки для сегментации')
    selected_features = st.multiselect(
        'Выберите колонки для кластеризации',
        ['CustAccountBalance', 'TransactionAmount', 'Age', 'CustGender', 'CustLocation'],
        default=['CustAccountBalance', 'TransactionAmount', 'Age']
    )
    if not selected_features:
        st.error('Нужно выбрать хотя бы один признак для сегментации')
        return

    # Кнопка обучения с индикатором
    if st.button(f'Обучить модель ({method_name}) на встроенных данных'):
        try:
            df_raw = pd.read_csv('data/raw/transactions.csv')
            with st.spinner(f'Обучаем модель {method_name} с {n_clusters} сегментами...'):
                train_fn(df_raw, n_clusters=n_clusters, features=selected_features, save=True)
            st.success(f'Модель сегментации {method_name} обучена с {n_clusters} сегментами.')
        except Exception as e:
            st.error(f'Ошибка при обучении: {e}')

    st.markdown('---')

    uploaded = st.file_uploader('Загрузить CSV/Excel для сегментации', type=['csv','xlsx'])
    if not uploaded:
        return
    try:
        df_user = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
    except (EmptyDataError, ParserError) as e:
        st.error(f'Ошибка при чтении файла: {e}')
        return
    except Exception as e:
        st.error(f'Неизвестная ошибка при чтении файла: {e}')
        return

    st.write('Исходные колонки:', df_user.columns.tolist())

    # Маппинг полей
    mapping = {}
    cols = df_user.columns.tolist()
    for field, label in REQUIRED_FIELDS_SEG:
        mapping[field] = st.selectbox(f'{label} →', ['<отсутствует>'] + cols, key=f'map_seg_{field}')
    missing = [lbl for fld, lbl in REQUIRED_FIELDS_SEG if mapping[fld] == '<отсутствует>']
    if missing:
        st.error('Укажите соответствие для полей: ' + ', '.join(missing))
        return

    # Переименование и обогащение
    df_mapped = df_user.rename(columns={mapping[f]: f for f, _ in REQUIRED_FIELDS_SEG})
    try:
        df_enriched = enrich_segmentation(df_mapped, method_key)
    except Exception as e:
        st.error(f'Ошибка при обогащении данными: {e}')
        return

    st.write("Пример:", df_enriched.head())

    # 1) Круговая диаграмма
    pie = alt.Chart(df_enriched).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("count()", title="Клиенты"),
        color=alt.Color("SegmentName:N", title="Сегмент"),
        tooltip=["SegmentName", "count()"]
    ).properties(width=300, height=300)
    st.altair_chart(pie)

    # 2) Точечный график (Age vs TransactionAmount)
    scatter = alt.Chart(df_enriched.sample(min(1000, len(df_enriched)))).mark_circle(size=60, opacity=0.6).encode(
        x="Age:Q", y="TransactionAmount:Q", color="SegmentName:N",
        tooltip=["SegmentName", "Age", "TransactionAmount"]
    ).properties(width=600, height=300)
    st.altair_chart(scatter)

    # 3) Гистограмма баланса
    hist = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X("CustAccountBalance:Q", bin=alt.Bin(maxbins=30)), y="count()",
        color="SegmentName:N"
    ).properties(width=600, height=300)
    st.altair_chart(hist)

    # 4) Столбчатая диаграмма числа клиентов
    bar = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X("SegmentName:N", sort=None), y="count()", color="SegmentName:N"
    ).properties(width=600, height=300)
    st.altair_chart(bar)

    # Скачать CSV
    csv = df_enriched.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать результат", csv, f"segmented_{method_key}.csv")