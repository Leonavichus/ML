import streamlit as st
import pandas as pd
import altair as alt
from pandas.errors import EmptyDataError, ParserError
from src.model_transactions import (
    train_iforest, train_ocsvm, train_lof,
    enrich
)

MODEL_OPTIONS = {
    "Isolation Forest": train_iforest,
    "One-Class SVM":    train_ocsvm,
    "Local Outlier Factor": train_lof
}

def app():
    st.title("Обнаружение аномалий в транзакциях")

    # Выбор метода
    st.markdown("#### Выберите модель для обучения и обработки")
    method = st.selectbox("Метод обнаружения аномалий", list(MODEL_OPTIONS.keys()))

    # Кнопка обучения
    if st.button(f"Обучить {method}"):
        try:
            df_raw = pd.read_csv("data/raw/metaverse_transactions_dataset.csv")
            with st.spinner(f"Обучаем {method}..."):
                MODEL_OPTIONS[method](df_raw, save=True)
            st.success(f"{method} обучен!")
        except Exception as e:
            st.error(f"Ошибка при обучении: {e}")

    st.markdown("---")
    uploaded = st.file_uploader("Загрузить CSV/Excel с транзакциями", type=["csv","xlsx"])
    if not uploaded:
        return

    try:
        df_user = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    except (EmptyDataError, ParserError) as e:
        st.error(f"Ошибка чтения файла: {e}")
        return

    # Обогащение
    try:
        df_enriched = enrich(df_user, method)
    except Exception as e:
        st.error(f"Ошибка при обогащении: {e}")
        return

    st.write("Пример обогащённых данных:", df_enriched.head())

    # 1) Гистограмма скоринга
    hist = alt.Chart(df_enriched).mark_bar().encode(
        x=alt.X("AnomalyScore:Q", bin=alt.Bin(maxbins=50), title="Anomaly Score"),
        y=alt.Y("count()", title="Количество"),
        color=alt.Color("AnomalyPred:N", scale=alt.Scale(domain=[1,-1], range=["#4caf50","#e91e63"]),
                        legend=alt.Legend(title="Normal / Anomaly"))
    ).properties(width=600, height=300)
    st.altair_chart(hist)

    # 2) Pie доли аномалий
    pie = alt.Chart(df_enriched).mark_arc(innerRadius=50).encode(
        theta="count()",
        color=alt.Color("AnomalyPred:N", scale=alt.Scale(domain=[1,-1], range=["#4caf50","#e91e63"])),
        tooltip=["AnomalyPred:N","count()"]
    ).properties(width=300, height=300)
    st.altair_chart(pie)

    # 3) Аномалии по часам
    df_enriched["Timestamp"] = pd.to_datetime(df_enriched["Timestamp"])
    df_hours = df_enriched.groupby(df_enriched["Timestamp"].dt.hour)["AnomalyPred"]\
                          .apply(lambda s: (s == -1).sum())\
                          .reset_index(name="Anomalies")
    line = alt.Chart(df_hours).mark_line(point=True).encode(
        x=alt.X("Timestamp:O", title="Час дня"),
        y=alt.Y("Anomalies:Q", title="Количество аномалий")
    ).properties(width=600, height=300)
    st.altair_chart(line)

    # 4) Scatter Amount vs Score
    sample = df_enriched.sample(min(1000, len(df_enriched)), random_state=42)
    scatter = alt.Chart(sample).mark_circle().encode(
        x="Amount:Q", y="AnomalyScore:Q",
        color=alt.Color("AnomalyPred:N", legend=None),
        tooltip=["Amount","AnomalyScore","AnomalyPred"]
    ).properties(width=600, height=300)
    st.altair_chart(scatter)

    # Скачать результат
    csv = df_enriched.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать обогащённый CSV", csv, "anomaly_enriched.csv", mime="text/csv")
