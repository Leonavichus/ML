import streamlit as st

def app():
    st.title("Транзакции")
    st.markdown("Загрузка данных для анализа транзакций")
    st.file_uploader("Загрузить файл (CSV/Excel)", type=["csv", "xlsx"], key="transactions_uploader")
