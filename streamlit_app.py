import streamlit as st

pages = [
    st.Page("app.py", title="📊 Главная страница"),
    st.Page("pages/chat.py", title=" 💬 Чат"),
    st.Page("pages/churn.py", title="📉 Отток клиентов"),
    st.Page("pages/segmentation.py", title="👥 Сегментация клиентов"),
    st.Page("pages/transactions.py", title="💱 Транзакции"),
    st.Page("pages/default_risk.py", title="⚠️ Риск дефолта"),
]

pg = st.navigation(pages)
pg.run()