import streamlit as st
from streamlit_option_menu import option_menu
from pages_custom import page_chat, page_churn, page_segmentation, page_transactions, page_default_risk

st.set_page_config(page_title="Analytics App", page_icon="📊", layout="wide")

# Собираем пункты меню с иконками
menu_items = [
    {"name": "Чат", "icon": "chat-dots"},
    {"name": "Отток клиентов", "icon": "graph-down"},
    {"name": "Сегментация клиентов", "icon": "people-fill"},
    {"name": "Транзакции", "icon": "currency-exchange"},
    {"name": "Риск дефолта", "icon": "exclamation-triangle-fill"},
]

with st.sidebar:
    st.sidebar.header("Analytics App")
    choice = option_menu(
        menu_title=None,
        options=[item["name"] for item in menu_items],
        icons=[item["icon"] for item in menu_items],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "background-color": "transparent",
            },
            "nav-link-selected": {
                "background-color": "indigo",
            },
        },
    )

# Mapping
PAGES = {
    "Чат": page_chat,
    "Отток клиентов": page_churn,
    "Сегментация клиентов": page_segmentation,
    "Транзакции": page_transactions,
    "Риск дефолта": page_default_risk,
}

PAGES[choice].app()
