import streamlit as st

# Настройка темы и стиля
st.set_page_config(
    page_title="Analytics App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Применение пользовательских стилей CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 1rem;
    }
    .st-emotion-cache-1wrcr25 {
        margin-bottom: 0.5rem;
    }
    /* Стиль для главной страницы в sidebar */
    section[data-testid="stSidebar"] .element-container:first-child .stMarkdown {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Главная страница
st.title("📈 Analytics App")
st.markdown("### Добро пожаловать в Analytics App!")

st.markdown("""
Выберите нужный раздел в меню слева:

- 💬 **Чат** - Интерактивный чат с AI-моделью
- 📉 **Отток клиентов** - Анализ и прогнозирование оттока
- 👥 **Сегментация клиентов** - Кластеризация и анализ сегментов
- 💱 **Транзакции** - Анализ финансовых операций
- ⚠️ **Риск дефолта** - Оценка кредитных рисков
""")