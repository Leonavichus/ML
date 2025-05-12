import streamlit as st
#from src.chat_ollama import get_response

def app():
    st.title("Чат")

    # Инициализация истории сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Отображаем чат
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Ввод пользователя
    user_input = st.chat_input("Введите сообщение...")
    if user_input:
        # Добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Получаем ответ от Ollama
        #with st.spinner("Ollama печатает..."):
            #response = get_response(user_input)
        # Добавляем ответ ассистента
        #st.session_state.messages.append({"role": "assistant", "content": response})
        # Обновляем вывод чата
        #st.experimental_rerun()