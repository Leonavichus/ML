import ollama
import streamlit as st
import torch

def app():
    st.title("Чат с моделью Ollama")

    # Инициализируем историю сообщений в сессии
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Получаем список моделей от Ollama
    try:
        response = ollama.list()
        # Правильная обработка результата ollama.list()
        models = [model.model for model in response.models]

        if not models:
            st.warning("Список моделей пуст. Проверьте, что Ollama запущена и есть загруженные модели.")
            return
    except Exception as e:
        st.error(f"Не удалось получить список моделей: {e}")
        return

    # Выбор модели из списка
    if "model" not in st.session_state or st.session_state.model not in models:
        st.session_state.model = models[0]
    st.session_state.model = st.selectbox(
        "Выберите модель для чата",
        models,
        index=models.index(st.session_state.model)
    )

    # Генератор потокового ответа от Ollama
    def generate_response():
        # Определяем устройство (CPU или GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Запускаем chat с параметром stream=True
        for chunk in ollama.chat(
            model=st.session_state.model,
            messages=st.session_state.messages,
            stream=True
        ):
            yield chunk["message"]["content"]

    # Выводим историю диалога
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ввод нового сообщения
    user_input = st.chat_input("Введите сообщение...")
    if user_input:
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Ответ ассистента
        with st.chat_message("assistant"):
            full_response = ""
            # создаём один placeholder для всего текста
            placeholder = st.empty()
            for part in generate_response():
                full_response += part
                # обновляем содержимое placeholder'а полностью
                placeholder.markdown(full_response)
            # сохраняем полный ответ в историю
            st.session_state.messages.append({"role": "assistant", "content": full_response})
