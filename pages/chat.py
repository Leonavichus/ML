import ollama
import streamlit as st
import torch
from typing import Generator, List, Dict
from logging import getLogger
import time

logger = getLogger(__name__)

st.set_page_config(
    page_title="Чат | Analytics App",
    page_icon="💬",
    layout="wide"
)

def get_available_models() -> List[str]:
    """
    Получение списка доступных моделей из сервиса Ollama.

    Возвращает:
        List[str]: Список имен доступных моделей

    Вызывает:
        Exception: Если не удалось получить список моделей из Ollama
    """
    try:
        with st.spinner("🔄 Загрузка доступных моделей..."):
            response = ollama.list()
            models = [model.model for model in response.models]
            if not models:
                raise ValueError("❌ Не найдены модели в сервисе Ollama")
            return models
    except Exception as e:
        logger.error(f"Не удалось получить список моделей: {str(e)}")
        raise


def generate_response(model: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Генерация потокового ответа от выбранной модели.

    Аргументы:
        model (str): Имя модели Ollama для использования
        messages (List[Dict[str, str]]): История чата

    Возвращает:
        str: Части ответа от модели
    """
    try:
        for chunk in ollama.chat(
                model=model,
                messages=messages,
                stream=True
        ):
            yield chunk["message"]["content"]
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        yield f"❌ Ошибка: Не удалось сгенерировать ответ. Пожалуйста, попробуйте снова."


def clear_chat_history():
    """Очистка истории чата"""
    st.session_state.messages = []


def app():
    """Основной интерфейс приложения чата."""
    st.title("💬 Чат")

    # Боковая панель с настройками
    with st.sidebar:
        st.subheader("⚙️ Настройки чата")
        if st.button("🗑️ Очистить историю чата", use_container_width=True):
            clear_chat_history()

        st.divider()
        st.markdown("""
        ### 📝 Инструкция
        1. Выберите модель из списка доступных
        2. Введите ваше сообщение в поле внизу
        3. Дождитесь ответа модели

        ### ℹ️ Информация
        - Используется сервис Ollama
        - Модель работает на: """ + ("🖥️ CPU" if torch.cuda.is_available() else "🎮 GPU"))

    # Инициализация состояния сессии
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        # Получение и отображение выбора модели
        models = get_available_models()
        if "model" not in st.session_state or st.session_state.model not in models:
            st.session_state.model = models[0]

        st.session_state.model = st.selectbox(
            "🤖 Выберите модель для чата",
            models,
            index=models.index(st.session_state.model)
        )

        st.divider()

        # Отображение истории чата
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=icon):
                    st.markdown(msg["content"])

        # Обработка пользовательского ввода
        user_input = st.chat_input("💭 Введите ваше сообщение...")
        if user_input:
            # Добавление сообщения пользователя в историю
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)

            # Генерация и отображение ответа ассистента
            with st.chat_message("assistant", avatar="🤖"):
                full_response = ""
                message_placeholder = st.empty()

                # Добавляем индикатор прогресса
                with st.spinner("🤔 Генерация ответа..."):
                    for response_chunk in generate_response(st.session_state.model, st.session_state.messages):
                        full_response += response_chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Небольшая задержка для эффекта печати

                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"❌ Произошла ошибка: {str(e)}")
        logger.exception("Ошибка в приложении чата")
        st.info("🔄 Попробуйте перезагрузить страницу или проверить подключение к сервису Ollama")

app() 