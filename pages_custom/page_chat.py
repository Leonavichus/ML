import ollama
import streamlit as st
import torch
from typing import Generator, List, Dict, Optional, Union
from logging import getLogger

logger = getLogger(__name__)

def get_available_models() -> List[str]:
    """
    Получение списка доступных моделей из сервиса Ollama.
    
    Возвращает:
        List[str]: Список имен доступных моделей
        
    Вызывает:
        Exception: Если не удалось получить список моделей из Ollama
    """
    try:
        response = ollama.list()
        models = [model.model for model in response.models]
        if not models:
            raise ValueError("Не найдены модели в сервисе Ollama")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            yield chunk["message"]["content"]
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        yield f"Ошибка: Не удалось сгенерировать ответ. Пожалуйста, попробуйте снова."

def app():
    """Основной интерфейс приложения чата."""
    st.title("Чат с моделью Ollama")

    # Инициализация состояния сессии
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        # Получение и отображение выбора модели
        models = get_available_models()
        if "model" not in st.session_state or st.session_state.model not in models:
            st.session_state.model = models[0]
        
        st.session_state.model = st.selectbox(
            "Выберите модель для чата",
            models,
            index=models.index(st.session_state.model)
        )

        # Отображение истории чата
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Обработка пользовательского ввода
        user_input = st.chat_input("Введите сообщение...")
        if user_input:
            # Добавление сообщения пользователя в историю
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Генерация и отображение ответа ассистента
            with st.chat_message("assistant"):
                full_response = ""
                message_placeholder = st.empty()
                
                for response_chunk in generate_response(st.session_state.model, st.session_state.messages):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")
        logger.exception("Ошибка в приложении чата")
