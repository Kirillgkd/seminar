import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🧠 AI-интерпретация результатов")

st.markdown("""
Получите понятное объяснение результатов обучения модели с помощью ИИ.

Провайдер выбирается по приоритету:
1. **Anthropic (Claude)** — если задан `ANTHROPIC_API_KEY`
2. **OpenAI-совместимый** — если задан `OPENAI_API_KEY`
3. **Офлайн** — встроенная интерпретация на основе правил (без ключа)
""")

# Список обученных моделей
try:
    r = requests.get(f"{API_URL}/models/trained")
    model_names = [m["model_name"] for m in r.json()["results"]] if r.status_code == 200 else []
except requests.exceptions.ConnectionError:
    st.error("Не удалось подключиться к API.")
    model_names = []

if not model_names:
    st.warning("Нет обученных моделей. Сначала обучите модель.")
    st.stop()

model_name = st.selectbox("Выберите модель", model_names)

if st.button("🧠 Интерпретировать", type="primary"):
    with st.spinner("Анализ результатов..."):
        try:
            r = requests.post(f"{API_URL}/interpret", json={"model_name": model_name})
            if r.status_code == 200:
                result = r.json()
                source_labels = {
                    "anthropic": "🤖 Anthropic (Claude)",
                    "openai": "🤖 OpenAI",
                    "rule_based": "📐 Правила (офлайн)",
                    "rule_based_fallback": "📐 Правила (fallback после ошибки LLM)",
                }
                st.caption(f"Источник: {source_labels.get(result.get('source'), result.get('source'))}")
                st.markdown(result["interpretation"])
                if result.get("error"):
                    st.warning(f"LLM недоступен: {result['error']}")
            else:
                st.error(f"Ошибка: {r.json().get('detail', r.text)}")
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API.")
