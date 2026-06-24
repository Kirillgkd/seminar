import os
import json
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🔮 Предсказание")

# Список обученных моделей
try:
    r = requests.get(f"{API_URL}/models/trained")
    if r.status_code == 200:
        results = r.json()["results"]
        model_names = [m["model_name"] for m in results]
    else:
        model_names = []
except requests.exceptions.ConnectionError:
    st.error("Не удалось подключиться к API.")
    model_names = []

if not model_names:
    st.warning("Нет обученных моделей. Сначала обучите модель на странице 'Обучение'.")
    st.stop()

model_name = st.selectbox("Выберите модель", model_names)

st.subheader("Введите данные для предсказания")
st.markdown("Введите признаки в формате JSON:")

# Пример ввода
example = '{"feature1": 1.0, "feature2": 0.5, "feature3": "category_a"}'
input_json = st.text_area("JSON с признаками", value=example, height=100)

# Или загрузка CSV для пакетного предсказания
st.markdown("---")
st.subheader("Или загрузите CSV для пакетного предсказания")
uploaded = st.file_uploader("CSV файл (без целевой переменной)", type=["csv"])

if st.button("🔮 Предсказать", type="primary"):
    if uploaded:
        # Пакетное предсказание
        df = pd.read_csv(uploaded)
        predictions = []
        progress = st.progress(0)
        for i, row in df.iterrows():
            try:
                r = requests.post(
                    f"{API_URL}/predict",
                    json={"model_name": model_name, "input_data": row.to_dict()},
                )
                if r.status_code == 200:
                    predictions.append(r.json()["prediction"][0])
                else:
                    predictions.append(None)
            except Exception:
                predictions.append(None)
            progress.progress((i + 1) / len(df))

        df["prediction"] = predictions
        st.dataframe(df)
        st.download_button(
            "📥 Скачать результат",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )
    else:
        # Одиночное предсказание
        try:
            input_data = json.loads(input_json)
            r = requests.post(
                f"{API_URL}/predict",
                json={"model_name": model_name, "input_data": input_data},
            )
            if r.status_code == 200:
                result = r.json()
                st.success(f"Предсказание: **{result['prediction'][0]}**")
                if result.get("probabilities"):
                    st.write("Вероятности классов:", result["probabilities"][0])
            else:
                st.error(f"Ошибка: {r.json().get('detail', r.text)}")
        except json.JSONDecodeError:
            st.error("Некорректный JSON. Проверьте формат ввода.")
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API.")
