import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

st.title("🏋️ Обучение модели")

# Доступные файлы
files = [fn for fn in os.listdir("data") if fn.endswith(".csv")] if os.path.exists("data") else []

if not files:
    st.warning("Сначала загрузите данные на странице 'Данные'")
    st.stop()

st.subheader("Данные и задача")
col1, col2 = st.columns(2)

with col1:
    filename = st.selectbox("Файл данных", files)
    model_name = st.text_input("Название модели", value="my_model")
    train_size = st.slider("Размер обучающей выборки", 0.1, 0.9, 0.8, 0.05)

with col2:
    task_type = st.selectbox(
        "Тип задачи",
        ["classification", "regression"],
        format_func=lambda x: "Классификация" if x == "classification" else "Регрессия",
    )

    # Колонки для выбора целевой переменной
    columns = []
    try:
        rc = requests.get(f"{API_URL}/data/columns", params={"filename": filename})
        if rc.status_code == 200:
            columns = rc.json()["columns"]
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API.")

    target_column = ""
    if columns:
        target_column = st.selectbox(
            "Целевая переменная",
            columns,
            index=len(columns) - 1,
        )

# Модели под тип задачи
model_options = (
    ["logistic_regression", "random_forest", "gradient_boosting"]
    if task_type == "classification"
    else ["linear_regression", "ridge", "random_forest", "gradient_boosting"]
)

st.subheader("Модель и гиперпараметры")
col3, col4 = st.columns(2)

with col3:
    model_type = st.selectbox(
        "Тип модели", model_options, format_func=lambda x: MODEL_LABELS[x]
    )

    params = {}
    if model_type == "logistic_regression":
        params["max_iter"] = st.number_input("max_iter", 100, 10000, 1000, 100)
        params["C"] = st.number_input("C (regularization)", 0.01, 100.0, 1.0, 0.1)
    elif model_type == "ridge":
        params["alpha"] = st.number_input("alpha", 0.01, 100.0, 1.0, 0.1)
    elif model_type == "linear_regression":
        st.caption("Без настраиваемых гиперпараметров")
    elif model_type == "random_forest":
        params["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100, 10)
        max_depth = st.number_input("max_depth (0 = None)", 0, 100, 0, 1)
        params["max_depth"] = max_depth if max_depth > 0 else None
    elif model_type == "gradient_boosting":
        params["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100, 10)
        params["learning_rate"] = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01)
        params["max_depth"] = st.number_input("max_depth", 1, 20, 3, 1)

with col4:
    st.markdown("**Препроцессинг**")
    scale_features = st.checkbox("Масштабировать признаки (StandardScaler)", value=False)
    fill_strategy = st.selectbox(
        "Заполнение пропусков (числовые)",
        ["mean", "median"],
        format_func=lambda x: "Среднее" if x == "mean" else "Медиана",
    )
    st.caption("Категориальные пропуски заполняются модой, категории кодируются one-hot.")

# Кнопка обучения
st.markdown("---")
if st.button("🚀 Обучить модель", type="primary"):
    with st.spinner("Отправка запроса на обучение..."):
        try:
            r = requests.post(
                f"{API_URL}/train",
                json={
                    "filename": filename,
                    "model_name": model_name,
                    "model_type": model_type,
                    "task_type": task_type,
                    "target_column": target_column,
                    "train_size": train_size,
                    "params": params,
                    "scale_features": scale_features,
                    "fill_strategy": fill_strategy,
                },
            )
            if r.status_code == 200:
                st.success(r.json()["message"])
                st.info("Модель обучается в фоновом режиме. Проверьте результаты на странице 'Модели'.")
            else:
                st.error(f"Ошибка: {r.json().get('detail', r.text)}")
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API. Убедитесь, что бэкенд запущен.")
