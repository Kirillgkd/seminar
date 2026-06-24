import os
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

os.makedirs("data", exist_ok=True)

st.title("📂 Данные")

# Загрузка файла
st.subheader("Загрузка файла")
f = st.file_uploader("Загрузите CSV файл", type=["csv"])
if f:
    with open(os.path.join("data", f.name), "wb") as w:
        w.write(f.getvalue())
    st.success(f"Файл '{f.name}' загружен успешно!")

# Доступные файлы
st.subheader("Доступные файлы")
files = [fn for fn in os.listdir("data") if fn.endswith(".csv")]

if files:
    name = st.selectbox("Выберите файл для просмотра", files)
    if name:
        df = pd.read_csv(os.path.join("data", name))
        st.write(f"**Размер:** {df.shape[0]} строк × {df.shape[1]} столбцов")
        st.write(f"**Целевая переменная (последний столбец):** `{df.columns[-1]}`")
        st.write(f"**Уникальных классов:** {df[df.columns[-1]].nunique()}")
        st.dataframe(df.head(20))

        with st.expander("Статистика"):
            st.write(df.describe())
else:
    st.info("Нет загруженных файлов. Загрузите CSV файл выше.")
