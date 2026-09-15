import streamlit as st
import joblib
import pandas as pd
import zipfile
import os

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.title("🚢 Предсказание выживаемости на «Титанике»")
st.write("Введите данные пассажира, чтобы узнать вероятность его выживания.")

@st.cache_resource
def load_model():
    model_filename = "best_random_forest_model.joblib"
    
    # 1. Если файл уже лежит в корне — загружаем
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
        
    # 2. Если файл лежит внутри вложенной папки model/
    nested_path = os.path.join("model", model_filename)
    if os.path.exists(nested_path):
        return joblib.load(nested_path)
        
    # 3. Если ничего не найдено, распаковываем ZIP-архив
    zip_path = "model.zip"
    if not os.path.exists(zip_path) and os.path.exists("Machine learning Model.zip"):
        zip_path = "Machine learning Model.zip"
        
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
            
        # Проверяем снова после распаковки
        if os.path.exists(model_filename):
            return joblib.load(model_filename)
        elif os.path.exists(nested_path):
            return joblib.load(nested_path)
            
    raise FileNotFoundError("Модель не найдена ни в корне, ни внутри архива!")

# Инициализация
model = None
try:
    model = load_model()
    st.success("Модель успешно загружена!")
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")

st.header("Информация о пассажире")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Класс билета (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Пол (Sex)", ["male", "female"])
    age = st.number_input("Возраст (Age)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

with col2:
    sibsp = st.number_input("Братья/сестры/супруг (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Родители/дети (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Стоимость билета (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Порт посадки (Embarked)", ["S", "C", "Q"])

input_data = {
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Sex_male": 1 if sex == "male" else 0,
    "Embarked_Q": 1 if embarked == "Q" else 0,
    "Embarked_S": 1 if embarked == "S" else 0,
}

input_df = pd.DataFrame([input_data])

st.markdown("---")

if st.button("Сделать предсказание 🚀", use_container_width=True):
    if model is None:
        st.error("Ошибка: модель не была загружена.")
    else:
        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
            
            if prediction == 1:
                st.success("🎉 Результат: Пассажир выжил!")
                if probabilities is not None:
                    st.write(f"Вероятность выживания: **{probabilities[1] * 100:.1f}%**")
            else:
                st.error("☠️ Результат: Пассажир не выжил.")
                if probabilities is not None:
                    st.write(f"Вероятность выживания: **{probabilities[1] * 100:.1f}%**")
        except Exception as err:
            st.error(f"Ошибка при расчете предсказания: {err}")
