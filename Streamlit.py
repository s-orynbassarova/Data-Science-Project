#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pickle
import requests

# Загрузка модели и масштабатора из GitHub
xgb_model_url = "https://raw.githubusercontent.com/s-orynbassarova/data-science-project/main/xgb_model.pkl"
scaler_url = "https://raw.githubusercontent.com/s-orynbassarova/data-science-project/main/scaler.pkl"

# Скачать модели с GitHub
xgb_model_content = requests.get(xgb_model_url).content
scaler_content = requests.get(scaler_url).content

# Загрузить модели из памяти
xgb = pickle.loads(xgb_model_content)
scaler = pickle.loads(scaler_content)

# Интерфейс Streamlit
st.title("Программа для прогнозирования качества воды")

st.markdown("Введите значения химико-биологических показателей воды:")

# Ввод 20 признаков
aluminum = st.number_input("Алюминий", 0.0, 1000.0, step=0.01)
ammonia = st.number_input("Аммиак", 0.0, 1000.0, step=0.01)
arsenic = st.number_input("Мышьяк", 0.0, 1000.0, step=0.01)
barium = st.number_input("Барий", 0.0, 1000.0, step=0.01)
cadmium = st.number_input("Кадмий", 0.0, 1000.0, step=0.01)
chloramine = st.number_input("Хлорамин", 0.0, 1000.0, step=0.01)
chromium = st.number_input("Хром", 0.0, 1000.0, step=0.01)
copper = st.number_input("Медь", 0.0, 1000.0, step=0.01)
fluoride = st.number_input("Фторид", 0.0, 1000.0, step=0.01)
bacteria = st.number_input("Бактерии", 0.0, 1000.0, step=0.01)
viruses = st.number_input("Вирусы", 0.0, 1000.0, step=0.01)
lead = st.number_input("Свинец", 0.0, 1000.0, step=0.01)
nitrates = st.number_input("Нитраты", 0.0, 1000.0, step=0.01)
nitrites = st.number_input("Нитриты", 0.0, 1000.0, step=0.01)
mercury = st.number_input("Ртуть", 0.0, 1000.0, step=0.01)
perchlorate = st.number_input("Перхлорат", 0.0, 1000.0, step=0.01)
radium = st.number_input("Радий", 0.0, 1000.0, step=0.01)
selenium = st.number_input("Селен", 0.0, 1000.0, step=0.01)
silver = st.number_input("Серебро", 0.0, 1000.0, step=0.01)
uranium = st.number_input("Уран", 0.0, 1000.0, step=0.01)

# Кнопка предсказания
if st.button("Прогноз качества воды"):
    input_data = np.array([[aluminum, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, fluoride, 
                            bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, 
                            silver, uranium]])
    input_scaled = scaler.transform(input_data)
    result = xgb.predict(input_scaled)

    if result[0] == 1:
        st.success("✅ Безопасная вода")
    else:
        st.error("⚠️ Небезопасная вода")

