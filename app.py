#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/dream.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Attrition Forecast",
        page_icon=image,

    )

    st.write(
        """
        # Классификация сотрудников 
        Определяем, кто из сотрудников уволится, а кто нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)

# del prediction_probas
def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)
    # del prediction_probas
    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    MonthlyIncome = st.sidebar.slider("Ежемесячный доход, USD", min_value=1000, max_value=20000, value=1200,step=100)
    OverTime = st.sidebar.selectbox("Перерабатывает ли сотрудник", ("Да", "Нет"))   
    EnvironmentSatisfaction = st.sidebar.slider("Удовлетворённость условиями труда: \n1 'Низкая', 2: 'Средняя', 3: 'Высокая', 4: 'Очень высокая'",
                                                min_value=1, max_value=4, value=1, step=1)
    TotalWorkingYears = st.sidebar.slider("Общий трудовой стаж",min_value=0, max_value=40, value=0, step=1)
    NumCompaniesWorked = st.sidebar.slider("Кол-во компаний в которых работал сотрудник",min_value=0, max_value=9, value=0, step=1)

    translatetion = {
        "Да": "Yes",
        "Нет": "No"    
    }

    data = {
        "MonthlyIncome": MonthlyIncome,
        "OverTime": translatetion[OverTime],
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "TotalWorkingYears": TotalWorkingYears,
        "NumCompaniesWorked": NumCompaniesWorked
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()


