import streamlit as st
import pandas as pd
import joblib


st.header('WHETHER NEWS IS FAKE OR NOT')


# data = pd.read_csv('data.csv')
#
#
# button_imp = st.button('click to see dataframes')
# if button_imp:
#     st.dataframe(data.head())


user_input = st.text_area('Enter your News Here')

col1, col2, col3 = st.columns(3)
with col1:
    pass

with col2:
    button_pressed = st.button('Click to find ANSWER')

with col3:
    pass

if button_pressed:
    model = joblib.load('Fake_news_detection_model.pkl')
    tf_idf = joblib.load('TF-idf-news vectorizer.pkl')
    user_input = tf_idf.transform([user_input, ])
    news = model.predict(user_input)
    st.write(f'NEWS  State: { news[0]}')

   # news = [fake or not]
    if (news == 0):
        st.write('Fake News')
    else: #(news == 1):
        st.write('Not A Fake News')


