import streamlit as st
import requests

st.title("Prediction UI")

input_ = st.text_input(
    "Try out your new machine learning model in this user interface. Just type some text below."
)

if st.button("Predict!"):
    if input_ is not None:
        # Get request output from the fastapi
        response = requests.post(
            "http://localhost:7531/predict", json={"text": [input_]}
        )
        if response.status_code == 200:
            st.markdown(response.json(), unsafe_allow_html=True)

st.markdown(
    "If you want to improve your model, try adding more high-quality labeled training data. Check out [Kern AI](https://www.kern.ai)"
)
