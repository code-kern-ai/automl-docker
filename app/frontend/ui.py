import streamlit as st
import requests

# Print kern image and a header
st.title("Prediction UI")

inpt = st.text_input("Input your text below!")

if st.button("Predict!"):
    if inpt is not None:
        # Get request output from the fastapi
        res = requests.post("http://localhost:7531/predict", json={"text": [inpt]})

        # Process the request content
        res_list = [c for c in res.text]
        res_list.remove("[")
        res_list.remove("]")
        res_conv = ["Clickbait." if x == "1" else "Not clickbait" for x in res_list]

        if res_conv[0] == "Clickbait.":
            text = '<p style="font-family:monospace; color:#EC7063; font-size: 18px;">Clickbait.</p>'
        else:
            text = '<p style="font-family:monospace; color:#5DADE2; font-size: 18px";">Not Clickbait.</p>'

        # Print out the results
        st.markdown(text, unsafe_allow_html=True)

    else:
        st.write("Please provide texts first!")
