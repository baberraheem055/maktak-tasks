import os
import pandas as pd
from pandasai import SmartDataframe
# from pandasai.llm.google_gemini import GoogleGemini
from dotenv import load_dotenv
import streamlit as st
import matplotlib
from langchain_groq import ChatGroq

# To select backend for matplotlib
matplotlib.use("TkAgg")

# Load environment variables
load_dotenv()

# Retrieve the API key
API_KEY = os.environ['GROQ_API_KEY']

llm = ChatGroq(
        temperature=0,
        groq_api_key= API_KEY,
        model_name="llama-3.1-70b-versatile"
    )

# Set up the Streamlit app title
st.image("C:\\Users\\Babar Raheem\\Desktop\\logo.png", use_column_width=True)
st.title("Prompt-Driven Analysis with PandasAI")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload CSV file here", type=['csv'])

if uploaded_file:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Initialize SmartDataframe with the data and LLM configuration
    df = SmartDataframe(data, config={"llm": llm})

    # Text area to input the prompt
    Prompt = st.text_area("Ask a question related to the uploaded file")

    # Button to generate the response
    if st.button("Generate"):
        if Prompt:
            with st.spinner("Generating response..."):
                response = df.chat(Prompt)
                # Display response with the logo
                col1, col2 = st.columns([1, 5])
                
                with col1:
                    st.image("C:\\Users\\Babar Raheem\\Desktop\\download.jpg", width=50)

                with col2:
                    st.write(response)
                
        else:
            st.warning("Please enter a prompt")

