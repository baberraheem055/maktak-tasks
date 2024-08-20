# maktak-tasks

# Streamlit Application with PandasAI
## Introduction
PandasAI is a powerful tool designed to enhance data analysis workflows by integrating large language models (LLMs) with the Pandas library. It allows users to interact with data using natural language queries, making complex data manipulations and analyses more accessible. By combining PandasAI with Streamlit, we can create an intuitive web application that facilitates prompt-driven data analysis.

# Why Use PandasAI
PandasAI is beneficial for:

Simplified Data Interaction: Enables users to perform data queries and manipulations using natural language, reducing the need for complex code.
Increased Efficiency: Automates data analysis tasks, making it quicker and easier to obtain insights.
Enhanced Accessibility: Makes data analysis tools available to users without extensive programming knowledge.
How to Use the Application
Start the Application: Run the Streamlit application script to launch a local server.
Upload Data: Use the file uploader to upload a CSV file.
Enter a Query: Type a natural language question or command related to the data into the provided text area.
Generate and View Results: Click the "Generate" button to receive a response based on the input, displayed alongside relevant visual elements.
Dependencies
The application relies on several key libraries and tools, which are listed in the requirements.txt file. Typical dependencies include:

streamlit – For building the interactive web application.
pandas – For data manipulation and analysis.
pandasai – For natural language processing capabilities.
langchain_groq – For integrating with the Groq LLM.
python-dotenv – For managing environment variables.
matplotlib – For plotting, though not directly used in this code.

Code Explanation
Initialization
python
Copy code
import os
import pandas as pd
from pandasai import SmartDataframe
from dotenv import load_dotenv
import streamlit as st
import matplotlib
from langchain_groq import ChatGroq

matplotlib.use("TkAgg")
load_dotenv()
Imports: Includes necessary libraries and sets up environment variable management.
Matplotlib Backend: Sets the backend for plotting (though not used directly in this code).
Language Model Setup
python
Copy code
llm = ChatGroq(
        temperature=0,
        groq_api_key="your_groq_api_key",
        model_name="llama-3.1-70b-versatile"
    )
LLM Initialization: Initializes the ChatGroq language model with the specified API key and model configuration. This model processes natural language queries and generates corresponding Pandas operations.
Streamlit Interface
python
Copy code
st.image("path_to_logo.png", use_column_width=True)
st.title("Prompt-Driven Analysis with PandasAI")
App Title and Logo: Displays a logo image and sets the application title.
Data Upload and Processing
python
Copy code
uploaded_file = st.file_uploader("Upload CSV file here", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    df = SmartDataframe(data, config={"llm": llm})
File Upload: Allows users to upload a CSV file.
Data Handling: Reads the file into a Pandas DataFrame and initializes SmartDataframe with the data and LLM configuration.
User Input and Response Generation
python
Copy code
Prompt = st.text_area("Ask a question related to the uploaded file")

if st.button("Generate"):
    if Prompt:
        with st.spinner("Generating response..."):
            response = df.chat(Prompt)
            col1, col2 = st.columns([1, 5])
            
            with col1:
                st.image("path_to_additional_image.jpg", width=50)

            with col2:
                st.write(response)
    else:
        st.warning("Please enter a prompt")
Prompt Input: Users enter queries into a text area.
Response Generation: The application processes the query using the language model and displays the response alongside an additional image.
