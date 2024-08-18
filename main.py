
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from amazoncaptcha import AmazonCaptcha
from bs4 import BeautifulSoup
import requests
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain_groq import ChatGroq
import time
from langchain_core.messages import HumanMessage, AIMessage


# Template for the conversation
system_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer."""

message = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template('{input}'),
] 

PROMPT = ChatPromptTemplate(message)

def detect_and_solve_captcha(driver):
    try:
        captcha_image_element = driver.find_element(By.XPATH, "//img[contains(@src, 'captcha')]")
        captcha_image_url = captcha_image_element.get_attribute('src')
        if captcha_image_url:
            captcha = AmazonCaptcha.fromlink(captcha_image_url)
            captcha_value = AmazonCaptcha.solve(captcha)

            input_field = driver.find_element(By.ID, "captchacharacters")
            input_field.send_keys(captcha_value)

            button = driver.find_element(By.CLASS_NAME, "a-button-text")
            button.click()
            return True
        
    except Exception as e:
        st.error(f"Error detecting or solving CAPTCHA: {e}")
    return False

# Function to process URLs
def Process_urls(url, chrome_driver_path=None):
    try:
        if chrome_driver_path:
            options = Options()
            options.add_argument("--headless")  # Optional: Run in headless mode
            service = Service(chrome_driver_path)
            driver = webdriver.Chrome(service=service, options=options)

            driver.get(url)

            # Check if CAPTCHA is present and solve it
            if detect_and_solve_captcha(driver):
                st.info("CAPTCHA detected and solved. Reloading page...")
                time.sleep(5)  # Wait for the page to reload
                driver.get(url)
            
            # Continue fetching page content if CAPTCHA solved or not present
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            captcha_image = soup.find('img', src=True, alt='Captcha')
            if captcha_image:
                st.error("CAPTCHA detected. Please solve it manually.")
                return None

            data = soup.get_text(separator='\n')
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=200)
            docs = text_splitter.split_text(data)
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_texts(docs, embedding_model)
            return db
        else:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            data = soup.get_text(separator='\n')
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=200)
            docs = text_splitter.split_text(data)
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_texts(docs, embedding_model)
            return db
    except requests.RequestException as e:
        st.error(f"Error fetching data from website: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return None

# Function to process user query
def process_user_query(question, vector_db, chat_history):
    # LLM
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.1-70b-versatile"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Conversation Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        verbose=True,
        return_source_documents=True,
        chain_type='stuff'
    )

    formatted_chat_history = [(item['question'], item['answer']) for item in chat_history]

    # Get response from the conversation chain
    response = chain({"question": question, "chat_history": formatted_chat_history})
    return response

# Main function for Streamlit app
def main():
    st.title("🦜🔗 Chat With Websites")
    st.header("ASK QUESTIONS")
    
    # Sidebar for URL configuration
    with st.sidebar:
        st.header("Configure URL")
        urls = st.text_area("Paste URL:")
        chrome_driver_path = st.text_input("Path to ChromeDriver (optional)", "")

        if st.button("Submit URL"):
            vector_db = Process_urls(urls, chrome_driver_path)
            if vector_db:
                st.session_state['vector_db'] = vector_db
                st.success("URL processed and data embedded successfully.")
                st.session_state['urls_processed'] = True

    # Ensure 'vector_db' exists in session state
    if 'vector_db' not in st.session_state:
        st.session_state['vector_db'] = None

    # Create a container for chat messages
    chat_container = st.container()

    # Display the chat messages
    with chat_container:
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Displaying chat history
        for chat in st.session_state['chat_history']:
            st.chat_message("user").write(chat["question"])
            st.chat_message("ai").write(chat["answer"])

    # User query input fixed at the bottom
    col1, col2 = st.columns([8, 1])
    with col1:
        question = st.text_input("Ask a question:", key="user_query")
    with col2:
        if st.button("Send"):
            if question and st.session_state['vector_db']:
                chat_history = st.session_state.get("chat_history", [])
                
                # Function call
                response = process_user_query(question, st.session_state['vector_db'], chat_history)

                # Update chat history
                chat_history.append({"question": question, "answer": response['answer']})
                st.session_state['chat_history'] = chat_history

                # Display the response in the chat container
                with chat_container:
                    st.chat_message("user").write(question)
                    st.chat_message("ai").write(response['answer'])

                # Scroll to the bottom of the chat container
                st.markdown(f'<div id="bottom"></div>', unsafe_allow_html=True)
                st.markdown('<script>document.getElementById("bottom").scrollIntoView();</script>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


