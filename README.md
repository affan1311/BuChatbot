<<<<<<< HEAD
# BuChatbot
build bahria university chat bot using there student handbook pdf with python
=======
# Chat with PDF - Interactive AI Chatbot

Chat with PDF is an interactive AI chatbot built with Streamlit, designed to allow you to engage with the content of your PDF documents. Powered by Groq's API and Google’s generative AI embeddings, it enables seamless querying of your uploaded PDFs.

![image](https://github.com/user-attachments/assets/940365bd-156f-49f1-8e10-048302a8a04a)

## Features

- **PDF Interaction**: Upload and query PDF documents directly.
- **Content-Based Responses**: The chatbot provides responses based on the content of your PDFs.
- **User-Friendly Interface**: Simple and intuitive UI built with Streamlit.
- **Customizable API Keys**: Enter your Groq and Google API keys for personalized interactions.

## Prerequisites

- **Python**: Version 3.12 or higher is required.
- **API Keys Required**:
    - Get your Groq API key from [Groq API Key Page](https://console.groq.com/keys).
    - Get your Google API key from [Google API Key Page](https://aistudio.google.com/app/apikey).

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Divyanshu9822/chat-with-pdf.git
    cd chat-with-pdf
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Enter your API keys** in the sidebar:
    - **Groq API Key**: Input your Groq API key.
    - **Google API Key**: Input your Google API key.

3. **Upload PDF(s)** using the file uploader.

4. **Process the documents** by clicking the 'Process Documents' button.

5. **Start querying** the chatbot about the content of the uploaded PDFs.

6. **View responses** based on the content of the documents. Relevant sections from the PDFs will be displayed to support the answers.
>>>>>>> 9b536c4 (first commit)
