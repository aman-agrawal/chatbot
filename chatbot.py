import os
import json
import streamlit as st
import toml
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

def init():
    if 'rag_chain' in st.session_state:
        return
    
    # Load secrets and set environment variables
    secrets = toml.load("streamlit/secrets.toml")
    os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_API_KEY"] = secrets["LANGCHAIN_API_KEY"]

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    print("Preparing Doc.")

    # Load JSON file and convert it to a list of documents
    with open("./data.json", "r") as f:
        json_data = json.load(f)
        docs = [Document(page_content=f"{key}: {value}") for key, value in json_data.items()]

    print("Doc prep done.")

    # Split documents and create a vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    retriever = vectorstore.as_retriever()

    # Define formatting function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the prompt template
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, answer it from your knowledge.
    Answer everything in English. Also answer if the question is asked in another language.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks.!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Setup the RAG chain
    st.session_state['rag_chain'] = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

def get_chatbot_response(inp):
    return st.session_state['rag_chain'].invoke(inp)


def display_chat():
    for sender, message in st.session_state['messages']:
        if sender == "You":
            st.write(f"**{sender}:** {message}")
        else:
            st.write(f"**{sender}:** {message}")

def main():
    st.title("Chatbot Interface")
    init()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    user_input = st.text_input("Question: ", "")

    if st.button("Send"):
        if user_input:
            st.session_state['messages'].append(("You", user_input))
            response = get_chatbot_response(user_input)
            st.session_state['messages'].append(("Bot", response))

    display_chat()

if __name__ == "__main__":
    main()
