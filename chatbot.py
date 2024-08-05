import os
import json
import streamlit as st
import toml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
# from langchain.vectorstores import Chroma

# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI



# Hardcoded admin credentials for simplicity
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"

def init():
    if 'rag_chain' in st.session_state: return

    # secrets = toml.load("streamlit/secrets.toml")
    # os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
    # os.environ["LANGCHAIN_TRACING_V2"] = secrets["LANGCHAIN_TRACING_V2"]
    # os.environ["LANGCHAIN_API_KEY"] = secrets["LANGCHAIN_API_KEY"]
    # # os.environ["OPENAI_API_KEY"] = st.secrets["secrets"]["OPENAI_API_KEY"]
    # os.environ['PINECONE_API_KEY'] = secrets["PINECONE_API_KEY"]

    os.environ["GOOGLE_API_KEY"] = st.secrets["secrets"]["GOOGLE_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["secrets"]["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["secrets"]["LANGCHAIN_API_KEY"]
    os.environ["abc"] = st.secrets["secrets"]["OPENAI_API_KEY"]
    os.environ['PINECONE_API_KEY'] = st.secrets["secrets"]["PINECONE_API_KEY"]

    # llm = ChatGoogleGenerativeAI(model="gemini-pro")
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key = os.environ['abc'])

    print("Preparing Doc...")

    # Load JSON file and convert it to a list of documents
    with open("./data.json", "r") as f:
        json_data = json.load(f)
        docs = [Document(page_content=f"{key}: {value}") for key, value in json_data.items()]

    print("Doc prep done...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ['abc'])
    print(embeddings)

    index_name = "streamlit"

    vectorstore = PineconeVectorStore.from_documents(documents=splits, index_name=index_name, embedding=embeddings)
    # vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY']))
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY']))
    # vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, answer it from your knowledge.
    Answer everything in English. Also answer if the question is asked in another language.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks.!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    st.session_state['rag_chain'] = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

def authenticate_user(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def add_question_to_json(question, answer):
    with open("./data.json", "r") as f:
        data = json.load(f)

    data[question] = answer

    with open("./data.json", "w") as f:
        json.dump(data, f, indent=4)

def logout():
    st.session_state['authenticated'] = False
    st.rerun()

def get_chatbot_response(inp):
    if 'rag_chain' not in st.session_state:
        init()  # Initialize if not already done
    return st.session_state['rag_chain'].invoke(inp)

def admin_panel():
    st.subheader("Admin Panel")

    # Admin actions
    new_question = st.text_input("New Question:")
    new_answer = st.text_input("Answer:")
    
    if st.button("Add Question"):
        if new_question and new_answer:
            add_question_to_json(new_question, new_answer)
            st.success("Question and answer added successfully.")
        else:
            st.error("Please fill both fields.")

    # Logout button
    if st.button("Logout"):
        logout()

# Function to handle chat panel
def chat_panel():
    st.subheader("Chat")

    user_input = st.text_input("Question: ", "")

    if st.button("Send"):
        if user_input:
            response = get_chatbot_response(user_input)
            st.session_state['messages'].append(("Bot", response))
            st.session_state['messages'].append(("You", user_input))

    for sender, message in reversed(st.session_state['messages']):
        if sender == "You":
            st.write(f"**{sender}:** {message}")
        else:
            st.write(f"**{sender}:** {message}")

# Function to handle login
def login_panel():
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state['authenticated'] = True
            st.success("Logged in successfully.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# Main Function
def main():
    st.title("Chatbot Interface")

    # Initialize session state if necessary
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    init()  # Ensure that the necessary initialization is done

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.session_state.get('authenticated', False):
            admin_panel()
        else:
            login_panel()

    with col2:
        chat_panel()

if __name__ == "__main__":
    main()
