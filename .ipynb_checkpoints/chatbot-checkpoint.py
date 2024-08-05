import os
# import bs4
# from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import toml

def init():
    if 'rag_chain' in st.session_state: return
    secrets = toml.load("streamlit/secrets.toml")

    os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_API_KEY"] = secrets["LANGCHAIN_API_KEY"]


    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    print("preparing Doc.")
    # loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
    loader = PyPDFLoader("./sodapdf-converted.pdf", extract_images=False)
    docs = loader.load()
    print("Doc prep done.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    # print(splits[0])
    # print(len(splits))

    vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    # prompt = hub.pull("rlm/rag-prompt")


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    from langchain_core.prompts import PromptTemplate

    # template = """Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Use three sentences maximum and keep the answer as concise as possible.
    # Always say "thanks for asking!" at the end of the answer.

    # {context}

    # Question: {question}

    # Helpful Answer:"""
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, answer it from your knowledge.
    Answer everything in hindi. also answer if question is asked in another language.
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

# print("\n\n: >>>>>>\n")
# while True:
#     q = input("Enter question: ")
#     ans = rag_chain.invoke(q)
#     print("Answer: ",ans, "\n")

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