import streamlit as st
import os

from langchain_openai.chat_models import ChatOpenAI  # Correct LLM model import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS  # Correct FAISS import
from langchain.document_loaders import PyPDFDirectoryLoader  # Correct document loader
from langchain.embeddings import OpenAIEmbeddings  # Use OpenAI Embeddings for vector
from dotenv import load_dotenv
import time

load_dotenv()

# Load OpenAI API key from .env
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set up the Streamlit interface
st.title("AI Document Q&A")

# Set up the language model (GPT-4 or GPT-3.5)
llm = ChatOpenAI(
    api_key=openai_api_key,
    model_name="gpt-4",  # or use "gpt-3.5-turbo"
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Using OpenAI's embeddings
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Load PDF directory
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk splitting
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Split the docs
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Create vector store

# Input box for user questions
prompt1 = st.text_input("Enter your question from documents")

# Button to trigger vector embedding
if st.button("Create Document Embeddings"):
    vector_embedding()
    st.write("Vector Store DB is ready.")

# If user has entered a question, process it
if prompt1:
    # Create the document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start
    print("Response time:", response_time)
    
    # Display the answer
    st.write(response['answer'])
    
    # Display relevant document context with expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
