import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit app setup
st.set_page_config(page_title="Chat with multiple PDFs", layout="wide")
st.header("Chat with PDF")

# Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Convert text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Convert text chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Chain for conversational answers
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "The answer is not available in the context."
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:"""
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User question processing
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search for similar documents
    docs = new_db.similarity_search(user_question)
    
    # Get conversational chain
    chain = get_conversational_chain()
    
    # Generate response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Append the question and response to chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"]})
    
    # Display the current chat history
    for chat in st.session_state.chat_history:
        st.write(f"**Question:** {chat['question']}")
        st.write(f"**Answer:** {chat['answer']}")
        st.write("---") 

def main():
    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents have been processed successfully!")

    # User question input
    user_question = st.text_input("Ask your question from the PDF files")
    
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
