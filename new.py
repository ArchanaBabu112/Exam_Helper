# import streamlit as st
# import PyPDF2
# import openai
# from PyPDF2 import PdfReader

# # PDF extraction function
# def extract_pdf_text(pdf_file):
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page_num in range(len(pdf_reader.pages)):
#         text += pdf_reader.pages[page_num].extract_text()
#     return text

# # Question answering function using GPT
# def ask_question_to_gpt(text_chunks, question):
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo",
#         prompt=f"Extract relevant info from the text: {text_chunks} and answer: {question}",
#         max_tokens=200
#     )
#     return response['choices'][0]['text']

# # Streamlit app interface
# st.title("AI Study Bot")

# pdf_file = st.file_uploader("Upload PDF file", type="pdf")
# if pdf_file:
#     pdf_text = extract_pdf_text(pdf_file)
#     st.write("PDF Uploaded. Ask your question below:")
    
#     question = st.text_input("Your Question:")
#     if st.button("Get Answer"):
#         answer = ask_question_to_gpt(pdf_text, question)
#         st.write("Answer: ", answer)
