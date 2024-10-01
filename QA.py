import os
import langchain
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai.chat_models import ChatOpenAI
from transformers import pipeline
# from langchain.llms import HuggingFace

# # model=ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],model=model)
import streamlit as st
st.set_page_config(page_title='QA chatbot')
st.header("chat with me")
st.title("hey there")

from dotenv import load_dotenv
load_dotenv()

chat=ChatOpenAI(temperature=0.5)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="You are a AI assistant")
    ]
def get_openai_response(question):
    # llm=OpenAI(model_name="text-davinci-003",temperature=0.5)
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input=st.text_input("Input: ",key="input")
response=get_openai_response(input)

submit=st.button("Ask the question")

#if ask button is clicked
if submit:
    st.subheader("The Response is")
    st.write(response)
    
    
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# model_name = "deepset/roberta-base-squad2"

# # a) Get predictions
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
# QA_input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
# res = nlp(QA_input)

# # b) Load model & tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# import streamlit as st
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
# import PyPDF2

# # Load model and tokenizer
# model_name = "deepset/roberta-base-squad2"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# # Streamlit UI
# st.title("PDF QA Bot")
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# if uploaded_file is not None:
#     # Read PDF file
#     pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
#     text = ""
#     for page_num in range(pdf_reader.numPages):
#         page = pdf_reader.getPage(page_num)
#         text += page.extract_text()

#     # Display PDF text
#     st.write("PDF Content:")
#     st.write(text)

#     # Question answering
#     question = st.text_input("Ask a question about the PDF:")
#     if question:
#         QA_input = {
#             'question': question,
#             'context': text
#         }
#         res = nlp(QA_input)
#         st.write("Answer:")
#         st.write(res['answer'])
