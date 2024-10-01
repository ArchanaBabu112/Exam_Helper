import streamlit as st
import openai

# Title
st.title("Your Science Revision Helper")

# Subheading
st.markdown(
    "<h3 style='color: darkRed;'>Revise Before Your Exam</h3>", 
    unsafe_allow_html=True
)

# Dropdown for class number with a unique key
class_number = st.selectbox(
    "Choose your class number:",
    options=[8, 9, 10],
    key="class_number_dropdown"
)

# Dropdown for chapter with a unique key
chapter = st.selectbox(
    "Choose the chapter:",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "all"],
    key="chapter_dropdown"
)

# Display selections
st.write(f"You selected class number: {class_number}")
st.write(f"You selected chapter: {chapter}")

# Navigation button
if st.button("Proceed to Revision"):
    st.session_state.class_number = class_number
    st.session_state.chapter = chapter
    st.session_state.next_page = True
    st.experimental_rerun()

# Define pages and their respective scripts
PAGES = {
    "Main Page": "main.py",
    "Next Page": "next_page.py"
}

# Sidebar for page selection
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="page_selection")

# Load the selected page
page = PAGES[selection]
exec(open(page).read())
