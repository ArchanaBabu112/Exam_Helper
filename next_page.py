
import streamlit as st
import os
import openai
# Check if navigation state is set
if st.session_state.get("next_page"):
    # Reset navigation state
    st.session_state.next_page = False

    # Title for the next page
    st.title("Answer to the question vocally or typed in the given box below")

    # Input box for user to type their answer
    answer = st.text_area("Type your answer here:")

    # Display the answer (for demonstration purposes)
    if answer:
        st.write("Your answer:")
        st.write(answer)
else:
    st.write("Please proceed from the main page.")



#####

openai.api_key = os.getenv("OPENAI_API_KEY")

