import streamlit as st
import time # Optional: To simulate bot thinking time
from rag import get_llm_response, generate_vectorstore
import os
# --- Configuration ---
st.set_page_config(page_title="My Basic Chatbot", layout="centered")
st.title("🤖 Simple Chatbot Interface")
uploaded_pdf_file = st.sidebar.file_uploader("Upload the pdf file", type="pdf", label_visibility="collapsed")

# if uploaded_file is None and st.session_state.count ==0:
#     st.write("Please upload the file to see the final results.\n")
if 'count' not in st.session_state:
	st.session_state.count = 0

if uploaded_pdf_file and st.session_state.count < 1:
    with st.spinner("Processing..."):
        DIR_NAME = "input_data"
        if not os.path.exists(DIR_NAME):
            os.makedirs(DIR_NAME)

        pdf_file_path = os.path.join(DIR_NAME, uploaded_pdf_file.name)

        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_pdf_file.getbuffer())

        generate_vectorstore(pdf_file_path)
# --- Placeholder for Chatbot Logic ---
# Replace this with your actual chatbot function/API call
def get_bot_response(user_message):
    """
    Simulates a bot response.
    Replace this with your actual chatbot logic (e.g., API call, model inference).
    """
    # Simulate thinking time (optional)
    # time.sleep(0.5)

    # Basic echo response for demonstration
    # return f"You said: '{user_message}'"

    # Slightly more interactive placeholder
    if "hello" in user_message.lower():
        return "Hello there! How can I help you today?"
    elif "how are you" in user_message.lower():
        return "I'm just a bunch of code, but I'm running smoothly! Thanks for asking."
    elif "bye" in user_message.lower():
        return "Goodbye! Have a great day."
    else:
        return get_llm_response(user_message)

# --- Session State Initialization ---
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optional: Add a starting welcome message from the assistant
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hi! Ask me anything."}
    )


# --- Display Chat History ---
st.write("--- Conversation History ---")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # Use markdown for potential formatting
st.write("---") # Separator


# --- Handle User Input ---
# Use st.chat_input which is designed specifically for chat interfaces
prompt = st.chat_input("What is up?")

if prompt:
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get bot response (using placeholder function)
    with st.spinner("Thinking..."): # Show a thinking indicator
        bot_response = get_bot_response(prompt)

    # 4. Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # 5. Display assistant response in chat message container
    # Use st.rerun() for a smoother update after adding messages
    st.rerun()


# --- Optional: Add a button to clear history ---
if st.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. How can I help?"}]
    st.rerun()
