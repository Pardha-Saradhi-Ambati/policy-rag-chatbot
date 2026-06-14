import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 Policy RAG Chatbot")

question = st.text_input("Ask a question:")

if question:
    try:
        response = ask_bot(question)

        if hasattr(response, "content"):
            st.write(response.content)
        else:
            st.write(response)

    except Exception as e:
        st.error(f"Error: {e}")