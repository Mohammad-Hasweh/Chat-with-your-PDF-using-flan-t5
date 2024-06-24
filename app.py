import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_template import css, bot_template, user_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vector_store):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Define a simple function to handle conversation with the model
    def local_model_conversation(input_text):
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(**inputs)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response_text

    conversation_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        memory=memory,
        generate_response=local_model_conversation
    )
    return conversation_chain


def handle_user_input(user_input):
    if st.session_state.conversation:
        with st.spinner("Generating response..."):
            response = st.session_state.conversation({'question': user_input})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please process a PDF first.")


def main():
    load_dotenv()

    st.set_page_config(page_title="PDF Chat")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.subheader('Chat with your PDF')
    user_question = st.text_input("Ask any question about your PDF document")
    if st.button("Send", key="send_button"):
        if user_question:
            handle_user_input(user_question)

    with st.sidebar:
        st.header("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
