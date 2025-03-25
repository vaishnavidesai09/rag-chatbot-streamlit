import streamlit as st
import os
from langchain_groq  import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()
st.title("RAG CHATBOT")

#setup a session state to hold the all the old messages 
if 'messages' not in st.session_state:
    st.session_state.messages = []

#display all historical messges
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vector_store()    :
    pdf_name = "main.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    #create chunks
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name = 'all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    ).from_loaders(loaders)

    return index.vectorstore


prompt = st.chat_input("pass your input")

if prompt:
    st.chat_message('user').markdown(prompt)

    st.session_state.messages.append({'role':'user','content':prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please""")


    model = 'llama3-8b-8192'

    groq_chat = ChatGroq(
        groq_api_key = os.getenv('GROK_API_KEY'),
        model_name =  model
    )
    
    try:
        vectorstore = get_vector_store()
        if vectorstore is None:
            st.error("Failed to load document")
      
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True)
       
        result = chain({"query": prompt})
        response = result["result"]  # Extract just the answer
        #response = get_response_from_groq(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    except Exception as e:
        st.error(f"Error: {str(e)}")