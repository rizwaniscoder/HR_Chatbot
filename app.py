from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import streamlit as st
import toml
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from googlesearch import search
from bs4 import BeautifulSoup
import requests



secrets = toml.load("secrets.toml")
api_key = secrets["OPENAI_API_KEY"]


prompt_template = PromptTemplate.from_template(
    "Based on the information from the Department of Labor and additional Google search results, answer the following question: {question}\n\nInformation:\n{dol_info}\n\nAnswer:"
)


def search_dol(query):
    search_results = search(f"site:dol.gov {query}", num=3, stop=3, pause=0)
    results_text = ""
    for result in search_results:

        try:
            response = requests.get(result)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            results_text += "\n".join([para.get_text() for para in paragraphs])
        except Exception as e:
            results_text += f"Error fetching {result}: {str(e)}\n"
    return results_text

def summarize_text(results_text):
    
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = char_text_splitter.split_documents([Document(page_content=results_text)])
    llm = OpenAI(temperature=0.5, openai_api_key=api_key)
    model = load_summarize_chain(llm=llm, chain_type='refine')
    summary = model.run(documents)
    return summary

def get_response(question, dol_info):
    llm = OpenAI(temperature=0.3, openai_api_key=api_key)
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
    )

    response = chain.run(question=question, dol_info=dol_info)
    return response


def main():
    st.set_page_config(page_title="HR Compliance Chatbot", layout="wide")

    st.title("HR Compliance Advisor")
    st.subheader("Interact with the Bot for Reliable Compliance Answers")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    with st.sidebar:
        st.markdown("""
            <style>
            .css-1d391kg {
                background-color: #f0f2f6;
                color: #333;
            }
            .css-1y0t0cm {
                background-color: #4a90e2;
                color: white;
            }
            .css-1nb8a2x {
                color: #4a90e2;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("### Information")
        st.markdown("Ask your HR compliance questions and get answers from the latest Department of Labor information and Google search.")

        if st.button("New Chat"):
            st.session_state.chat_history = []


    for chat in st.session_state.chat_history:
        with st.chat_message("assistant" if chat["role"] == "assistant" else "user"):
            st.markdown(chat["content"])


    if user_question := st.chat_input("Ask a compliance question:"):

        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)


        dol_info = search_dol(user_question)
        dol_info_summary = summarize_text(dol_info)
        
        response = get_response(user_question, dol_info_summary)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
