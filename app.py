import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQA

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = Groq(
    model="Llama-3.2-90b-Text-Preview",
    groq_api_key="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"
)

def load_wikipedia(query):
    loader = WikipediaLoader(query)
    documents = loader.load()
    return documents

def create_qa_chain():
    llm = llm
    qa_chain = RetrievalQA.from_llm(llm, documents)
    return qa_chain

st.title("RAG : Study Assistance")

def main():
    query = st.text_input("Enter your query")

    if query:
        documents = load_wikipedia(query)
        qa_chain = create_qa_chain()

        try:
            response = qa_chain.run(query)
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

