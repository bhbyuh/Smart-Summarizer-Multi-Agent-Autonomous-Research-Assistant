import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain.prompts import ChatPromptTemplate

class State(TypedDict):
    original_word: str
    key_words:None
    article_texts:None
    summaries:None
    comparison_results:None

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

def compare_summaries(state: State):
    summaries = state["summaries"]
    word = state["original_word"]

    prompt = """
        Below is a list of research paper summaries related to the keyword provided. 
        Please compare the research papers by identifying shared findings, conflicting results, and areas that require further research.

        Keyword: {word}
        Summaries: {papers_summary}
        """

    prompt = ChatPromptTemplate.from_template(prompt)
        
    chain = prompt | llm | StrOutputParser

    comparison_results=chain.invoke({"word":word,"papers_summary": summaries})

    state["comparison_results"]=comparison_results

    return state