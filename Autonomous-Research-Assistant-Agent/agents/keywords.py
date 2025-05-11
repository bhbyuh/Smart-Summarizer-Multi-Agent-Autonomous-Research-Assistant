import os
from dotenv import load_dotenv
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

class State(TypedDict):
    original_word: str
    key_words:None
    article_texts:None
    summaries:None
    comparison_results:None

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

class KeyWords(BaseModel):
    key_words: List[str] = Field(description="list of related keywords")

def fetch_keywords(state: State):
    original_word = state["original_word"]

    parser = PydanticOutputParser(pydantic_object=KeyWords)

    prompt = PromptTemplate(
        template="Generated related keywords of this word.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    keywords_list=chain.invoke({"query": original_word})

    state["key_words"]=keywords_list.key_words

    return state