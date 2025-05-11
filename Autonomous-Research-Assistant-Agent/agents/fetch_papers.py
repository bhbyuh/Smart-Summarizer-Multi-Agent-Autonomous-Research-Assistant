import arxiv
import requests
import fitz  # PyMuPDF

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

def fetch_arxiv_papers(state:State):
    article_texts = []
    print(state)
    keywords = state["key_words"]

    for words in keywords:
        print(words)
        search = arxiv.Search(
                query=words,
                max_results=1,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
        for result in search.results():
            continue
        
        response = requests.get(result.pdf_url)
        if response.status_code == 200:
            pdf_bytes = response.content
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text = ""
            for page in doc:
                text += page.get_text()
            
            article_texts.append((text,result.summary,result.journal_ref,result.published,result.title,result.authors))
        else:
            print(f"Failed to fetch PDF for {result.title}")

    unique_articles = []
    seen_titles = set()

    for article in article_texts:
        title = article[4]  # assuming title is at index 4
        if title not in seen_titles:
            unique_articles.append(article)
            seen_titles.add(title)

    article_texts = unique_articles

    state["article_texts"] = article_texts

    return state