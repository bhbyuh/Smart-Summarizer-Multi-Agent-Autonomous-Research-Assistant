import os
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

class Score(BaseModel):
    score:int

class State(TypedDict):
    original_word: str
    key_words:None
    article_texts:None
    summaries:None
    comparison_results:None

def rank_papers(state:State):
    scores = []

    keywords = state["key_words"]
    papers= state["article_texts"]

    for paper in papers:
        prompt = '''
            You are a research paper ranking assistant.

            Given the following details of a paper:
            - Summary: {PAPER_SUMMARY}
            - Citation Count: {CITATION_COUNT}
            - Publication Date: {PUBLICATION_DATE}
            - Target Keywords: {KEYWORDS}

            Evaluate and return a **single relevance score from 0 to 10** based on:
            1. How recent the publication is.
            2. How many citations it has received.
            3. How well the summary aligns with the provided keywords.

            Respond only with the numerical score.
            {format_instructions}
            '''
        parser = PydanticOutputParser(pydantic_object=Score)

        prompt = PromptTemplate(
            template=prompt,
            input_variables=["PAPER_SUMMARY","CITATION_COUNT","PUBLICATION_DATE","KEYWORDS"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        score=chain.invoke({"PAPER_SUMMARY":paper[1],"CITATION_COUNT":paper[2],"PUBLICATION_DATE":paper[3],"KEYWORDS":keywords})
        scores.append(score.score)

    combined = sorted(zip(scores, papers), reverse=True)
    sorted_score, sorted_papers = zip(*combined)

    scores = list(sorted_score)
    papers = list(sorted_papers)

    print(scores)

    state["article_texts"] = papers[:3]