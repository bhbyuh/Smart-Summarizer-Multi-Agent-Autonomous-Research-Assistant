import json

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

from typing import Literal

class SummaryEvaluation(BaseModel):
    fluency_score: Literal[1, 2, 3, 4, 5]
    fluency_justification: str
    factuality_score: Literal[1, 2, 3, 4, 5]
    factuality_justification: str
    coverage_score: Literal[1, 2, 3, 4, 5]
    coverage_justification: str

with open("comparison_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def llm_evaluator(abstracts,fine_summaries):
    evaluations=[]
    prompt = '''
            You are an expert evaluator for scientific summaries. Evaluate the following two generated summaries based on three criteria:

            1. *Fluency*: Is the summary readable and grammatically correct?  
            2. *Factuality*: Are the statements correct and do they accurately reflect the source text?  
            3. *Coverage*: Does the summary include the main problem, method, and key findings?

            *Instructions*:
            - Evaluate **each summary separately** (Base model and Fine-tuned model).
            - For each criterion, assign a score from 1 (poor) to 5 (excellent).
            - Also, provide a short justification (1–2 sentences) for each score.

            Actual summary: {actualsummary}
            Model Summary: {modelsummary}

            - Respond **only** in the following JSON format:
            {format_instructions}
            '''
    parser = PydanticOutputParser(pydantic_object=SummaryEvaluation)

    prompt = PromptTemplate(
        template=prompt,
        input_variables=["PAPER_SUMMARY","CITATION_COUNT","PUBLICATION_DATE","KEYWORDS"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    for abstract,model_summary in zip(abstracts,fine_summaries):
        evaluation_metrix=chain.invoke({"actualsummary":abstract,"modelsummary":model_summary})
        evaluations.append(evaluation_metrix)

    return evaluations

if __name__=="__main__":
    abstracts = [item['abstract'] for item in data]
    fine_summaries = [item['finetuned_summary'] for item in data]

    evaluations=llm_evaluator(abstracts,fine_summaries)

    # Convert Pydantic models to dicts using model_dump()
    evaluations_dict = [e.model_dump() if hasattr(e, "model_dump") else e for e in evaluations]

    # Save to JSON file
    with open("summary_evaluations.json", "w", encoding="utf-8") as f:
        json.dump(evaluations_dict, f, indent=4, ensure_ascii=False)