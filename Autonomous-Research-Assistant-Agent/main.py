from dotenv import load_dotenv
import os
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START ,END
from langchain_openai import ChatOpenAI
from agents.keywords import fetch_keywords
from agents.fetch_papers import fetch_arxiv_papers
from agents.ranker import rank_papers
from agents.arrticle_summary import paper_summary_generator
from agents.comparison import compare_summaries

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

class State(TypedDict):
    original_word: str
    key_words:None
    article_texts:None
    summaries:None
    comparison_results:None

graph_builder = StateGraph(State)

graph_builder.add_node("keywords", fetch_keywords)
graph_builder.add_node("fetchpapers", fetch_arxiv_papers)
graph_builder.add_node("rankpapers", rank_papers)
graph_builder.add_node("paper_summary_generator", paper_summary_generator)
graph_builder.add_node("compare_summaries", compare_summaries)

graph_builder.add_edge(START, "keywords")
graph_builder.add_edge("keywords", "fetchpapers")
graph_builder.add_edge("fetchpapers", "rankpapers")
graph_builder.add_edge("rankpapers", "paper_summary_generator")
graph_builder.add_edge("paper_summary_generator", "compare_summaries")
graph_builder.add_edge("compare_summaries", END)

graph = graph_builder.compile()


if __name__ == "__main__":

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        inputs = {
            "original_word": user_input,
            "key_words":None,
            "article_texts":None,
            "summaries":None,
            "comparison_results":None
            }
        for output in graph.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
            print("\n---\n")