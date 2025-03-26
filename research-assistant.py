from langchain.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json
from langchain_groq import ChatGroq
import getpass
import os
from dotenv import load_dotenv

load_dotenv()

# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

RESULTS_PER_QUESTION = 2

ddg_search = DuckDuckGoSearchAPIWrapper()

def webSearch(query: str, nums_results: int=RESULTS_PER_QUESTION):
    results = ddg_search.results(query, nums_results)
    return [r["link"] for r in results]


template = """{context} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""

summarize_prompt = ChatPromptTemplate.from_template(template=template)

# function for scraping the text of the website
def scrapeText(url: str):
    try:
        response = requests.get(url=url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retrieve webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
    context = lambda x: scrapeText(x["url"])[:5000]
) | summarize_prompt | model | StrOutputParser()
) | (lambda x: f"URL: {x['url']} \n\nSummary: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: webSearch(x["question"])
)| (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 2 google search queries to search online that are form an"
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format"
            '["query1", "query2", "query3"]',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | model | StrOutputParser() | json.loads

# chain = search_question_chain | (lambda x: [{"question": q} for q in x]) |web_search_chain.map()

WRITER_SYSTEM_TEMPLATE = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_TEMPLATE),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary = web_search_chain | collapse_lists
) | prompt | model | StrOutputParser()



from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/research-assistant/",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


# response = chain.invoke(
#     {
#         "question": "What is the difference between langchain and langgraph?"
#     }
# )