from langchain.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import secrets

load_dotenv()

# Initialize DuckDuckGo search
ddg_search = DuckDuckGoSearchAPIWrapper()

def get_model():
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

RESULTS_PER_QUESTION = 2

def webSearch(query: str, nums_results: int=RESULTS_PER_QUESTION):
    results = ddg_search.results(query, nums_results)
    return [r["link"] for r in results]

template = """{context} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""

def get_summarize_prompt():
    return ChatPromptTemplate.from_template(template=template)

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

def get_scrape_and_summarize_chain(model):
    return RunnablePassthrough.assign(
        summary = RunnablePassthrough.assign(
        context = lambda x: scrapeText(x["url"])[:5000]
    ) | get_summarize_prompt() | model | StrOutputParser()
    ) | (lambda x: f"URL: {x['url']} \n\nSummary: {x['summary']}")

def get_web_search_chain(model):
    return RunnablePassthrough.assign(
        urls = lambda x: webSearch(x["question"])
    )| (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | get_scrape_and_summarize_chain(model).map()

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

def get_search_question_chain(model):
    return SEARCH_PROMPT | model | StrOutputParser() | json.loads

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

def get_prompt():
    return ChatPromptTemplate.from_messages(
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

def get_chain(model):
    return RunnablePassthrough.assign(
        research_summary = get_web_search_chain(model) | collapse_lists
    ) | get_prompt() | model | StrOutputParser()

# FastAPI app setup
app = FastAPI(
    title="Research Assistant",
    version="1.0",
    description="An AI-powered research assistant that generates detailed reports based on web searches",
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

# Create templates directory and add HTML template
templates = Jinja2Templates(directory="templates")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create the HTML template for API key input
api_key_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Assistant - API Key Required</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #666;
        }
        input[type="password"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Research Assistant</h1>
        <p>Please enter your Groq API key to access the Research Assistant.</p>
        <form method="post" action="/">
            <div class="form-group">
                <label for="api_key">Groq API Key:</label>
                <input type="password" id="api_key" name="api_key" required>
            </div>
            <button type="submit">Submit</button>
        </form>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

# Save the template
with open("templates/api_key.html", "w") as f:
    f.write(api_key_template)

@app.get("/", include_in_schema=False)
async def root(request: Request):
    if "api_key" not in request.session:
        return templates.TemplateResponse("api_key.html", {"request": request})
    return RedirectResponse(url="/research")

@app.post("/", include_in_schema=False)
async def verify_api_key(request: Request, api_key: str = Form(...)):
    # Store the API key in session
    request.session["api_key"] = api_key
    # Set the environment variable
    os.environ["GROQ_API_KEY"] = api_key
    return RedirectResponse(url="/research", status_code=303)

@app.get("/logout", include_in_schema=False)
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

@app.post("/research")
async def research(request: Request, question: str = Form(...)):
    if "api_key" not in request.session:
        return RedirectResponse(url="/", status_code=303)
    
    try:
        # Initialize model only when needed
        model = get_model()
        chain = get_chain(model)
        response = chain.invoke({"question": question})
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .question {{
                    color: #333;
                    margin-bottom: 20px;
                }}
                .response {{
                    white-space: pre-wrap;
                    line-height: 1.6;
                }}
                .back-button {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }}
                .back-button:hover {{
                    background-color: #0056b3;
                }}
                .logout-button {{
                    background-color: #dc3545;
                    margin-left: 10px;
                }}
                .logout-button:hover {{
                    background-color: #c82333;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Research Results</h1>
                <div class="question">
                    <strong>Question:</strong> {question}
                </div>
                <div class="response">
                    {response}
                </div>
                <a href="/research" class="back-button">Ask Another Question</a>
                <a href="/logout" class="back-button logout-button">Logout</a>
            </div>
        </body>
        </html>
        """)
    except Exception as e:
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .error {{
                    color: red;
                    margin-top: 10px;
                }}
                .back-button {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }}
                .back-button:hover {{
                    background-color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Error</h1>
                <div class="error">
                    An error occurred: {str(e)}
                </div>
                <a href="/research" class="back-button">Back to Research</a>
            </div>
        </body>
        </html>
        """)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)