# from langchain.prompts import ChatPromptTemplate
# import requests
# from bs4 import BeautifulSoup
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# import json
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException, Request, Form
# from fastapi.responses import RedirectResponse, HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from starlette.middleware.sessions import SessionMiddleware
# import secrets
# from fastapi.middleware.cors import CORSMiddleware
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field
# from typing import List, Optional
# from fastapi.responses import JSONResponse
# from fastapi import Depends

# load_dotenv()

# # Initialize DuckDuckGo search
# ddg_search = DuckDuckGoSearchAPIWrapper()

# def get_model(api_key: str):
#     if not api_key:
#         raise ValueError("API key is required")
#     return ChatGroq(
#         groq_api_key=api_key,
#         model_name="llama-3.3-70b-versatile",
#         temperature=0.7,
#         max_tokens=32768,
#         top_p=1,
#         verbose=True
#     )

# RESULTS_PER_QUESTION = 3

# def webSearch(query: str, nums_results: int=RESULTS_PER_QUESTION):
#     results = ddg_search.results(query, nums_results)
#     return [r["link"] for r in results]

# template = """{context} 
# -----------
# Using the above text, answer in short the following question: 
# > {question}
# -----------
# if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""

# def get_summarize_prompt():
#     return ChatPromptTemplate.from_template(template=template)

# def scrapeText(url: str):
#     try:
#         response = requests.get(url=url)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, "html.parser")
#             page_text = soup.get_text(separator=" ", strip=True)
#             return page_text
#         else:
#             return f"Failed to retrieve webpage: Status code {response.status_code}"
#     except Exception as e:
#         print(e)
#         return f"Failed to retrieve the webpage: {e}"

# def get_scrape_and_summarize_chain(model):
#     return RunnablePassthrough.assign(
#         summary = RunnablePassthrough.assign(
#         context = lambda x: scrapeText(x["url"])[:5000]
#     ) | get_summarize_prompt() | model | StrOutputParser()
#     ) | (lambda x: f"URL: {x['url']} \n\nSummary: {x['summary']}")

# def get_web_search_chain(model):
#     return RunnablePassthrough.assign(
#         urls = lambda x: webSearch(x["question"])
#     )| (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | get_scrape_and_summarize_chain(model).map()

# SEARCH_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "user",
#             "Write 3 google search queries to search online that are form an"
#             "objective opinion from the following: {question}\n"
#             "You must respond with a list of strings in the following format"
#             '["query1", "query2", "query3"]',
#         ),
#     ]
# )

# def get_search_question_chain(model):
#     return SEARCH_PROMPT | model | StrOutputParser() | json.loads

# WRITER_SYSTEM_TEMPLATE = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

# RESEARCH_REPORT_TEMPLATE = """Information:
# --------
# {research_summary}
# --------
# Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
# The report should focus on the answer to the question, should be well structured, informative, \
# in depth, with facts and numbers if available and a minimum of 1,200 words.
# You should strive to write the report as long as you can using all relevant and necessary information provided.
# You must write the report with markdown syntax.
# You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
# Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
# You must write the report in apa format.
# Please do your best, this is very important to my career."""

# def get_prompt():
#     return ChatPromptTemplate.from_messages(
#         [
#             ("system", WRITER_SYSTEM_TEMPLATE),
#             ("user", RESEARCH_REPORT_TEMPLATE),
#         ]
#     )

# def collapse_lists(list_of_lists):
#     content = []
#     for l in list_of_lists:
#         content.append("\n\n".join(l))
#     return "\n\n".join(content)

# def get_chain(model):
#     return RunnablePassthrough.assign(
#         research_summary = get_web_search_chain(model) | collapse_lists
#     ) | get_prompt() | model | StrOutputParser()

# # FastAPI app setup
# app = FastAPI(
#     title="Research Assistant",
#     version="1.0",
#     description="An AI-powered research assistant that generates detailed reports based on web searches",
# )

# # Add session middleware
# app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32), session_cookie="research_session", max_age=3600)

# # Create templates directory and add HTML template
# templates = Jinja2Templates(directory="templates")

# # Create templates directory if it doesn't exist
# os.makedirs("templates", exist_ok=True)

# # Create the HTML template for API key input
# api_key_template = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Research Assistant - API Key Required</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             max-width: 600px;
#             margin: 0 auto;
#             padding: 20px;
#             background-color: #f5f5f5;
#         }
#         .container {
#             background-color: white;
#             padding: 20px;
#             border-radius: 8px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         }
#         h1 {
#             color: #333;
#             margin-bottom: 20px;
#         }
#         .form-group {
#             margin-bottom: 15px;
#         }
#         label {
#             display: block;
#             margin-bottom: 5px;
#             color: #666;
#         }
#         input[type="password"] {
#             width: 100%;
#             padding: 8px;
#             border: 1px solid #ddd;
#             border-radius: 4px;
#             box-sizing: border-box;
#         }
#         button {
#             background-color: #007bff;
#             color: white;
#             padding: 10px 20px;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#         }
#         button:hover {
#             background-color: #0056b3;
#         }
#         .error {
#             color: red;
#             margin-top: 10px;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>Research Assistant</h1>
#         <p>Please enter your Groq API key to access the Research Assistant.</p>
#         <form method="post" action="/api-key">
#             <div class="form-group">
#                 <label for="api_key">Groq API Key:</label>
#                 <input type="password" id="api_key" name="api_key" required>
#             </div>
#             <button type="submit">Submit</button>
#         </form>
#         {% if error %}
#         <div class="error">{{ error }}</div>
#         {% endif %}
#     </div>
# </body>
# </html>
# """

# # Save the template
# with open("templates/api_key.html", "w") as f:
#     f.write(api_key_template)

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("api_key.html", {"request": request})

# @app.post("/api-key")
# async def set_api_key(request: Request):
#     form_data = await request.form()
#     api_key = form_data.get("api_key")
#     if not api_key:
#         raise HTTPException(status_code=400, detail="API key is required")
    
#     # Store the API key in the session
#     request.session["api_key"] = api_key
#     return templates.TemplateResponse("research.html", {"request": request})

# @app.get("/logout")
# async def logout(request: Request):
#     # Clear the session
#     request.session.clear()
#     return templates.TemplateResponse("api_key.html", {"request": request})

# @app.post("/research")
# async def research(request: Request):
#     # Get the API key from the session
#     api_key = request.session.get("api_key")
#     if not api_key:
#         raise HTTPException(status_code=401, detail="API key not found in session")
    
#     form_data = await request.form()
#     question = form_data.get("question")
#     if not question:
#         raise HTTPException(status_code=400, detail="Question is required")

#     try:
#         # Get the model with the API key from session
#         model = get_model(api_key)
        
#         # Create the final chain
#         final_chain = get_chain(model)
        
#         # Run the chain
#         result = final_chain.invoke({"question": question})
        
#         return templates.TemplateResponse(
#             "research.html",
#             {
#                 "request": request,
#                 "result": result,
#                 "question": question
#             }
#         )
#     except Exception as e:
#         return templates.TemplateResponse(
#             "research.html",
#             {
#                 "request": request,
#                 "error": str(e),
#                 "question": question
#             }
#         )

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)


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
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
# Fix for pydantic v1 deprecation warning
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import Depends
# Using Python's built-in markdown module instead of markdown2
import markdown

load_dotenv()

# Initialize DuckDuckGo search
ddg_search = DuckDuckGoSearchAPIWrapper()

def get_model(api_key: str):
    if not api_key:
        raise ValueError("API key is required")
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=32768,
        top_p=1,
        verbose=True
    )

RESULTS_PER_QUESTION = 3

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
            "Write 3 google search queries to search online that are form an"
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
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32), session_cookie="research_session", max_age=3600)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)

# Create templates directory and add HTML template
templates = Jinja2Templates(directory="templates")

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
        <form method="post" action="/api-key">
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

# Create the research template with Markdown rendering
research_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Assistant</title>
    <link rel="stylesheet" href="/static/css/github-markdown.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #007bff;
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 {
            margin: 0;
        }
        .logout {
            color: white;
            text-decoration: none;
            padding: 5px 10px;
            border: 1px solid white;
            border-radius: 4px;
        }
        .form-section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .result-section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #666;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            min-height: 100px;
            font-family: Arial, sans-serif;
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
        .markdown-body {
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
        }
        @media (max-width: 767px) {
            .markdown-body {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Research Assistant</h1>
        <a href="/logout" class="logout">Logout</a>
    </header>
    <div class="container">
        <div class="form-section">
            <h2>Research Question</h2>
            <form method="post" action="/research">
                <div class="form-group">
                    <label for="question">Enter your research question:</label>
                    <textarea id="question" name="question" required>{{ question if question else "" }}</textarea>
                </div>
                <button type="submit">Generate Research</button>
            </form>
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
        </div>
        
        {% if result %}
        <div class="result-section">
            <h2>Research Report</h2>
            <div class="markdown-body">
                {{ result_html|safe }}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Create the CSS file for GitHub-like Markdown styling
github_markdown_css = """
.markdown-body {
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
  line-height: 1.5;
  color: #24292e;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.5;
  word-wrap: break-word;
}

.markdown-body a {
  background-color: transparent;
  color: #0366d6;
  text-decoration: none;
}

.markdown-body a:active,
.markdown-body a:hover {
  outline-width: 0;
  text-decoration: underline;
}

.markdown-body strong {
  font-weight: bolder;
}

.markdown-body h1 {
  font-size: 2em;
  margin: 0.67em 0;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 0.3em;
}

.markdown-body h2 {
  font-size: 1.5em;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 0.3em;
}

.markdown-body h3 {
  font-size: 1.25em;
}

.markdown-body h4 {
  font-size: 1em;
}

.markdown-body h5 {
  font-size: 0.875em;
}

.markdown-body h6 {
  font-size: 0.85em;
  color: #6a737d;
}

.markdown-body img {
  border-style: none;
  max-width: 100%;
  box-sizing: content-box;
  background-color: #fff;
}

.markdown-body code,
.markdown-body pre {
  font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace;
  font-size: 1em;
}

.markdown-body pre {
  margin-top: 0;
  margin-bottom: 16px;
  padding: 16px;
  overflow: auto;
  line-height: 1.45;
  background-color: #f6f8fa;
  border-radius: 3px;
}

.markdown-body code {
  padding: 0.2em 0.4em;
  margin: 0;
  font-size: 85%;
  background-color: rgba(27, 31, 35, 0.05);
  border-radius: 3px;
}

.markdown-body pre code {
  padding: 0;
  margin: 0;
  font-size: 100%;
  word-break: normal;
  white-space: pre;
  background: transparent;
  border: 0;
}

.markdown-body blockquote {
  margin: 0;
  padding: 0 1em;
  color: #6a737d;
  border-left: 0.25em solid #dfe2e5;
}

.markdown-body ul,
.markdown-body ol {
  padding-left: 2em;
}

.markdown-body table {
  border-spacing: 0;
  border-collapse: collapse;
  width: 100%;
  overflow: auto;
  margin-top: 0;
  margin-bottom: 16px;
}

.markdown-body table th {
  font-weight: 600;
}

.markdown-body table th,
.markdown-body table td {
  padding: 6px 13px;
  border: 1px solid #dfe2e5;
}

.markdown-body table tr {
  background-color: #fff;
  border-top: 1px solid #c6cbd1;
}

.markdown-body table tr:nth-child(2n) {
  background-color: #f6f8fa;
}

.markdown-body hr {
  height: 0.25em;
  padding: 0;
  margin: 24px 0;
  background-color: #e1e4e8;
  border: 0;
}
"""

# Save the templates and CSS
with open("templates/api_key.html", "w") as f:
    f.write(api_key_template)

with open("templates/research.html", "w") as f:
    f.write(research_template)

with open("static/css/github-markdown.css", "w") as f:
    f.write(github_markdown_css)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("api_key.html", {"request": request})

@app.post("/api-key")
async def set_api_key(request: Request):
    form_data = await request.form()
    api_key = form_data.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    
    # Store the API key in the session
    request.session["api_key"] = api_key
    return templates.TemplateResponse("research.html", {"request": request})

@app.get("/logout")
async def logout(request: Request):
    # Clear the session
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)

@app.post("/research")
async def research(request: Request):
    # Get the API key from the session
    api_key = request.session.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key not found in session")
    
    form_data = await request.form()
    question = form_data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        # Get the model with the API key from session
        model = get_model(api_key)
        
        # Create the final chain
        final_chain = get_chain(model)
        
        # Run the chain
        result = final_chain.invoke({"question": question})
        
        # Convert markdown to HTML using Python's built-in markdown module
        # The 'extensions' parameter adds support for tables, fenced code blocks, etc.
        result_html = markdown.markdown(
            result, 
            extensions=[
                'markdown.extensions.fenced_code',
                'markdown.extensions.tables',
                'markdown.extensions.nl2br'
            ]
        )
        
        return templates.TemplateResponse(
            "research.html",
            {
                "request": request,
                "result": result,
                "result_html": result_html,
                "question": question
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "research.html",
            {
                "request": request,
                "error": str(e),
                "question": question
            }
        )

@app.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    return templates.TemplateResponse("research.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
