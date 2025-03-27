# Research Assistant AI

A powerful AI-powered research assistant built with FastAPI and Groq LLM that helps users conduct comprehensive research on any topic. The application provides detailed summaries, key points, and recommendations based on user queries.

## 🌟 Features

- 🔒 Secure API key management through session storage
- 🔍 Advanced research capabilities powered by Groq LLM
- 🌐 Web search integration using DuckDuckGo
- 📝 Comprehensive research summaries
- 🎯 Key points extraction
- 📚 Source citations
- 💡 Smart recommendations
- 🎨 Modern, responsive UI

## 🛠️ Technology Stack

- **Backend**: FastAPI
- **AI/ML**: 
  - Groq LLM (lama-3.3-70b-versatile)
  - LangChain
- **Frontend**: 
  - HTML
  - CSS
  - Jinja2 Templates
- **Search**: DuckDuckGo API
- **Deployment**: Render.com

## 📋 Prerequisites

- Python 3.11 or higher
- Groq API key ([Get it here](https://console.groq.com))
- Git

## 🚀 Local Development Setup

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd research-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn research-assistant:app --reload
   ```

5. Visit `http://localhost:8000` in your browser

## 🌍 Deployment

The application is configured for deployment on Render.com using the following files:

### render.yaml
```yaml
services:
  - type: web
    name: research-assistant
    runtime: python3.11
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn research-assistant:app --host 0.0.0.0 --port $PORT
    autoDeploy: true
```

### Deployment Steps

1. Push your code to GitHub
2. Create a new Web Service on Render.com
3. Connect your GitHub repository
4. Render will automatically detect the configuration and deploy

## 📁 Project Structure

```
research-assistant/
├── research-assistant.py    # Main application file
├── requirements.txt        # Python dependencies
├── render.yaml            # Render deployment configuration
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
└── templates/           # HTML templates
    ├── api_key.html    # API key input page
    └── research.html   # Research interface
```

## 🔑 Environment Variables

No environment variables need to be set on the server as the application handles the Groq API key through user input and session management.

## 🔒 Security

- API keys are stored only in user sessions
- No API keys are stored on the server
- Secure password field for API key input
- Session-based authentication

## 🚦 Usage

1. Visit the deployed application
2. Enter your Groq API key
3. Submit your research question
4. Receive comprehensive research results including:
   - Detailed summary
   - Key points
   - Source citations
   - Recommendations

## ⚠️ Limitations

- Free tier on Render.com will spin down after 15 minutes of inactivity
- Initial request after inactivity may take a few seconds
- Research quality depends on the Groq API key's rate limits and quotas

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Groq](https://groq.com/) for the LLM API
- [LangChain](https://python.langchain.com/) for AI/ML tools
- [Render](https://render.com/) for hosting

## 📞 Support

If you encounter any issues or have questions, please open an issue in the GitHub repository.
