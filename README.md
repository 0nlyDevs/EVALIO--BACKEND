# Evalio

An AI-powered hackathon project evaluation platform that automates code analysis, market research, and scoring using LLMs. Evalio provides comprehensive project assessment through specialized agents that analyze code quality, tech stack, innovation potential, and market viability.

## Features

- **Automated Code Analysis** - Analyzes GitHub repositories for code quality, tech stack, and best practices
- **Market Analysis** - Evaluates market potential, competitors, and revenue models using web search
- **LLM-Based Scoring** - Generates overall project scores (0-10) with detailed explanations
- **Hackathon Management** - Create hackathons with custom evaluation criteria
- **Leaderboard** - Rank projects based on AI-generated scores
- **Semantic Search** - Search projects using vector embeddings

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL running with a database (configure `DB_HOST`, `DB_PORT`, `DB_NAME`)
- GitHub API token (optional, increases rate limit)

### Setup

1. Clone the repository and navigate to the project directory

2. Create and configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` and fill in your values:
- `LLM_API_KEY` - Your LLM API key
- `LLM_BASE_URL` - Your LLM API endpoint
- `HF_TOKEN` - HuggingFace token for embeddings
- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `DB_NAME` - Database name (default: evalio)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password
- `GITHUB_TOKEN` - GitHub API token (optional)

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python server.py
```

The server runs on `http://0.0.0.0:8000`

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | LLM API key | Required |
| `LLM_BASE_URL` | LLM API endpoint | `https://api.example.com/v1` |
| `FREE_LLM_MODEL` | LLM model to use | `liquid/lfm-2.5-1.2b-thinking:free` |
| `HF_TOKEN` | HuggingFace token | Required |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | PostgreSQL database | `evalio` |
| `DB_USER` | PostgreSQL username | `postgres` |
| `DB_PASSWORD` | PostgreSQL password | Required |
| `GITHUB_TOKEN` | GitHub API token | Optional |
| `BASE_PROMPT` | Base prompt template | Optional |

## Requirements

```
fastapi>=0.110.0
uvicorn>=0.27.1
starlette>=0.36.0
psycopg2-binary>=2.9.9
langchain>=0.1.14
langchain-core>=0.1.0
langchain-community>=0.0.38
langchain-openai>=0.1.0
langchain-huggingface>=0.1.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.0.1
openai>=1.14.0
chromadb>=0.4.24
sentence-transformers>=2.2.2
huggingface-hub>=0.20.0
torch>=2.0.0
transformers>=4.36.0
gitpython>=3.1.40
ddgs>=1.0.0
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.26.3,<2
pydantic>=2.6.0,<3
pandas>=2.1.4
tiktoken>=0.5.2
httpx>=0.27.0
tenacity>=8.2.3
pyyaml>=6.0.1
aiohttp>=3.9.5
```

## API Endpoints

### Hackathons
- `POST /api/create-hackathon` - Create hackathon with criteria
- `GET /api/get-hackathon/{id}` - Get hackathon details
- `GET /api/get-all-hackathons` - List all hackathons

### Projects
- `POST /api/create-project` - Submit project to hackathon
- `GET /api/get-project/{id}` - Get project with analyses and score
- `GET /api/get-hackathon-projects/{hackathon_id}` - List projects in hackathon
- `GET /api/get-all` - List all projects

### Scoring
- `GET /api/get-project-score/{project_id}` - Get LLM-generated score with explanation
- `GET /api/get-hackathon-leaderboard/{hackathon_id}` - Get ranked projects

## Project Structure

```
server.py         - FastAPI entry point, initializes DB
db.py            - PostgreSQL connection and table creation
agents/
  marketagent.py  - Market analysis (web search + README)
  codeagent.py    - Code analysis (GitHub repo analysis)
  chatagent.py    - Chat functionality
  crudagent.py    - CRUD operations + LLM scoring
```

## How It Works

1. **Submit Project** - Create a hackathon and submit projects with GitHub links
2. **Code Analysis** - Code agent fetches and analyzes repository contents against hackathon criteria
3. **Market Analysis** - Market agent researches market potential and competitors
4. **Scoring** - LLM evaluates all factors and generates overall score with explanation
5. **Leaderboard** - View ranked projects based on AI-generated scores
