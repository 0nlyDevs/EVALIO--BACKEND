# AGENTS.md

## Running the Server

```bash
python server.py
```

Runs on `http://0.0.0.0:8000`

## Setup Requirements

1. Copy `.env.example` to `.env` and fill in values
2. PostgreSQL must be running on localhost:5432 with database `judgy`
3. Install deps: `pip install -r requirements.txt`

## Required Environment Variables

- `LLM_API_KEY` - LLM API key
- `LLM_BASE_URL` - LLM API base URL
- `FREE_LLM_MODEL` - Default: `liquid/lfm-2.5-1.2b-thinking:free`
- `HF_TOKEN` - HuggingFace token
- `EMBEDDING_MODEL` - Default: `sentence-transformers/all-MiniLM-L6-v2`
- `DB_USER`, `DB_PASSWORD` - PostgreSQL credentials
- `GITHUB_TOKEN` - GitHub API token (optional, increases rate limit)

## Project Structure

- `server.py` - FastAPI entry point; loads `.env` via `dotenv`, initializes DB on startup
- `db.py` - PostgreSQL connection (`host=localhost:5432, database=judgy`) and table creation
- `agents/` - Contains 4 agent routers:
  - `marketagent.py` - Market analysis (web search + README analysis)
  - `codeagent.py` - Code analysis (GitHub repo analysis via API)
  - `chatagent.py` - Chat functionality
  - `crudagent.py` - CRUD operations + LLM-based scoring

All agents mounted under `/api` prefix.

## API Endpoints

### Hackathons
- `POST /api/create-hackathon` - Create hackathon with `criteria` (e.g., "Code Quality, Innovation, Tech Stack")
- `GET /api/get-hackathon/{id}` - Get hackathon details
- `GET /api/get-all-hackathons` - List all hackathons

### Projects
- `POST /api/create-project` - Submit project to hackathon (body: `shortDescription`, `longDescription`, `githubLink`, `demoLink` (optional), `theme`, `hackathonId`)
- `GET /api/get-project/{id}` - Get project with analyses and score
- `GET /api/get-hackathon-projects/{hackathon_id}` - List projects in hackathon (includes `demo_link`)
- `GET /api/get-all` - List all projects

### Scoring (LLM-Generated)
- `GET /api/get-project-score/{project_id}` - Get LLM-generated overall score (0-10) with explanation
  - Score based on: theme alignment, code quality, market analysis, hackathon criteria
  - Returns `overall_score`, `score_explanation` (bullet points)
- `GET /api/get-hackathon-leaderboard/{hackathon_id}` - Get ranked projects by overall score

### Legacy
- `POST /api/search` - Semantic search projects
- `POST /api/review` - Mark project as reviewed

## Code Analysis (codeagent.py)

### How it works:
1. Fetches repository contents via GitHub API (prioritizes source code over config files)
2. Creates vectorstore from code files using Chroma + HuggingFace embeddings
3. Evaluates hackathon criteria against actual code:
   - **Code Quality** - Analyzes readability, modularity, error handling, best practices
   - **Tech Stack** - Detects frameworks, libraries, languages from code
   - **Innovation** - Assessed from project description (NOT code)
   - **Custom criteria** - Evaluated against actual code with specific feedback
4. LLM explicitly states whether project meets each criterion or what's missing

### Key Functions:
- `fetch_repo_contents()` - Fetches up to 50 source files + 10 config files
- `query_codebase_detailed()` - Analyzes code with hackathon criteria context (10 docs, 6000 chars)
- `assess_innovation()` - Evaluates innovation from project description only
- `generate_overall_project_score()` - Triggers LLM scoring after analyses complete

## Market Analysis (marketagent.py)

### How it works:
1. Fetches README from GitHub repository
2. Uses web search (DDGS) for market data
3. Analyzes: target audience, market potential, competitors, pitfalls, revenue models
4. Updates `market_agent_analysis` JSONB in projects table
5. Triggers overall score generation after completion

## LLM-Based Scoring (crudagent.py)

### How it works:
1. Called automatically after code + market analyses complete
2. LLM evaluates ALL factors:
   - Theme alignment (does project fit hackathon theme?)
   - Code quality (from code analysis)
   - Market potential (from market analysis)
   - Hackathon criteria compliance
3. Returns JSON: `{"score": 0-10, "reasons": ["reason1", "reason2"]}`
4. Score saved to `projects.overall_score` (0-1 scale)
5. Explanation saved to `projects.score_explanation` (bullet points)

### Score Explanation Format:
```
- Good theme alignment with AI focus
- Code quality needs improvement in error handling
- Strong market potential in healthcare
```

## Database Schema

```
hackathons:
  id SERIAL PRIMARY KEY
  name VARCHAR(255) NOT NULL
  description TEXT DEFAULT ''
  theme TEXT DEFAULT ''
  is_allowed BOOLEAN DEFAULT FALSE
  criteria TEXT DEFAULT '' (comma-separated: "Code Quality, Innovation")
  deadline TIMESTAMP
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

projects:
  id SERIAL PRIMARY KEY
  project_id VARCHAR(255) UNIQUE NOT NULL
  hackathon_id INTEGER REFERENCES hackathons(id) ON DELETE SET NULL
  short_description TEXT DEFAULT ''
  long_description TEXT DEFAULT ''
  github_link TEXT DEFAULT ''
  demo_link TEXT DEFAULT NULL
  theme TEXT DEFAULT ''
  is_reviewed BOOLEAN DEFAULT FALSE
  code_agent_analysis JSONB DEFAULT '[]'::jsonb
  market_agent_analysis JSONB DEFAULT '[]'::jsonb
  overall_score DECIMAL(3,2) DEFAULT NULL (0-1 scale, LLM-generated)
  score_explanation TEXT DEFAULT '' (bullet points)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

evaluations: (legacy, kept for reference)
  id SERIAL PRIMARY KEY
  project_id VARCHAR(255) REFERENCES projects(project_id)
  criteria_name VARCHAR(255) NOT NULL
  score DECIMAL(3,2) DEFAULT 0.00
  remarks TEXT DEFAULT ''
  agent_type VARCHAR(50) DEFAULT 'code'
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

## Criteria Format

`criteria` TEXT in hackathons:
```
"Code Quality, Innovation, Tech Stack, Market Potential"
```

The AI generates specific questions for each criteria and evaluates code against them.

## Test Commands

```bash
# Start server
python server.py

# 1. Create a hackathon with criteria
curl -X POST http://localhost:8000/api/create-hackathon \
  -H "Content-Type: application/json" \
  -d '{"name": "AI Hackathon", "description": "Build with AI", "criteria": "Code Quality, Innovation, Tech Stack"}'

# 2. Get all hackathons
curl http://localhost:8000/api/get-all-hackathons

# 3. Get specific hackathon
curl http://localhost:8000/api/get-hackathon/1

# 4. Submit project to hackathon (with demo link)
curl -X POST http://localhost:8000/api/create-project \
  -H "Content-Type: application/json" \
  -d '{"shortDescription": "My AI Project", "longDescription": "Detailed description", "githubLink": "https://github.com/user/repo", "demoLink": "https://myproject.demo.com", "hackathonId": 1}'

# 5. Get project (after evaluation - includes score)
curl http://localhost:8000/api/get-project/{project_id}

# 6. Get projects in hackathon (includes demo_link)
curl http://localhost:8000/api/get-hackathon-projects/1

# 7. Get project score (LLM-generated with explanation)
curl http://localhost:8000/api/get-project-score/{project_id}

# 8. Get leaderboard (ranked by overall_score)
curl http://localhost:8000/api/get-hackathon-leaderboard/1

# OpenAPI docs
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

## SQL Schema File

See `SQL_SCHEMA.sql` for complete database schema with migrations.
