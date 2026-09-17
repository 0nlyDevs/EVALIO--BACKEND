# AGENTS.md

## Running the Server

```bash
python server.py
```

Runs on `http://0.0.0.0:8000`. Swagger UI at `/docs`, ReDoc at `/redoc`.

## Setup

1. Copy `.env.example` to `.env` and fill in values
2. PostgreSQL must be reachable (configure `DB_HOST`, `DB_PORT`, `DB_NAME`)
3. Install deps: `pip install -r requirements.txt`

## Environment Variables

- `LLM_API_KEY` - LLM API key (required)
- `LLM_BASE_URL` - LLM API endpoint (e.g. `https://api.example.com/v1`)
- `FREE_LLM_MODEL` - Default: `liquid/lfm-2.5-1.2b-thinking:free`
- `HF_TOKEN` - HuggingFace token (required, used for embeddings download)
- `EMBEDDING_MODEL` - Default: `sentence-transformers/all-MiniLM-L6-v2`
- `BASE_PROMPT` - Optional base prompt prepended to all LLM system prompts
- `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` - PostgreSQL (defaults: `localhost` / `5432` / `evalio` / `postgres` / `postgres`)
- `GITHUB_TOKEN` - GitHub token for gitingest (optional, used for private repos)
- `CORS_ORIGINS` - Comma-separated allowed origins. When unset or empty, allows ALL origins (`*`)

## Project Structure

- `server.py` - FastAPI entry point; loads `.env` via `dotenv`, initializes DB on startup, mounts all agent routers under `/api`
- `db.py` - PostgreSQL connection via `psycopg2` and `init_db()` that creates tables + runs migrations
- `agents/` - Four agent routers:
  - `crudagent.py` - CRUD operations, LLM-based scoring (`generate_overall_score`), semantic search
  - `codeagent.py` - GitHub repo analysis via gitingest (clones repo, extracts content, builds Chroma vectorstore, evaluates criteria)
  - `marketagent.py` - Market analysis (fetches README via gitingest, web search via `ddgs`, evaluates market questions)
  - `chatagent.py` - Chat with project context (simple chat + project-aware chat)

## Key Architecture Notes

- **Gitingest**: Repo content fetched via `gitingest` Python package (clones temporarily, extracts all text content). No manual GitHub API calls.
- **Background tasks**: `POST /api/create-project` triggers `invoke_code_agent` and `invoke_market_agent` as `asyncio.create_task` (fire-and-forget). Analysis runs after the response is returned.
- **Score generation**: After code and market analyses complete, `generate_overall_project_score()` is called automatically. It uses the LLM to produce a 0-10 score saved as 0-1 scale in `overall_score`.
- **DB auto-migration**: `init_db()` uses try/except `ALTER TABLE ADD COLUMN` to add columns if they don't exist. Safe to run on existing DBs.

## API Endpoints

### Health
- `GET /health`

### Hackathons
- `POST /api/create-hackathon` - Body: `name`, `description`, `theme`, `criteria` (comma-separated), `deadline` (ISO timestamp), `isAllowed` (bool)
- `GET /api/get-hackathon/{id}`
- `GET /api/get-all-hackathons`

### Projects
- `POST /api/create-project` - Body: `shortDescription`, `longDescription`, `githubLink`, `demoLink` (optional), `theme`, `hackathonId` (optional), `projectType` (optional, enum value from `project_type` e.g. `REACT`, `NEXT_JS`, `FASTAPI`)
- `GET /api/get-project/{project_id}` - Returns project with analyses and score
- `GET /api/get-hackathon-projects/{hackathon_id}`
- `GET /api/get-all`

### Scoring
- `GET /api/get-project-score/{project_id}` - Returns `overall_score` (0-10 scale) with `score_explanation`
- `GET /api/get-hackathon-leaderboard/{hackathon_id}` - Ranked projects by score

### Chat
- `POST /api/chat-agent/simple` - Body: `question`. Simple Q&A without project context
- `POST /api/chat-agent` - Body: `question`, `project_id` (optional), `chathistory` (optional). Project-aware chat that loads code analysis context

### Legacy
- `POST /api/search` - Semantic search projects
- `POST /api/review` - Mark project as reviewed

## Database Schema

See `SQL_SCHEMA.sql` for the full schema. Key tables:

**hackathons**: `id`, `name`, `description`, `theme`, `is_allowed`, `criteria` (comma-separated string), `deadline`, `created_at`

**projects**: `id`, `project_id` (UUID), `hackathon_id` (FK), `short_description`, `long_description`, `github_link`, `demo_link`, `theme`, `is_reviewed`, `code_agent_analysis` (JSONB), `market_agent_analysis` (JSONB), `overall_score` (DECIMAL 0-1), `score_explanation` (text), `project_type` (enum), `created_at`

**project_type enum**: `VANILLA_JS`, `REACT`, `NEXT_JS`, `VUE`, `NUXT`, `ANGULAR`, `SVELTE`, `SVELTEKIT`, `ASTRO`, `REMIX`, `TAILWIND`, `NODE_EXPRESS`, `FASTAPI`, `DJANGO`, `SPRING_BOOT`, `GIN`, `RAILS`, `LARAVEL`, `ACTIX`, `SWIFT_UI`, `KOTLIN_JETPACK`, `REACT_NATIVE`, `EXPO`, `FLUTTER`, `DOTNET_MAUI`, `IONIC`, `NATIVESCRIPT`, `OTHER`

**evaluations** (legacy): `id`, `project_id` (FK), `criteria_name`, `score`, `remarks`, `agent_type`

## Code Analysis Flow

1. Parse GitHub URL
2. Ingest repo via gitingest (clones temporarily, extracts all text files)
3. Split content into chunks, embed with HuggingFace, store in Chroma vectorstore
4. For each hackathon criterion, query the vectorstore with a specific prompt
5. Exception: Innovation is assessed from project description only, not code

## Market Analysis Flow

1. Fetch README from GitHub repo via gitingest
2. If README < 50 chars, returns "insufficient data" for all questions
3. Uses `ddgs` (DuckDuckGo) for web search to supplement market data
4. Evaluates 5 market questions: audience, potential, competitors, pitfalls, revenue
