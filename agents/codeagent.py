from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import APIRouter, Request, HTTPException
import asyncio
import re
from db import get_database_connection
from psycopg2.extras import RealDictCursor
import json
import os
import tempfile
import shutil
import uuid
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
import re
from dotenv import load_dotenv

# Project type to relevant paths/extensions mapping
PROJECT_TYPE_CONFIG = {
    "NEXT_JS": {
        "dirs": {
            "src",
            "pages",
            "app",
            "components",
            "lib",
            "utils",
            "hooks",
            "public",
        },
        "extensions": {".js", ".jsx", ".ts", ".tsx", ".ts", ".css", ".scss"},
    },
    "REACT": {
        "dirs": {"src", "components", "pages", "utils", "hooks", "public"},
        "extensions": {".js", ".jsx", ".ts", ".tsx", ".css", ".scss"},
    },
    "VUE": {
        "dirs": {"src", "components", "views", "pages", "utils", "public"},
        "extensions": {".vue", ".js", ".ts", ".jsx", ".tsx", ".css", ".scss"},
    },
    "NUXT": {
        "dirs": {"components", "pages", "layouts", "composables", "utils", "public"},
        "extensions": {".vue", ".js", ".ts", ".jsx", ".tsx", ".css", ".scss"},
    },
    "ANGULAR": {
        "dirs": {"src", "app", "components", "services", "pages"},
        "extensions": {".ts", ".js", ".html", ".css", ".scss"},
    },
    "SVELTE": {
        "dirs": {"src", "components", "routes", "lib"},
        "extensions": {".svelte", ".js", ".ts", ".css"},
    },
    "SVELTEKIT": {
        "dirs": {"src", "lib", "routes", "components"},
        "extensions": {".svelte", ".js", ".ts", ".css"},
    },
    "ASTRO": {
        "dirs": {"src", "components", "pages", "layouts"},
        "extensions": {".astro", ".js", ".ts", ".jsx", ".tsx", ".css"},
    },
    "REMIX": {
        "dirs": {"app", "routes", "components", "lib"},
        "extensions": {".js", ".jsx", ".ts", ".tsx", ".css", ".scss"},
    },
    "TAILWIND": {
        "dirs": {"src", "components", "pages", "public"},
        "extensions": {".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".html"},
    },
    "NODE_EXPRESS": {
        "dirs": {"src", "routes", "controllers", "models", "middlewares", "utils"},
        "extensions": {".js", ".ts", ".json"},
    },
    "FASTAPI": {
        "dirs": {"app", "src", "routes", "api", "models", "schemas"},
        "extensions": {".py"},
    },
    "DJANGO": {
        "dirs": set(),  # Scan all, filter by extension
        "extensions": {".py"},
    },
    "SPRING_BOOT": {
        "dirs": {"src/main/java", "src/main/resources", "src/test/java"},
        "extensions": {".java", ".xml", ".properties", ".yml", ".yaml"},
    },
    "GIN": {
        "dirs": {"cmd", "internal", "pkg", "api"},
        "extensions": {".go"},
    },
    "RAILS": {
        "dirs": {"app", "lib", "config"},
        "extensions": {".rb"},
    },
    "LARAVEL": {
        "dirs": {"app", "resources", "routes", "config"},
        "extensions": {".php", ".blade.php"},
    },
    "ACTIX": {
        "dirs": {"src"},
        "extensions": {".rs"},
    },
    "SWIFT_UI": {
        "dirs": {"Sources", "App"},
        "extensions": {".swift"},
    },
    "KOTLIN_JETPACK": {
        "dirs": {"app/src/main/java", "app/src/main/kotlin"},
        "extensions": {".kt", ".java", ".xml"},
    },
    "REACT_NATIVE": {
        "dirs": {"src", "components", "screens", "navigation", "utils"},
        "extensions": {".js", ".jsx", ".ts", ".tsx"},
    },
    "EXPO": {
        "dirs": {"app", "src", "components", "screens", "utils"},
        "extensions": {".js", ".jsx", ".ts", ".tsx"},
    },
    "FLUTTER": {
        "dirs": {"lib", "lib/src"},
        "extensions": {".dart"},
    },
    "DOTNET_MAUI": {
        "dirs": {"Platforms", "ViewModels", "Views", "Models", "Services"},
        "extensions": {".cs", ".xaml"},
    },
    "IONIC": {
        "dirs": {"src", "app", "components", "pages", "services"},
        "extensions": {".ts", ".js", ".html", ".scss"},
    },
    "NATIVESCRIPT": {
        "dirs": {"app", "components", "pages"},
        "extensions": {".ts", ".js", ".xml", ".css"},
    },
    "VANILLA_JS": {
        "dirs": {"src", "js", "public", "static"},
        "extensions": {".js", ".html", ".css", ".scss"},
    },
    "OTHER": {
        "dirs": set(),  # Scan all
        "extensions": set(),  # All extensions
    },
}


def get_github_session():
    """Create a requests session with retry logic for transient errors"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


load_dotenv()

BASE_PROMPT = os.getenv("BASE_PROMPT", "").strip()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

router = APIRouter()


def get_github_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def parse_repo_url(repo_url):
    """Extract owner and repo from URL like https://github.com/owner/repo"""
    match = re.search(r"github\.com/([^/]+)/([^/?#]+?)(?:\.git)?(?:/|$|[?#])", repo_url)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        # Clean up any remaining .git or path components
        repo = repo.replace(".git", "").split("/")[0]
        return owner, repo
    return None, None


def get_default_branch(owner, repo):
    """Get the default branch of a repository"""
    session = get_github_session()
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = session.get(url, headers=get_github_headers())
    if response.status_code == 404:
        raise ValueError(f"Repository '{owner}/{repo}' not found.")
    if response.status_code == 403:
        raise ValueError(f"Access forbidden to '{owner}/{repo}'.")
    response.raise_for_status()
    return response.json().get("default_branch", "main")


def get_repo_tree(owner, repo, branch=None):
    """Get all files in repository using Git Trees API"""
    session = get_github_session()

    # Get default branch if not specified
    if branch is None:
        branch = get_default_branch(owner, repo)

    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    response = session.get(url, headers=get_github_headers())

    if response.status_code == 404:
        # Try common branch names
        for fallback_branch in ["main", "master", "prod", "develop"]:
            if fallback_branch == branch:
                continue
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{fallback_branch}?recursive=1"
            response = session.get(url, headers=get_github_headers())
            if response.status_code == 200:
                branch = fallback_branch
                break

    if response.status_code == 404:
        raise ValueError(
            f"Repository '{owner}/{repo}' not found or has no commits. Check the URL and ensure the repo exists."
        )
    if response.status_code == 403:
        raise ValueError(
            f"Access forbidden to '{owner}/{repo}'. Check GITHUB_TOKEN has proper permissions."
        )
    if response.status_code == 429:
        raise ValueError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN in .env to increase limit."
        )
    response.raise_for_status()
    return response.json().get("tree", [])


def get_file_content(owner, repo, path, branch="main"):
    """Get file content from GitHub API"""
    session = get_github_session()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    response = session.get(url, headers=get_github_headers())
    if response.status_code == 404:
        raise FileNotFoundError(f"File '{path}' not found in {owner}/{repo}")
    if response.status_code == 403:
        raise PermissionError(f"Access forbidden to file '{path}' in {owner}/{repo}")
    response.raise_for_status()
    data = response.json()
    if data.get("encoding") == "base64":
        content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        return content
    return data.get("content", "")


def fetch_repo_contents(owner, repo, branch=None, project_type="OTHER"):
    """Fetch code files from repository, filtered by project_type structure"""
    if branch is None:
        branch = get_default_branch(owner, repo)

    tree = get_repo_tree(owner, repo, branch)

    # Get config for this project type
    config = PROJECT_TYPE_CONFIG.get(project_type, PROJECT_TYPE_CONFIG["OTHER"])
    relevant_dirs = config["dirs"]
    relevant_extensions = config["extensions"]

    # Source code extensions (higher priority for analysis)
    source_extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".html",
        ".css",
        ".scss",
        ".less",
        ".vue",
        ".svelte",
        ".astro",
        ".dart",
    }

    # Config/metadata files (lower priority, still useful for tech stack)
    config_extensions = {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
    }

    skip_dirs = {
        ".git",
        "node_modules",
        "venv",
        "__pycache__",
        ".venv",
        "dist",
        "build",
        "target",
        ".github",
        "assets",
        "static",
        "public",
    }

    source_files = []
    config_files = []

    for item in tree:
        if item.get("type") == "blob":
            path = item.get("path", "")

            # Check if directory is in relevant_dirs (if specified)
            path_parts = path.split("/")
            is_relevant_dir = True
            if relevant_dirs:
                is_relevant_dir = any(part in relevant_dirs for part in path_parts[:-1])

            # Skip if not in relevant directories
            if relevant_dirs and not is_relevant_dir:
                continue

            # Skip common excluded directories
            if any(skip_dir in path_parts for skip_dir in skip_dirs):
                continue

            ext = os.path.splitext(path)[1].lower()

            # Filter by extension if specified
            if (
                relevant_extensions
                and ext not in relevant_extensions
                and ext not in source_extensions
                and ext not in config_extensions
            ):
                continue

            try:
                content = get_file_content(owner, repo, path, branch)

                # Skip very large files
                if len(content) > 50000:
                    continue

                file_info = {"path": path, "content": content, "type": ext}

                if ext in source_extensions or (
                    relevant_extensions and ext in relevant_extensions
                ):
                    source_files.append(file_info)
                elif ext in config_extensions or path.endswith(
                    ("requirements.txt", "package.json", "Dockerfile")
                ):
                    config_files.append(file_info)

            except Exception as e:
                print(f"Skipping {path}: {e}")

    # Prioritize source files, limit config files
    files = source_files[:50] + config_files[:10]
    return files


def get_llm():
    return ChatOpenAI(
        model=os.getenv("FREE_LLM_MODEL", "liquid/lfm-2.5-1.2b-thinking:free"),
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )


def query_codebase(vectorstore, question, llm):
    """Query the codebase with a specific question"""
    if vectorstore is None:
        return "No code found to analyze"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    try:
        relevant_docs = retriever.invoke(question)
    except AttributeError:
        try:
            relevant_docs = retriever.get_relevant_documents(question)
        except:
            return "Unable to retrieve code context"

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    system_prompt = """You are a code reviewer analyzing a hackathon project.
Answer based on the code provided. Be concise - one paragraph, max 70 words.
If you can't determine something, say so honestly."""
    if BASE_PROMPT:
        system_prompt = BASE_PROMPT + "\n\n" + system_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                """Code Context:
{context}

Question: {question}

Answer:""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"context": context[:3000], "question": question})
        return response
    except Exception as e:
        return f"Analysis error: {str(e)}"


def query_codebase_detailed(vectorstore, llm, question, project_description=""):
    """Query the codebase with detailed analysis, using more context"""
    if vectorstore is None:
        return "No code found to analyze"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    try:
        relevant_docs = retriever.invoke(question)
    except AttributeError:
        try:
            relevant_docs = retriever.get_relevant_documents(question)
        except:
            return "Unable to retrieve code context"

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    system_prompt = """You are a senior software engineer evaluating a hackathon project.
Analyze the actual code provided below. Do NOT just summarize package.json or dependency files.
Base your evaluation on the source code structure, patterns, and implementation quality.

If evaluating code quality: look for readability, modularity, error handling, naming conventions, and best practices.
If a specific criterion is requested: explicitly state whether the project meets it or not, and why.

Be specific and provide concrete observations from the code. Max 200 words."""
    if BASE_PROMPT:
        system_prompt = BASE_PROMPT + "\n\n" + system_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                """Project Description: {project_description}

Code Context:
{context}

Evaluation Task:
{question}

Detailed Assessment:""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke(
            {
                "context": context[:6000],
                "question": question,
                "project_description": project_description or "No description provided",
            }
        )
        return response
    except Exception as e:
        return f"Analysis error: {str(e)}"


def assess_innovation(llm, project_description: str, hackathon_name: str = "") -> str:
    """Assess project innovation based on description (NOT code)"""
    if not project_description or len(project_description.strip()) < 20:
        return "Unable to assess innovation: insufficient project description."

    system_prompt = """You are a hackathon judge evaluating project innovation.
Assess how innovative, creative, and unique this project is.
Consider: novelty of the idea, originality, problem-solving approach, and potential impact.

Rate from 0-10 and explain your reasoning in 2-3 sentences.
Format: "Score: X/10 - <reasoning>"

Do NOT analyze code. Focus on the project idea and its innovative potential."""
    if BASE_PROMPT:
        system_prompt = BASE_PROMPT + "\n\n" + system_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                """Hackathon: {hackathon_name}

Project Description:
{project_description}

How innovative is this project?""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke(
            {
                "project_description": project_description[:2000],
                "hackathon_name": hackathon_name or "Unknown Hackathon",
            }
        )
        return response
    except Exception as e:
        return f"Innovation assessment error: {str(e)}"


def analyze_repository(
    repo_url: str, questions: list[str], project_type: str = "OTHER"
) -> tuple[list[dict], int]:
    """Analyze a GitHub repository using the same logic as POST /code-agent/analyze
    Returns: (analysis_results, number_of_files_analyzed)
    """
    owner, repo_name = parse_repo_url(repo_url)
    if not owner or not repo_name:
        raise ValueError("Invalid repository URL format")

    print(f"Fetching repo: {owner}/{repo_name} (type: {project_type})")
    files = fetch_repo_contents(owner, repo_name, project_type=project_type)
    num_files = len(files)
    print(f"Fetched {num_files} files")

    if not files:
        return ([], 0)

    documents = []
    for f in files:
        from langchain_core.documents import Document

        documents.append(
            Document(
                page_content=f"File: {f['path']}\n\n{f['content']}",
                metadata={"source": f["path"]},
            )
        )

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Create vectorstore
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"repo_{uuid.uuid4().hex[:8]}",
    )

    # Analyze with questions
    llm = get_llm()
    results = []
    for question in questions:
        print(f"Analyzing: {question}")
        answer = query_codebase(vectorstore, question, llm)
        results.append({"question": question, "answer": answer})
        print(f"Answer: {answer[:100]}...")

    return (results, num_files)


def extract_score_from_text(text: str) -> float:
    """Extract a score between 0 and 1 from text response"""
    patterns = [
        r"(\d+\.?\d*)/10",
        r"(\d+\.?\d*)%",
        r"[Ss]core:?\s*(\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if value > 1:
                value = value / 100 if value <= 100 else value / 10
            return min(1.0, max(0.0, value))

    return 0.5


def save_evaluation(
    project_id: str, criteria_name: str, score: float, remarks: str, agent_type: str
):
    """Save evaluation to database"""
    conn = get_database_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO evaluations (project_id, criteria_name, score, remarks, agent_type)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (project_id, criteria_name, score, remarks, agent_type),
    )
    conn.commit()
    cur.close()
    conn.close()


@router.get("/code-agent")
async def codeAgent_endpoint():
    """Test endpoint - returns available functionality"""
    return {
        "message": "Code Agent is running",
        "capabilities": [
            "Clone and analyze GitHub repositories",
            "Technology stack detection",
            "Code quality assessment",
            "Dependency analysis",
        ],
        "usage": {
            "endpoint": "POST /api/code-agent/analyze",
            "body": {"repo_url": "https://github.com/username/repository"},
        },
    }


@router.post("/code-agent/analyze")
async def code_agent_analyze(request: Request):
    """Analyze a GitHub repository"""
    try:
        data = await request.json()
        repo_url = data.get("repo_url", "")
        project_type = data.get("project_type", "OTHER")

        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")

        print(f"Analyzing repository: {repo_url} (type: {project_type})")

        questions = [
            "What technologies and programming languages are used?",
            "Explain the project structure and purpose",
            "How is the code quality?",
            "What dependencies and libraries are used?",
        ]

        results, num_files = analyze_repository(
            repo_url, questions, project_type=project_type
        )

        if num_files == 0:
            return {"message": "No code files found in repository", "analysis": []}

        return {
            "message": "Code analysis complete",
            "repo_url": repo_url,
            "files_analyzed": num_files,
            "analysis": results,
        }

    except ValueError as e:
        print(f"Validation error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except (FileNotFoundError, PermissionError) as e:
        print(f"File access error: {str(e)}")
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        print(f"Error in code analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


async def invoke_code_agent(repolink: str, project_id: str, hackathon_id: int = None):
    """Background task for automatic code analysis via GitHub API"""
    try:
        criteria_text = ""
        project_description = ""
        hackathon_name = ""

        # Fetch project and hackathon details
        conn = get_database_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            "SELECT short_description, long_description, hackathon_id, project_type FROM projects WHERE project_id = %s",
            (project_id,),
        )
        project = cur.fetchone()
        if project:
            project_description = f"{project.get('short_description', '')} {project.get('long_description', '')}".strip()
            project_type = project.get("project_type", "OTHER")
        else:
            project_type = "OTHER"

        if hackathon_id:
            cur.execute(
                "SELECT criteria, name FROM hackathons WHERE id = %s", (hackathon_id,)
            )
            hackathon = cur.fetchone()
            if hackathon:
                criteria_text = hackathon.get("criteria") or ""
                hackathon_name = hackathon.get("name", "")

        cur.close()
        conn.close()

        if not repolink or not repolink.startswith("http"):
            print(f"Invalid repo URL: {repolink}")
            save_evaluation(
                project_id, "Repo Validation", 0, "Invalid GitHub URL", "code"
            )
            return

        owner, repo_name = parse_repo_url(repolink)
        if not owner or not repo_name:
            print(f"Invalid repo URL format: {repolink}")
            save_evaluation(
                project_id,
                "Repo Validation",
                0,
                "Invalid repository URL format",
                "code",
            )
            return

        print(f"Fetching {owner}/{repo_name} via GitHub API")
        files = fetch_repo_contents(owner, repo_name, project_type=project_type)
        print(f"Fetched {len(files)} files")

        if not files:
            save_evaluation(
                project_id,
                "Code Quality",
                0,
                "No code files found in repository",
                "code",
            )
            return

        documents = []
        for f in files:
            from langchain_core.documents import Document

            documents.append(
                Document(
                    page_content=f"File: {f['path']}\n\n{f['content']}",
                    metadata={"source": f["path"]},
                )
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, collection_name=f"repo_{project_id}"
        )

        llm = get_llm()

        criteria_list = [c.strip() for c in criteria_text.split(",") if c.strip()]
        if not criteria_list:
            criteria_list = ["Code Quality", "Tech Stack", "Innovation"]

        results = []
        for name in criteria_list:
            answer = ""
            score = 0.5

            # Innovation should be assessed from project description, NOT code
            if name.lower() == "innovation":
                answer = assess_innovation(llm, project_description, hackathon_name)
                score = extract_score_from_text(answer)

            # Tech Stack - analyze from codebase
            elif name.lower() == "tech stack" or "tech" in name.lower():
                answer = query_codebase(
                    vectorstore,
                    "What technologies, frameworks, libraries, and programming languages are used in this project? List all dependencies, frameworks, and tools found in the code.",
                    llm,
                )
                score = 1.0 if answer and answer != "No code found to analyze" else 0.5

            # Code Quality and other criteria - analyze from codebase with criteria context
            else:
                criteria_context = f"the hackathon criteria: {name}"
                answer = query_codebase_detailed(
                    vectorstore,
                    llm,
                    f"""Evaluate the project's {name} based on these aspects:
1. Does the code follow best practices for {name}?
2. Are there any specific issues or violations related to {name}?

Context: This project is part of the "{hackathon_name}" hackathon. Assess whether the code meets expectations for {name}.
If the project does not follow good practices for {name}, specify exactly what is missing or poorly implemented.""",
                    project_description,
                )
                score = extract_score_from_text(answer)

            save_evaluation(project_id, name, score, answer, "code")
            results.append(
                {"name": name, "score": score, "answer": answer[:500] if answer else ""}
            )

        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE projects SET code_agent_analysis = %s WHERE project_id = %s",
            (json.dumps(results), project_id),
        )
        conn.commit()
        cur.close()
        conn.close()

        # Generate overall score after code analysis
        generate_overall_project_score(project_id)

        print(f"Code Agent: Analysis complete for project {project_id}")

    except Exception as e:
        print(f"Code Agent Error: {str(e)}")
        import traceback

        traceback.print_exc()
        # Ensure DB connections are closed on error
        try:
            cur.close()
            conn.close()
        except:
            pass


def generate_overall_project_score(project_id: str):
    """Generate and save overall project score based on all analyses"""
    try:
        from agents.crudagent import generate_overall_score

        conn = get_database_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get project details
        cur.execute(
            """
            SELECT p.*, h.name as hackathon_name, h.theme as hackathon_theme, h.criteria
            FROM projects p
            LEFT JOIN hackathons h ON p.hackathon_id = h.id
            WHERE p.project_id = %s
        """,
            (project_id,),
        )
        project = cur.fetchone()

        if not project:
            print(f"Project {project_id} not found for scoring")
            cur.close()
            conn.close()
            return

        # Parse JSONB fields
        import json

        code_analysis = project.get("code_agent_analysis") or []
        market_analysis = project.get("market_agent_analysis") or []

        if isinstance(code_analysis, str):
            code_analysis = json.loads(code_analysis)
        if isinstance(market_analysis, str):
            market_analysis = json.loads(market_analysis)

        score, explanation = generate_overall_score(
            project_id=project_id,
            short_description=project.get("short_description", ""),
            long_description=project.get("long_description", ""),
            hackathon_name=project.get("hackathon_name", ""),
            hackathon_theme=project.get("hackathon_theme", ""),
            criteria=project.get("criteria", ""),
            code_analysis=code_analysis,
            market_analysis=market_analysis,
        )

        # Save to database
        cur.execute(
            """
            UPDATE projects SET overall_score = %s, score_explanation = %s
            WHERE project_id = %s
        """,
            (score, explanation, project_id),
        )

        conn.commit()
        cur.close()
        conn.close()

        print(f"Overall score generated for project {project_id}: {score}")

    except Exception as e:
        print(f"Error generating overall score: {str(e)}")
        import traceback

        traceback.print_exc()
