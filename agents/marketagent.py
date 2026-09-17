import asyncio
import base64
import re
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastapi import APIRouter, Request, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from db import get_database_connection
from psycopg2.extras import RealDictCursor
import json
import os
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

BASE_PROMPT = os.getenv("BASE_PROMPT", "").strip()

router = APIRouter()


def get_llm():
    return ChatOpenAI(
        model=os.getenv("FREE_LLM_MODEL", "liquid/lfm-2.5-1.2b-thinking:free"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0.2,
    )


def run_search(query: str, max_results: int = 5) -> str:
    """Reliable search using ddgs"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No relevant search results found."

        # Extract useful text
        formatted = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            formatted.append(f"{title}: {body}")

        return "\n".join(formatted)

    except Exception as e:
        return f"Search failed: {str(e)}"


def research_question(idea, question, llm, readme_info: str = ""):
    """Research a market question using web search and LLM"""

    search_query = f"{idea} startup market research: {question}"

    search_results = run_search(search_query)

    system_prompt = """You are a market research analyst.

Rules:
- Base your answer PRIMARYLY on the provided README/project description
- Use web search only to supplement with market data (numbers, trends, competitors)
- If README provides clear information about the product, use it to make educated answers
- Do NOT say "insufficient data" if the README clearly describes the product
- Max 70 words
- One paragraph"""
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
                """Project Idea: {idea}
{readme_info}
Web &&&  (for market data only):
{search_results}

Question: {question}

Answer:""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke(
            {
                "idea": idea,
                "readme_info": readme_info,
                "search_results": search_results[:3000],
                "question": question,
            }
        )
        return response.strip()

    except Exception as e:
        return f"LLM error: {str(e)}"


async def analyze_market(idea: str, theme: str, readme_content: str = ""):
    """Perform full market analysis"""
    llm = get_llm()

    marketQuestions = [
        "Who is the target audience of this idea?",
        "What is the market potential and size?",
        "What are the main competitors?",
        "What are the potential pitfalls?",
        "What is the revenue model potential?",
    ]

    readme_info = ""
    results = []
    if readme_content and len(readme_content.strip()) > 50:
        readme_info = f"""
Project README/Description:
{readme_content[:5000]}
"""
    else:
        readme_info = """
No sufficient README data available. The project does not have a meaningful README file.
"""
        for question in marketQuestions:
            print(f"Researching: {question}")
            answer = "insufficient data - no README available"
            results.append({"question": question, "answer": answer})

        return {"analysis": results, "matched_theme": "Unknown"}

    results = []
    for question in marketQuestions:
        print(f"Researching: {question}")
        answer = research_question(idea, question, llm, readme_info)
        results.append({"question": question, "answer": answer})
        print(f"Answer: {answer[:100]}...")

    # Theme matching
    theme_system = "Match this idea to one theme. Return ONLY the theme name."
    if BASE_PROMPT:
        theme_system = BASE_PROMPT + "\n\n" + theme_system

    theme_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", theme_system),
            ("human", "Themes: {themes}\nIdea: {idea}\nMatched Theme:"),
        ]
    )

    theme_chain = theme_prompt | llm | StrOutputParser()
    try:
        matched_theme = theme_chain.invoke({"themes": theme, "idea": idea})
        matched_theme = matched_theme.strip().split("\n")[0]  # Take first line only
    except:
        matched_theme = "General"

    return {"analysis": results, "matched_theme": matched_theme}


@router.get("/market-agent")
async def marketAgent_endpoint():
    """Test endpoint - returns available functionality"""
    return {
        "message": "Market Agent is running",
        "capabilities": [
            "Market analysis with web research",
            "Target audience identification",
            "Competitor analysis",
            "Theme matching",
            "Revenue potential analysis",
        ],
        "usage": {
            "endpoint": "POST /api/market-agent/analyze",
            "body": {
                "idea": "Your project idea description",
                "theme": "Comma-separated list of themes",
            },
        },
    }


@router.post("/market-agent/analyze")
async def market_agent_analyze(request: Request):
    """Analyze a market idea"""
    try:
        data = await request.json()
        idea = data.get("idea", "")
        theme = data.get("theme", "")

        if not idea:
            raise HTTPException(status_code=400, detail="Idea is required")

        if not theme:
            # Get theme from database if not provided
            try:
                conn = get_database_connection()
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute("SELECT theme FROM hackathons LIMIT 1")
                hackathon = cur.fetchone()
                cur.close()
                conn.close()
                theme = hackathon["theme"] if hackathon else "General"
            except:
                theme = "General"

        print(f"Analyzing idea: {idea}")
        print(f"Themes: {theme}")

        result = await analyze_market(idea, theme)

        return {
            "message": "Market analysis complete",
            "idea": idea,
            "matched_theme": result["matched_theme"],
            "analysis": result["analysis"],
        }

    except Exception as e:
        print(f"Error in market analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def fetch_readme(owner: str, repo: str) -> str:
    """Fetch README content from GitHub repository"""
    session = get_github_session()
    default_branch = get_default_branch(owner, repo)

    readme_names = [
        "README.md",
        "README.rst",
        "README.txt",
        "README",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "readme",
    ]

    for readme_name in readme_names:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{readme_name}?ref={default_branch}"
        response = session.get(url, headers=get_github_headers())

        if response.status_code == 200:
            data = response.json()
            content = data.get("content", "")
            encoding = data.get("encoding", "")

            if encoding == "base64" and content:
                try:
                    decoded = base64.b64decode(content).decode("utf-8")
                    return decoded.strip()
                except:
                    pass

    return ""


def get_github_session():
    """Create a requests session with retry logic"""
    session = Session()
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


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def get_github_headers():
    """Get headers for GitHub API requests"""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def get_default_branch(owner: str, repo: str) -> str:
    """Get the default branch of a repository"""
    session = get_github_session()
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = session.get(url, headers=get_github_headers())
    if response.status_code == 404:
        return "main"
    response.raise_for_status()
    return response.json().get("default_branch", "main")


def parse_repo_url(repo_url: str):
    """Extract owner and repo from URL"""
    owner, repo = None, None
    if repo_url:
        match = re.search(
            r"github\.com/([^/]+)/([^/?#]+?)(?:\.git)?(?:/|$|[?#])", repo_url
        )
        if match:
            owner = match.group(1)
            repo = match.group(2).replace(".git", "").split("/")[0]
    return owner, repo


async def invoke_market_agent(
    project_id: str, idea: str, github_link: str = None, hackathon_id: int = None
):
    """Background task for automatic project analysis"""
    try:
        criteria_text = ""
        if hackathon_id:
            conn = get_database_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT criteria FROM hackathons WHERE id = %s", (hackathon_id,)
            )
            hackathon = cur.fetchone()
            if hackathon:
                criteria_text = hackathon["criteria"] or ""
            cur.close()
            conn.close()

        readme_content = ""
        if github_link:
            owner, repo = parse_repo_url(github_link)
            if owner and repo:
                readme_content = fetch_readme(owner, repo)
                print(
                    f"Market Agent: Fetched README ({len(readme_content)} chars) from {owner}/{repo}"
                )

        result = await analyze_market(idea, "", readme_content)

        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE projects SET market_agent_analysis = %s WHERE project_id = %s",
            (json.dumps(result["analysis"]), project_id),
        )
        conn.commit()
        cur.close()
        conn.close()

        # Generate overall score after market analysis
        from agents.codeagent import generate_overall_project_score

        generate_overall_project_score(project_id)

        print(f"Market Agent: Analysis complete for project {project_id}")

    except Exception as e:
        print(f"Market Agent Error: {str(e)}")
