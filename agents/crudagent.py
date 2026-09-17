from fastapi import APIRouter, Request, HTTPException
from fastapi import Query
from typing import Optional
from db import get_database_connection
from agents.codeagent import invoke_code_agent
import asyncio
from agents.marketagent import invoke_market_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import os
from psycopg2.extras import RealDictCursor
import uuid
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


def generate_overall_score(
    project_id: str,
    short_description: str,
    long_description: str,
    hackathon_name: str,
    hackathon_theme: str,
    criteria: str,
    code_analysis: list,
    market_analysis: list,
) -> tuple[float, str]:
    """Generate overall project score using LLM based on all factors"""
    llm = get_llm()

    code_summary = (
        "\n".join(
            [
                f"- {item.get('name', 'Criteria')}: {item.get('answer', 'N/A')[:200]}"
                for item in code_analysis
            ]
        )
        if code_analysis
        else "No code analysis available"
    )

    market_summary = (
        "\n".join(
            [
                f"- {item.get('question', 'Q')}: {item.get('answer', 'N/A')[:200]}"
                for item in market_analysis
            ]
        )
        if market_analysis
        else "No market analysis available"
    )

    system_prompt = """You are a hackathon judge evaluating projects.
Rate the project from 0-10 based on ALL these factors:
1. Theme alignment - Does it fit the hackathon theme?
2. Code quality - From the code analysis
3. Market potential - From the market analysis
4. Hackathon criteria - How well does it meet each criterion?

Return JSON format:
{{"score": <0-10>, "reasons": ["reason1", "reason2", "reason3"]}}

Keep reasons short (5-10 words each). Max 3 reasons."""
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
                """Project: {short_desc}
{long_desc}

Hackathon: {hackathon_name}
Theme: {hackathon_theme}
Criteria: {criteria}

CODE ANALYSIS:
{code_summary}

MARKET ANALYSIS:
{market_summary}

Evaluate and return JSON:""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        import json
        import re

        response = chain.invoke(
            {
                "short_desc": short_description[:500],
                "long_desc": long_description[:1000],
                "hackathon_name": hackathon_name,
                "hackathon_theme": hackathon_theme,
                "criteria": criteria or "General evaluation",
                "code_summary": code_summary,
                "market_summary": market_summary,
            }
        )

        # Parse JSON response - try multiple patterns
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if not json_match:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())
                score = float(result.get("score", 5)) / 10  # Convert 0-10 to 0-1
                reasons = result.get("reasons", ["No specific reasons provided"])
                if not isinstance(reasons, list):
                    reasons = [str(reasons)]
                explanation = "\n".join([f"- {r}" for r in reasons[:3]])
                return min(1.0, max(0.0, score)), explanation
            except json.JSONDecodeError:
                # Fallback: extract score from text
                score_match = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', response)
                if score_match:
                    score = float(score_match.group(1)) / 10
                    return min(1.0, max(0.0, score)), "- Score extracted from response"

    except Exception as e:
        print(f"Score generation error: {e}")

    return 0.5, "- Unable to generate detailed score explanation"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )


def semantic_search(query, documents, k=10):
    """Search documents using embeddings similarity"""
    if not documents:
        return []

    embeddings = get_embeddings()

    # Create temporary vectorstore for search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        for chunk in chunks:
            docs.append(Document(page_content=chunk))

    if not docs:
        return []

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=f"search_{uuid.uuid4().hex[:8]}",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)

    return [doc.page_content for doc in results]


@router.get("/crud-agent")
def crudAgent_endpoint():
    return {"message": "Hello from Crud Agent, Okay I'm not really an agent"}


@router.post(
    "/create-project",
    tags=["Projects"],
    summary="Create project and trigger AI analysis",
)
async def create_project(request: Request):
    """Create a new project. Triggers code and market analysis asynchronously.

    Body fields: shortDescription, longDescription, githubLink, demoLink (optional), theme, hackathonId (optional), projectType (optional)
    """
    data = await request.json()
    print(data)

    project_id = str(uuid.uuid4())
    hackathon_id = data.get("hackathonId")
    project_type = data.get("projectType", "OTHER")

    conn = get_database_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (project_id, hackathon_id, short_description, long_description, github_link, demo_link, theme, is_reviewed, project_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            project_id,
            hackathon_id,
            data.get("shortDescription", ""),
            data.get("longDescription", ""),
            data.get("githubLink", ""),
            data.get("demoLink", None),
            data.get("theme", ""),
            False,
            project_type,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()

    if hackathon_id:
        asyncio.create_task(
            invoke_market_agent(
                project_id,
                data.get("shortDescription", ""),
                data.get("githubLink", ""),
                hackathon_id,
            )
        )
        asyncio.create_task(
            invoke_code_agent(data.get("githubLink", ""), project_id, hackathon_id)
        )
    else:
        asyncio.create_task(
            invoke_market_agent(
                project_id,
                data.get("shortDescription", ""),
                data.get("githubLink", ""),
                None,
            )
        )
        asyncio.create_task(
            invoke_code_agent(data.get("githubLink", ""), project_id, None)
        )

    return {"message": "Project created", "project_id": project_id}


@router.post("/create-hackathon", tags=["Hackathons"], summary="Create a new hackathon")
async def create_hackathon(request: Request):
    """Create a new hackathon with evaluation criteria.

    Body fields:
    - name: Hackathon name
    - description: Description
    - theme: Theme (optional)
    - is_allowed: Whether submissions are allowed
    - criteria: Comma-separated criteria (e.g., "Code Quality, Innovation")
    - deadline: Deadline timestamp (optional)
    """
    data = await request.json()

    conn = get_database_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO hackathons (name, description, theme, is_allowed, criteria, deadline)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            data.get("name", ""),
            data.get("description", ""),
            data.get("theme", ""),
            data.get("isAllowed", False),
            data.get("criteria", ""),
            data.get("deadline", None),
        ),
    )
    hackathon_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return {"message": "Hackathon created", "hackathon_id": hackathon_id}


@router.get(
    "/get-hackathon/{hackathon_id}",
    tags=["Hackathons"],
    summary="Get hackathon details",
)
async def get_hackathon(hackathon_id: int):
    """Get details of a specific hackathon by ID."""
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM hackathons WHERE id = %s", (hackathon_id,))
    hackathon = cur.fetchone()
    cur.close()
    conn.close()

    if hackathon:
        hackathon["_id"] = str(hackathon["id"])
        del hackathon["id"]
        if hackathon.get("created_at"):
            hackathon["created_at"] = hackathon["created_at"].isoformat()
        if hackathon.get("deadline"):
            hackathon["deadline"] = hackathon["deadline"].isoformat()

    return {"message": "successful", "hackathon": hackathon}


@router.get("/get-all-hackathons", tags=["Hackathons"], summary="List all hackathons")
async def get_all_hackathons():
    """Get list of all hackathons ordered by creation date."""
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM hackathons ORDER BY created_at DESC")
    hackathons = cur.fetchall()
    cur.close()
    conn.close()

    result = []
    for h in hackathons:
        h["_id"] = str(h["id"])
        del h["id"]
        if h.get("created_at"):
            h["created_at"] = h["created_at"].isoformat()
        if h.get("deadline"):
            h["deadline"] = h["deadline"].isoformat()
        result.append(h)

    return {"message": "successful", "hackathons": result}


@router.get(
    "/get-hackathon-projects/{hackathon_id}",
    tags=["Projects"],
    summary="List projects in a hackathon",
)
async def get_hackathon_projects(hackathon_id: int):
    """Get all projects submitted to a specific hackathon."""
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT * FROM projects WHERE hackathon_id = %s ORDER BY created_at DESC",
        (hackathon_id,),
    )
    projects = cur.fetchall()
    cur.close()
    conn.close()

    result = []
    for p in projects:
        p["_id"] = str(p["id"])
        del p["id"]
        if p.get("created_at"):
            p["created_at"] = p["created_at"].isoformat()
        # Ensure demo_link is included (nullable)
        if "demo_link" not in p:
            p["demo_link"] = None
        # Ensure project_type is included
        if "project_type" not in p:
            p["project_type"] = "OTHER"
        result.append(p)

    return {"message": "successful", "projects": result}


@router.get(
    "/get-project-score/{project_id}", tags=["Scoring"], summary="Get project score"
)
async def get_project_score(project_id: str):
    """Get overall project score generated by AI based on all factors.

    Returns LLM-generated score with explanation of why that score was given.
    """
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute(
        """
        SELECT p.project_id, p.short_description, p.overall_score, p.score_explanation,
               h.name as hackathon_name, h.theme, h.criteria
        FROM projects p
        LEFT JOIN hackathons h ON p.hackathon_id = h.id
        WHERE p.project_id = %s
    """,
        (project_id,),
    )
    project = cur.fetchone()

    if not project:
        cur.close()
        conn.close()
        return {"message": "error", "error": "Project not found"}

    # If score hasn't been generated yet, trigger generation
    if project.get("overall_score") is None:
        cur.close()
        conn.close()
        from agents.codeagent import generate_overall_project_score

        generate_overall_project_score(project_id)

        # Re-fetch the project with score
        conn = get_database_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT p.project_id, p.overall_score, p.score_explanation,
                   h.name as hackathon_name
            FROM projects p
            LEFT JOIN hackathons h ON p.hackathon_id = h.id
            WHERE p.project_id = %s
        """,
            (project_id,),
        )
        project = cur.fetchone()

    cur.close()
    conn.close()

    overall_score = (
        float(project.get("overall_score", 0)) if project.get("overall_score") else 0
    )

    return {
        "message": "successful",
        "project_id": project_id,
        "hackathon_name": project.get("hackathon_name"),
        "overall_score": round(overall_score * 10, 1),
        "score_explanation": project.get(
            "score_explanation", "Score not yet generated"
        ),
        "score_breakdown": {
            "raw_score": overall_score,
            "out_of_10": round(overall_score * 10, 1),
        },
    }


@router.get(
    "/get-hackathon-leaderboard/{hackathon_id}",
    tags=["Scoring"],
    summary="Get hackathon leaderboard",
)
async def get_hackathon_leaderboard(hackathon_id: int):
    """Get ranked projects for a hackathon based on their overall scores."""
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT name FROM hackathons WHERE id = %s", (hackathon_id,))
    hackathon = cur.fetchone()
    if not hackathon:
        cur.close()
        conn.close()
        return {"message": "error", "error": "Hackathon not found"}

    cur.execute(
        """SELECT project_id, short_description, github_link, overall_score, score_explanation
                  FROM projects WHERE hackathon_id = %s""",
        (hackathon_id,),
    )
    projects = cur.fetchall()

    ranked = []
    for proj in projects:
        score = float(proj.get("overall_score", 0)) if proj.get("overall_score") else 0
        ranked.append(
            {
                "project_id": proj["project_id"],
                "short_description": proj["short_description"],
                "github_link": proj["github_link"],
                "score": round(score * 10, 1),  # Convert to 0-10 scale
                "score_explanation": proj.get("score_explanation", "")[:100],
            }
        )

    cur.close()
    conn.close()
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return {
        "message": "successful",
        "hackathon_name": hackathon["name"],
        "leaderboard": ranked,
    }


@router.get("/get-project/{project_id}")
async def get_project(project_id: str):
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM projects WHERE project_id = %s", (project_id,))
    project = cur.fetchone()
    cur.close()
    conn.close()

    if project:
        project["_id"] = str(project["id"])
        del project["id"]
        if "created_at" in project and project["created_at"]:
            project["created_at"] = project["created_at"].isoformat()
        # Ensure demo_link is included (nullable)
        if "demo_link" not in project:
            project["demo_link"] = None
        # Ensure project_type is included
        if "project_type" not in project:
            project["project_type"] = "OTHER"

    return {"message": "successful", "project": project}


@router.get("/get-all")
async def get_all_projects():
    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
    projects = cur.fetchall()
    cur.close()
    conn.close()

    final = []
    for project in projects:
        project["_id"] = str(project["id"])
        del project["id"]
        if "created_at" in project and project["created_at"]:
            project["created_at"] = project["created_at"].isoformat()
        # Ensure demo_link is included (nullable)
        if "demo_link" not in project:
            project["demo_link"] = None
        # Ensure project_type is included
        if "project_type" not in project:
            project["project_type"] = "OTHER"
        final.append(project)

    return {"message": "successful", "projects": final}


@router.post("/review")
async def review_project(request: Request):
    data = await request.json()
    project_id = data["project_id"]

    conn = get_database_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE projects SET is_reviewed = %s WHERE project_id = %s",
        (data["isReviewed"], project_id),
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"message": "successful", "project_id": project_id}


@router.post("/search")
async def search_projects(request: Request):
    data = await request.json()
    query = data.get("query", "")

    if not query:
        return {"message": "error", "error": "Query is required"}

    conn = get_database_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM projects")
    projects = cur.fetchall()
    cur.close()
    conn.close()

    if not projects:
        return {"message": "successful", "projects": []}

    # Prepare documents for semantic search
    formatted_projects = []
    documents = []

    for project in projects:
        project_dict = dict(project)
        project_dict["_id"] = str(project_dict["id"])
        del project_dict["id"]
        if "created_at" in project_dict and project_dict["created_at"]:
            project_dict["created_at"] = project_dict["created_at"].isoformat()
        formatted_projects.append(project_dict)

        # Create searchable text for each project
        search_text = f"""
        Project Description: {project_dict.get("long_description", "")}
        Short Description: {project_dict.get("short_description", "")}
        Theme: {project_dict.get("theme", "")}
        """
        documents.append(search_text)

    try:
        # Use semantic search with embeddings
        relevant_docs = semantic_search(query, documents, k=min(10, len(documents)))

        # Find matching projects
        results = []
        seen_ids = set()

        for doc in relevant_docs:
            for project in formatted_projects:
                if project["_id"] not in seen_ids:
                    search_text = f"""
                    Project Description: {project.get("long_description", "")}
                    Short Description: {project.get("short_description", "")}
                    Theme: {project.get("theme", "")}
                    """
                    if doc in search_text or any(
                        word in search_text for word in doc.split()[:5]
                    ):
                        results.append(project)
                        seen_ids.add(project["_id"])

        # If no results from semantic search, use LLM to rank projects
        if not results:
            llm = get_llm()
            system_prompt = """You are a project search engine. Rank the following projects by relevance to the query.
                Return only the project IDs in order of relevance, separated by commas. Example: "id1,id2,id3"

                Projects:
                {projects}

                Query: {query}

                Relevant project IDs (in order):"""
            if BASE_PROMPT:
                system_prompt = BASE_PROMPT + "\n\n" + system_prompt

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                ]
            )

            # Limit to first 10 projects for LLM processing
            projects_text = "\n".join(
                [
                    f"ID: {p['_id']} - {p.get('short_description', '')[:100]} - {p.get('theme', '')}"
                    for p in formatted_projects[:10]
                ]
            )

            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"projects": projects_text, "query": query})

            # Parse LLM response to get project IDs
            ranked_ids = [id.strip() for id in response.split(",")]

            for pid in ranked_ids:
                for project in formatted_projects:
                    if project["_id"] == pid:
                        results.append(project)
                        break

        return {"message": "successful", "projects": results}

    except Exception as e:
        # Fallback to simple text matching if semantic search fails
        print(f"Search error: {e}")
        results = []
        query_lower = query.lower()

        for project in formatted_projects:
            if (
                query_lower in project.get("short_description", "").lower()
                or query_lower in project.get("long_description", "").lower()
                or query_lower in project.get("theme", "").lower()
            ):
                results.append(project)

        return {"message": "successful", "projects": results[:10]}
