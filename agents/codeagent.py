from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from fastapi import APIRouter, Request, HTTPException
import asyncio
import re
from db import get_database_connection
from psycopg2.extras import RealDictCursor
import json
import os
import uuid
from gitingest import ingest
from dotenv import load_dotenv

load_dotenv()

BASE_PROMPT = os.getenv("BASE_PROMPT", "").strip()

router = APIRouter()


def get_llm():
    return ChatOpenAI(
        model=os.getenv("FREE_LLM_MODEL", "liquid/lfm-2.5-1.2b-thinking:free"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        api_key=os.getenv("LLM_API_KEY"),
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
    """Analyze a GitHub repository using gitingest for content retrieval.
    Returns: (analysis_results, number_of_chunks)
    """
    print(f"Fetching repo: {repo_url} (type: {project_type})")

    try:
        summary, tree, content = ingest(repo_url)
    except Exception as e:
        raise ValueError(f"Failed to fetch repository: {e}")

    if not content or len(content.strip()) < 50:
        return ([], 0)

    print(f"Ingested repo, {len(content)} chars")

    document = Document(
        page_content=content,
        metadata={"source": repo_url},
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([document])
    print(f"Created {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"repo_{uuid.uuid4().hex[:8]}",
    )

    llm = get_llm()
    results = []
    for question in questions:
        print(f"Analyzing: {question}")
        answer = query_codebase(vectorstore, question, llm)
        results.append({"question": question, "answer": answer})
        print(f"Answer: {answer[:100]}...")

    return (results, len(chunks))


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

        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")

        print(f"Analyzing repository: {repo_url}")

        questions = [
            "What technologies and programming languages are used?",
            "Explain the project structure and purpose",
            "How is the code quality?",
            "What dependencies and libraries are used?",
        ]

        results, num_chunks = analyze_repository(repo_url, questions)

        if num_chunks == 0:
            return {"message": "No code files found in repository", "analysis": []}

        return {
            "message": "Code analysis complete",
            "repo_url": repo_url,
            "chunks_analyzed": num_chunks,
            "analysis": results,
        }

    except ValueError as e:
        print(f"Validation error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error in code analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


async def invoke_code_agent(repolink: str, project_id: str, hackathon_id: int = None):
    """Background task for automatic code analysis via gitingest"""
    try:
        criteria_text = ""
        project_description = ""
        hackathon_name = ""

        conn = get_database_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            "SELECT short_description, long_description, hackathon_id FROM projects WHERE project_id = %s",
            (project_id,),
        )
        project = cur.fetchone()
        if project:
            project_description = f"{project.get('short_description', '')} {project.get('long_description', '')}".strip()

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

        print(f"Fetching repo content: {repolink}")

        try:
            summary, tree, content = ingest(repolink)
        except Exception as e:
            print(f"Failed to fetch repository: {e}")
            save_evaluation(
                project_id,
                "Repo Validation",
                0,
                f"Failed to fetch repository: {str(e)}",
                "code",
            )
            return

        if not content or len(content.strip()) < 50:
            save_evaluation(
                project_id,
                "Code Quality",
                0,
                "No code files found in repository",
                "code",
            )
            return

        print(f"Ingested repo, {len(content)} chars")

        document = Document(
            page_content=content,
            metadata={"source": repolink},
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300
        )
        chunks = text_splitter.split_documents([document])

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

            if name.lower() == "innovation":
                answer = assess_innovation(llm, project_description, hackathon_name)
                score = extract_score_from_text(answer)

            elif name.lower() == "tech stack" or "tech" in name.lower():
                answer = query_codebase(
                    vectorstore,
                    "What technologies, frameworks, libraries, and programming languages are used in this project? List all dependencies, frameworks, and tools found in the code.",
                    llm,
                )
                score = 1.0 if answer and answer != "No code found to analyze" else 0.5

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

        generate_overall_project_score(project_id)

        print(f"Code Agent: Analysis complete for project {project_id}")

    except Exception as e:
        print(f"Code Agent Error: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_overall_project_score(project_id: str):
    """Generate and save overall project score based on all analyses"""
    try:
        from agents.crudagent import generate_overall_score

        conn = get_database_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

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
