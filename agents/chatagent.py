from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import APIRouter, Request, HTTPException
from db import get_database_connection
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

def get_llm():
    return ChatOpenAI(
        model=os.getenv("FREE_LLM_MODEL", "liquid/lfm-2.5-1.2b-thinking:free"),
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

def get_relevant_context(vectorstore, query, k=5):
    """FIXED: Use invoke() instead of deprecated get_relevant_documents()"""
    if vectorstore is None:
        return ""
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Try the new invoke() method first
        try:
            docs = retriever.invoke(query)
        except AttributeError:
            # Fallback for older versions
            try:
                docs = retriever.get_relevant_documents(query)
            except:
                return ""
        
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

@router.get("/chat-agent")
async def chatAgent_endpoint():
    """Test endpoint - returns available functionality"""
    return {
        "message": "Chat Agent is running",
        "capabilities": [
            "AI-powered hackathon judging",
            "Code-aware conversations",
            "Market-aware responses",
            "Multi-turn conversations with history"
        ],
        "usage": {
            "simple_chat": "POST /api/chat-agent/simple",
            "project_chat": "POST /api/chat-agent"
        }
    }

@router.post("/chat-agent/simple")
async def simple_chat(request: Request):
    """Simple chat without project context"""
    try:
        data = await request.json()
        question = data.get("question", "")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        llm = get_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI hackathon judge assistant. Answer concisely in one paragraph, max 70 words."),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        
        return {
            "answer": answer,
            "chathistory": [
                {"input": question, "output": answer}
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-agent")
async def invoke_chat_agent(request: Request):
    """Chat with project context"""
    try:
        data = await request.json()
        
        # Get hackathon settings
        conn = get_database_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT criteria, theme, is_allowed FROM hackathons LIMIT 1")
        hackathon = cur.fetchone()
        
        if not hackathon or not hackathon.get("is_allowed", False):
            cur.close()
            conn.close()
            return {
                "answer": "Sorry, we have reached our credit limit or no hackathon is active.",
                "chathistory": data.get("chathistory", [])
            }
        
        theme = hackathon.get("theme", "")
        
        # Get project if project_id provided
        project_context = ""
        hackathon_criteria = ""
        project_description = ""
        if hackathon:
            hackathon_criteria = hackathon.get("criteria", "")
        
        if "project_id" in data and data["project_id"]:
            cur.execute("SELECT * FROM projects WHERE project_id = %s", (data["project_id"],))
            project = cur.fetchone()
            
            if project:
                project_description = project.get("short_description", "")
                project_context = project.get("code_agent_analysis", [])
        
        cur.close()
        conn.close()
        
        # Build conversation
        chatHistory = data.get("chathistory", [])
        history_context = ""
        if chatHistory:
            history_context = "Previous conversation:\n"
            for chat in chatHistory[-3:]:
                history_context += f"Human: {chat['input']}\nAI: {chat['output']}\n"
        
        llm = get_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hackathon judge AI, both market researcher and code reviewer.

Project: {description}
Theme: {theme}
Criteria: {hackathon_criteria}

Code Context: {code_context}

{history}

Rules:
1. Answer in one paragraph
2. Max 70 words
3. Be specific and analytical
4. Don't use formatting"""),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "description": project_description,
            "theme": theme,
            "hackathon_criteria": hackathon_criteria,
            "code_context": project_context[:1500] if project_context else "No code context available",
            "history": history_context if history_context else "No previous conversation",
            "question": data.get("question", "")
        })
        
        chatHistory.append({
            "input": data.get("question", ""),
            "output": answer
        })
        
        return {
            "answer": answer,
            "chathistory": chatHistory
        }
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")