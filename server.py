from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from agents.marketagent import router as marketAgent_router
from agents.codeagent import router as codeAgent_router
from agents.chatagent import router as chatAgent_router
from agents.crudagent import router as crudAgent_router
from fastapi.middleware.cors import CORSMiddleware
from db import init_db
import os

load_dotenv()

init_db()

app = FastAPI(
    title="Hackathon API",
    description="API for hackathon project management and evaluation",
    version="1.0.0"
)

frontend_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []

if not frontend_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(marketAgent_router, prefix="/api", tags=["Market Agent"])
app.include_router(codeAgent_router, prefix="/api", tags=["Code Agent"])
app.include_router(chatAgent_router, prefix="/api", tags=["Chat Agent"])
app.include_router(crudAgent_router, prefix="/api", tags=["CRUD Agent"])


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
