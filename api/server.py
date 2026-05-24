"""FastAPI server for CEO Agent."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agent import CEOAgent

# ── Global agent instance ──────────────────────────

agent: CEOAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup if needed


app = FastAPI(title="AI Agent Company API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ─────────────────────────────────────────


class ConfigRequest(BaseModel):
    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    system_prompt: str | None = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ResearchRequest(BaseModel):
    task: str
    thread_id: str = "default"


# ── Endpoints ──────────────────────────────────────


@app.get("/")
def root():
    status = "ready" if agent else "not configured"
    return {"message": "AI Agent Company API", "status": status}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/config")
def config(req: ConfigRequest):
    global agent
    agent = CEOAgent(
        api_key=req.api_key,
        model=req.model,
        base_url=req.base_url,
        system_prompt=req.system_prompt,
    )
    return {"status": "ok", "message": "CEO Agent configured successfully"}


@app.post("/chat")
def chat(req: ChatRequest):
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not configured. POST /config first.")
    response = agent.chat(req.message, thread_id=req.thread_id)
    return {"response": response, "thread_id": req.thread_id}


@app.post("/research")
def research(req: ResearchRequest):
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not configured. POST /config first.")
    response = agent.chat(
        f"ค้นคว้าเรื่อง: {req.task}\n\nให้ค้นหาข้อมูลจากหลายแหล่ง แล้วสรุปเป็นภาษาไทย พร้อมอ้างอิงแหล่งที่มา",
        thread_id=req.thread_id,
    )
    return {"topic": req.task, "summary": response, "thread_id": req.thread_id}
