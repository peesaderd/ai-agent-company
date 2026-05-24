"""FastAPI server for CEO Agent with Chat UI."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agent import CEOAgent

# ── Global agent instance ──────────────────────────

agent: CEOAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        agent = CEOAgent(
            api_key=api_key,
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    yield


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
    api_key: str | None = None
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    system_prompt: str | None = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ResearchRequest(BaseModel):
    task: str
    thread_id: str = "default"


# ── Chat UI ────────────────────────────────────────

CHAT_UI = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Agent Company - CEO Chat</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
  .header{background:linear-gradient(135deg,#1e293b,#334155);padding:16px 24px;border-bottom:1px solid #475569;display:flex;align-items:center;gap:12px;flex-shrink:0}
  .header h1{font-size:18px;font-weight:600;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header .badge{background:#22c55e;color:#fff;font-size:11px;padding:2px 10px;border-radius:10px;-webkit-text-fill-color:#fff}
  .chat-container{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}
  .message{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.6;font-size:14px;animation:fadeIn .3s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .user{background:#2563eb;align-self:flex-end;border-bottom-right-radius:4px}
  .bot{background:#1e293b;align-self:flex-start;border-bottom-left-radius:4px;border:1px solid #334155}
  .bot h3{color:#60a5fa;margin:8px 0 4px;font-size:13px}
  .bot h3:first-child{margin-top:0}
  .bot ul,.bot ol{margin:4px 0;padding-left:20px}
  .bot li{margin:2px 0}
  .bot table{border-collapse:collapse;margin:8px 0;width:100%;font-size:13px}
  .bot td,.bot th{border:1px solid #475569;padding:6px 10px;text-align:left}
  .bot th{background:#334155;color:#93c5fd}
  .bot code{background:#0f172a;padding:1px 6px;border-radius:4px;font-size:13px;color:#fbbf24}
  .bot strong{color:#f1f5f9}
  .bot em{color:#94a3b8}
  .typing{display:flex;gap:4px;padding:4px 0}
  .typing span{width:8px;height:8px;background:#64748b;border-radius:50%;animation:bounce 1.4s infinite}
  .typing span:nth-child(2){animation-delay:.2s}
  .typing span:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
  .input-area{background:#1e293b;border-top:1px solid #334155;padding:16px 24px;flex-shrink:0}
  .input-row{display:flex;gap:12px;max-width:900px;margin:0 auto}
  .input-row input{flex:1;background:#0f172a;border:1px solid #475569;border-radius:10px;padding:12px 16px;color:#e2e8f0;font-size:14px;outline:none;transition:border-color .2s}
  .input-row input:focus{border-color:#60a5fa}
  .input-row input::placeholder{color:#64748b}
  .input-row button{background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;border:none;border-radius:10px;padding:12px 24px;font-size:14px;font-weight:600;cursor:pointer;transition:opacity .2s;white-space:nowrap}
  .input-row button:hover{opacity:.9}
  .input-row button:disabled{opacity:.5;cursor:not-allowed}
  .error{background:#7f1d1d;color:#fca5a5;align-self:center;padding:8px 16px;border-radius:8px;font-size:13px}
  .welcome{text-align:center;padding:40px 20px;color:#94a3b8}
  .welcome h2{font-size:22px;color:#e2e8f0;margin-bottom:8px}
  .welcome p{font-size:14px;line-height:1.8}
  .welcome .agents{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:16px}
  .welcome .agent-card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 16px;text-align:center;min-width:120px}
  .welcome .agent-card .icon{font-size:24px}
  .welcome .agent-card .name{font-size:12px;color:#94a3b8;margin-top:4px}
  @media(max-width:600px){.message{max-width:90%}.header h1{font-size:15px}.input-area{padding:12px 16px}}
</style>
</head>
<body>
<div class="header">
  <h1>🤖 AI Agent Company</h1>
  <span class="badge">CEO Online</span>
</div>
<div class="chat-container" id="chatContainer">
  <div class="welcome" id="welcome">
    <h2>👋 สวัสดีครับ ท่านกรรมการ</h2>
    <p>ผมคือ <strong>CEO</strong> ของ AI Agent Company<br>พร้อมให้บริการคุณด้วยทีมงาน AI Agents</p>
    <div class="agents">
      <div class="agent-card"><div class="icon">🔍</div><div class="name">Research Agent</div></div>
      <div class="agent-card"><div class="icon">✍️</div><div class="name">Content Agent</div></div>
      <div class="agent-card"><div class="icon">📊</div><div class="name">Data Agent</div></div>
    </div>
    <p style="margin-top:20px;font-size:13px;color:#64748b">พิมพ์ข้อความด้านล่างเพื่อเริ่มพูดคุย</p>
  </div>
</div>
<div class="input-area">
  <div class="input-row">
    <input type="text" id="messageInput" placeholder="พิมพ์ข้อความ..." autofocus>
    <button id="sendBtn">ส่ง</button>
  </div>
</div>
<script>
const chatContainer=document.getElementById('chatContainer');
const welcome=document.getElementById('welcome');
const input=document.getElementById('messageInput');
const sendBtn=document.getElementById('sendBtn');
let threadId='web-'+(Date.now().toString(36));

function addMessage(text,role){
  welcome?.remove();
  const div=document.createElement('div');
  div.className='message '+(role==='user'?'user':'bot');
  if(role==='bot'){
    div.innerHTML=markedToHtml(text);
  }else{
    div.textContent=text;
  }
  chatContainer.appendChild(div);
  chatContainer.scrollTop=chatContainer.scrollHeight;
}

function markedToHtml(text){
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/```(\w*)\n([\s\S]*?)```/g,'<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/### (.+)/g,'<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/^- (.+)/gm,'<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g,'<ul>$1</ul>')
    .replace(/<\/ul>\s*<ul>/g,'')
    .replace(/\|(.+)\|/g,function(m){return m.includes('---')?'':m})
    .replace(/\|(.+)\|/g,function(m){
      const cells=m.slice(1,-1).split('|').map(c=>c.trim());
      return '<tr><td>'+cells.join('</td><td>')+'</td></tr>';
    })
    .replace(/(<tr>.*<\/tr>)/g,'<table>$1</table>')
    .replace(/<\/table>\s*<table>/g,'')
    .replace(/\n\n/g,'<br><br>')
    .replace(/\n/g,'<br>');
}

function showTyping(){
  const div=document.createElement('div');
  div.className='message bot';
  div.id='typingIndicator';
  div.innerHTML='<div class="typing"><span></span><span></span><span></span></div>';
  chatContainer.appendChild(div);
  chatContainer.scrollTop=chatContainer.scrollHeight;
}

function hideTyping(){
  const el=document.getElementById('typingIndicator');
  if(el)el.remove();
}

async function sendMessage(){
  const text=input.value.trim();
  if(!text)return;
  input.value='';
  sendBtn.disabled=true;
  addMessage(text,'user');
  showTyping();
  try{
    const res=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,thread_id:threadId})
    });
    const data=await res.json();
    hideTyping();
    if(data.response){
      addMessage(data.response,'bot');
    }else if(data.detail){
      addMessage('⚠️ '+data.detail,'bot');
    }
  }catch(e){
    hideTyping();
    addMessage('⚠️ ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้ กรุณาลองใหม่อีกครั้ง','bot');
  }
  sendBtn.disabled=false;
  input.focus();
}

sendBtn.addEventListener('click',sendMessage);
input.addEventListener('keydown',e=>{if(e.key==='Enter')sendMessage()});
</script>
</body>
</html>"""


# ── Endpoints ──────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def root():
    return CHAT_UI


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/config")
def config(req: ConfigRequest):
    global agent
    api_key = req.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide in request body or set DEEPSEEK_API_KEY env var.",
        )
    agent = CEOAgent(
        api_key=api_key,
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
