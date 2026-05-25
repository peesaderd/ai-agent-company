"""Monologue Agent (CEO) using LangGraph.

Human (Board) ←→ Mattermost ←→ CEO Agent (LangGraph)
                                    │
                          ┌─────────┼─────────┐
                     [Tool]     [Tool]     [Tool]
                    Search     Fetch     Analyze
"""

import json
import httpx
from typing import Literal, Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
import os

from tools.search_tool import search_web, fetch_page, search_and_fetch

# ── Tools ──────────────────────────────────────────


@tool
def search(query: str, max_results: int = 5) -> str:
    """Search the web for information on a topic. Returns list of titles, URLs and snippets."""
    results = search_web(query, max_results=max_results)
    if not results:
        return "ไม่พบผลการค้นหา"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}")
    return "\n\n".join(lines)


@tool
def fetch(url: str, max_length: int = 5000) -> str:
    """Fetch and extract the main text content from a webpage URL."""
    content = fetch_page(url, max_length=max_length)
    if content is None:
        return f"ไม่สามารถโหลดเนื้อหาจาก {url} ได้"
    return content


@tool
def research_topic(topic: str) -> str:
    """Research a topic by searching and fetching content from multiple sources."""
    results = search_and_fetch(topic, max_results=3, max_length=3000)
    if not results:
        return f"ไม่พบข้อมูลเกี่ยวกับ {topic}"

    output = f"ผลการค้นคว้า: {topic}\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- แหล่งที่ {i}: {r['title']} ---\n"
        output += f"URL: {r['url']}\n"
        output += f"เนื้อหา:\n{r.get('content', 'ไม่มีเนื้อหา')[:2000]}\n\n"
    return output

# ── Agent State ────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: str


# ── LLM Backends ──────────────────────────────────


class OllamaLLM:
    """Call Ollama API (local LLM).

    Uses prompt-based tool calling instead of structured tools parameter,
    because qwen2.5:7b is unreliable with Ollama's built-in tool calling.
    """

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """Call Ollama chat API.

        For Ollama, we do NOT pass the tools parameter because qwen2.5:7b
        is inconsistent with structured tool calling. Instead, tool instructions
        are embedded in the system prompt, and we parse tool calls from text.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7},
        }

        with httpx.Client(timeout=180) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                import logging
                logging.basicConfig(level=logging.ERROR)
                logger = logging.getLogger(__name__)
                logger.error(f"Ollama API error {resp.status_code}: {resp.text[:500]}")
                logger.error(f"Request payload (truncated): {json.dumps(payload, ensure_ascii=False)[:500]}")
            resp.raise_for_status()
            return resp.json()

    def parse_response(self, data: dict) -> tuple[str, list[dict]]:
        """Extract content and tool_calls from Ollama response.

        Handles two cases:
        1. Structured tool_calls from Ollama (when it works)
        2. Tool calls embedded in text content (prompt-based, more reliable)
        """
        msg = data.get("message", {})
        content = msg.get("content", "") or ""

        # Case 1: Check for structured tool_calls from Ollama
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append({
                "id": tc.get("id", f"call_{hash(str(fn))}"),
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": json.dumps(fn.get("arguments", {})),
                },
            })

        if tool_calls:
            return content, tool_calls

        # Case 2: Parse tool calls from text content (prompt-based)
        # Look for JSON block with tool call info
        text_tool_calls = self._parse_tool_calls_from_text(content)
        if text_tool_calls:
            # Remove the tool call JSON from the content
            content = self._strip_tool_calls_from_text(content)
            return content, text_tool_calls

        return content, []

    def _parse_tool_calls_from_text(self, text: str) -> list[dict]:
        """Parse tool calls embedded in text content."""
        import re
        tool_calls = []

        # Strategy 1: Try to parse the entire text as JSON (model sometimes outputs only JSON)
        try:
            tc_data = json.loads(text.strip())
            name = tc_data.get("name", "")
            args = tc_data.get("arguments", {})
            if name and args:
                return [{
                    "id": f"call_{hash(text)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    },
                }]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Strategy 2: Find ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                tc_data = json.loads(block)
                name = tc_data.get("name", "")
                args = tc_data.get("arguments", {})
                if name and args:
                    tool_calls.append({
                        "id": f"call_{hash(block)}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        if tool_calls:
            return tool_calls

        # Strategy 3: Find <tool_call> ... </tool_call> blocks
        tool_blocks = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        for block in tool_blocks:
            try:
                tc_data = json.loads(block)
                name = tc_data.get("name", "")
                args = tc_data.get("arguments", {})
                if name and args:
                    tool_calls.append({
                        "id": f"call_{hash(block)}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        return tool_calls

    def _strip_tool_calls_from_text(self, text: str) -> str:
        """Remove tool call JSON blocks from text content."""
        import re
        # If the entire text is a JSON tool call (Strategy 1 case), return empty
        try:
            tc_data = json.loads(text.strip())
            if tc_data.get("name") and tc_data.get("arguments"):
                return ""
        except (json.JSONDecodeError, ValueError):
            pass
        # Remove ```json ... ``` blocks
        text = re.sub(r'```json[\s\S]*?```', '', text)
        # Remove <tool_call> ... </tool_call> blocks
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text


class DeepseekLLM:
    """Call Deepseek API via httpx."""

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash", base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """Call Deepseek chat API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def parse_response(self, data: dict) -> tuple[str, list[dict]]:
        """Extract content and tool_calls from Deepseek response."""
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""

        tool_calls = []
        if msg.get("tool_calls"):
            tool_calls = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in msg["tool_calls"]
            ]

        return content, tool_calls


# ── CEO Agent ──────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """คุณคือ CEO ของ AI Agent Company เป็นผู้บริหารสูงสุดที่คอยดูแลองค์กร

คุณมีลูกน้องที่เป็น AI Agents ที่คอยทำงานต่างๆ ให้:
- Research Agent: ค้นคว้าและรวบรวมข้อมูล
- Content Agent: สร้างและสรุปเนื้อหา
- Data Agent: วิเคราะห์ข้อมูลและแนวโน้ม

คุณสามารถใช้ tools ต่างๆ เพื่อทำงานได้โดยตรง:
- search: ค้นหาข้อมูลจากเว็บ (ใช้ query และ max_results)
- fetch: ดึงเนื้อหาจาก URL (ใช้ url และ max_length)
- research_topic: ค้นคว้าหัวข้อแบบละเอียด (ใช้ topic)

วิธีการทำงาน:
1. รับคำสั่งจาก Human (กรรมการ)
2. คิดและวางแผน (monologue) ว่าจะทำอะไร
3. ถ้าต้องใช้ tool ให้แสดงเป็น JSON ในรูปแบบนี้:
   ```json
   {"name": "search", "arguments": {"query": "คำค้นหา", "max_results": 5}}
   ```
4. สรุปและรายงานกลับ Human

ตอบเป็นภาษาไทย เว้นแต่ Human จะขอเป็นภาษาอื่น"""

OLLAMA_SYSTEM_PROMPT = """คุณคือ CEO ของ AI Agent Company เป็นผู้บริหารสูงสุดที่คอยดูแลองค์กร

คุณมีลูกน้องที่เป็น AI Agents ที่คอยทำงานต่างๆ ให้:
- Research Agent: ค้นคว้าและรวบรวมข้อมูล
- Content Agent: สร้างและสรุปเนื้อหา
- Data Agent: วิเคราะห์ข้อมูลและแนวโน้ม

คุณสามารถใช้ tools ต่างๆ เพื่อทำงานได้โดยตรง:
- search(query, max_results): ค้นหาข้อมูลจากเว็บ
- fetch(url, max_length): ดึงเนื้อหาจาก URL
- research_topic(topic): ค้นคว้าหัวข้อแบบละเอียด

วิธีการทำงาน:
1. รับคำสั่งจาก Human (กรรมการ)
2. คิดและวางแผน (monologue) ว่าจะทำอะไร
3. ถ้าต้องการใช้ tool ให้พิมพ์ JSON ในรูปแบบนี้:
   ```json
   {"name": "search", "arguments": {"query": "คำค้นหา", "max_results": 5}}
   ```
   แล้วฉันจะจัดการเรียก tool ให้เอง
4. สรุปและรายงานกลับ Human

สำคัญมาก: ถ้าต้องการใช้ tool ให้ใส่ JSON ใน ```json ... ``` block เท่านั้น
ตอบเป็นภาษาไทย เว้นแต่ Human จะขอเป็นภาษาอื่น"""

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch and extract text content from a webpage URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Webpage URL"},
                    "max_length": {"type": "integer", "description": "Max content length (default 5000)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_topic",
            "description": "Research a topic by searching and fetching from multiple sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to research"},
                },
                "required": ["topic"],
            },
        },
    },
]

TOOL_MAP = {
    "search": search,
    "fetch": fetch,
    "research_topic": research_topic,
}


class CEOAgent:
    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        system_prompt: str = None,
        llm_backend: str = "deepseek",
        ollama_model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm_backend = llm_backend

        # Init LLM backend
        if llm_backend == "ollama":
            self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
        else:
            self.llm = DeepseekLLM(api_key=api_key, model=model, base_url=base_url)

        # Build graph
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", self._call_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.memory)

    def _call_agent(self, state: AgentState):
        # Convert LangChain messages to API format
        system_prompt = self.system_prompt
        if self.llm_backend == "ollama":
            system_prompt = OLLAMA_SYSTEM_PROMPT

        msgs = [{"role": "system", "content": system_prompt}]
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                            },
                        }
                        for tc in m.tool_calls
                    ]
                msgs.append(msg)
            elif isinstance(m, ToolMessage):
                msgs.append({"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id})

        # Call LLM
        # For Ollama, don't pass tools parameter (uses prompt-based tool calling)
        tools_arg = None if self.llm_backend == "ollama" else TOOL_SCHEMA
        data = self.llm.chat(msgs, tools=tools_arg)
        content, tool_calls = self.llm.parse_response(data)

        ai_msg = AIMessage(content=content)

        if tool_calls:
            ai_msg.tool_calls = [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                }
                for tc in tool_calls
            ]

        return {"messages": [ai_msg]}

    def _call_tools(self, state: AgentState):
        last_msg = state["messages"][-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"messages": []}

        tool_messages = []
        for tc in last_msg.tool_calls:
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return {"messages": tool_messages}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "continue"
        return "end"

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Send a message to the CEO agent and get response."""
        config = {"configurable": {"thread_id": thread_id}}
        state = {"messages": [HumanMessage(content=message)]}
        result = self.graph.invoke(state, config=config)
        return result["messages"][-1].content

    def chat_stream(self, message: str, thread_id: str = "default"):
        """Stream response from the CEO agent."""
        config = {"configurable": {"thread_id": thread_id}}
        state = {"messages": [HumanMessage(content=message)]}
        for event in self.graph.stream(state, config=config):
            for node, value in event.items():
                if node == "agent" and value.get("messages"):
                    msg = value["messages"][-1]
                    if hasattr(msg, "content") and msg.content:
                        yield msg.content
