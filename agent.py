"""Monologue Agent (CEO) using LangGraph.

Human (Board) ←→ Rocket.chat ←→ CEO Agent (LangGraph)
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
from langgraph.prebuilt import ToolNode
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


# ── CEO Agent ──────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """คุณคือ CEO ของ AI Agent Company เป็นผู้บริหารสูงสุดที่คอยดูแลองค์กร

คุณมีลูกน้องที่เป็น AI Agents ที่คอยทำงานต่างๆ ให้:
- Research Agent: ค้นคว้าและรวบรวมข้อมูล
- Content Agent: สร้างและสรุปเนื้อหา
- Data Agent: วิเคราะห์ข้อมูลและแนวโน้ม

คุณสามารถใช้ tools ต่างๆ เพื่อทำงานได้โดยตรง:
- search: ค้นหาข้อมูลจากเว็บ
- fetch: ดึงเนื้อหาจาก URL
- research_topic: ค้นคว้าหัวข้อแบบละเอียด

วิธีการทำงาน:
1. รับคำสั่งจาก Human (กรรมการ)
2. คิดและวางแผน (monologue) ว่าจะทำอะไร
3. ใช้ tools ที่จำเป็น
4. สรุปและรายงานกลับ Human

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
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        system_prompt: str = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

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

    def _call_deepseek(self, messages: list[dict]) -> dict:
        """Call Deepseek API directly via httpx (bypasses OpenAI client issues with reasoning_content)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_SCHEMA,
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    def _call_agent(self, state: AgentState):
        # Convert LangChain messages to Deepseek API format
        msgs = [{"role": "system", "content": self.system_prompt}]
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content or ""}
                # Preserve reasoning_content if present (Deepseek thinking mode)
                rc = m.additional_kwargs.get("reasoning_content")
                if rc:
                    msg["reasoning_content"] = rc
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

        # Call Deepseek API
        data = self._call_deepseek(msgs)
        choice = data["choices"][0]
        msg = choice["message"]

        # Convert response to AIMessage
        content = msg.get("content") or ""
        ai_msg = AIMessage(content=content)

        # Store reasoning_content for Deepseek thinking mode
        if msg.get("reasoning_content"):
            ai_msg.additional_kwargs["reasoning_content"] = msg["reasoning_content"]

        # Handle tool calls
        if msg.get("tool_calls"):
            ai_msg.tool_calls = [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                }
                for tc in msg["tool_calls"]
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
