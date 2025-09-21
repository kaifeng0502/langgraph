"""Simple internal helpdesk assistant built with LangGraph.

This script demonstrates how to assemble a reusable helpdesk agent that
answers employee IT questions by combining LangGraph's prebuilt
ReAct-style agent, lightweight Python tools, and an in-memory
checkpointer for conversational memory.

The example walks through how to:

* register Python functions as tools with human-readable descriptions
* enable stateful memory so repeated queries on the same ``thread_id``
  share history
* inspect the stored conversation after handling multiple questions

Setup
-----
1. Install dependencies::

       pip install -U langgraph "langchain-openai"

2. Provide an API key for the chat model (e.g. OpenAI)::

       export OPENAI_API_KEY="sk-..."

3. (Optional) Override the model with ``HELPDESK_MODEL`` if you prefer
   a different provider::

       export HELPDESK_MODEL="anthropic:claude-3-5-sonnet"

4. Run the script::

       python simple_internal_helpdesk.py

If you do not have model credentials configured the script exits
gracefully after printing a reminder.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Checkpointer, StateSnapshot

# ---------------------------------------------------------------------------
# Internal knowledge sources and lightweight tools
# ---------------------------------------------------------------------------

COMPANY_POLICIES: dict[str, str] = {
    "vpn": "员工需使用公司提供的 VPN 客户端，密码每 90 天自动过期。",
    "wifi": "访客可连接 Guest Wi-Fi，员工请使用 Corp Wi-Fi 并开启 802.1X 认证。",
    "hardware": "临时借用笔记本请提交 IT 工单，审批通过后 2 个工作日内领取。",
}

DEFAULT_THREAD_ID = "employee-1001"
DEFAULT_QUESTIONS = (
    "公司 VPN 的密码策略是什么？",
    "我的电脑网卡坏了，可以帮我升级处理吗？",
)
_MEMORY = InMemorySaver()


@tool
def lookup_policy(topic: str) -> str:
    """查询指定主题的内部政策摘要。"""

    topic = topic.lower().strip()
    if topic in COMPANY_POLICIES:
        return COMPANY_POLICIES[topic]
    return "暂无该主题的正式条款，请联系 IT 团队确认，或在知识库中创建新的 FAQ。"


@tool
def escalate_ticket(summary: str) -> str:
    """生成可直接复制到工单系统的升级请求模板。"""

    return (
        "新建 IT 工单：\n"
        f"- 摘要: {summary}\n"
        "- 影响范围: 员工\n"
        "- 优先级: 中\n"
        "- 指派给: helpdesk@internal\n"
    )


# ---------------------------------------------------------------------------
# Agent factory and helpers
# ---------------------------------------------------------------------------

def create_helpdesk_agent(
    *,
    model: str | None = None,
    checkpointer: Checkpointer | None = None,
) -> CompiledStateGraph:
    """Create the reusable helpdesk agent instance.

    Args:
        model: Optional model identifier. Defaults to the ``HELPDESK_MODEL``
            environment variable or ``"openai:gpt-4o-mini"``.
        checkpointer: Optional LangGraph checkpointer. When omitted, an
            in-memory saver is used so that repeated calls in the same
            process share conversation state.
    """

    selected_model = model or os.getenv("HELPDESK_MODEL", "openai:gpt-4o-mini")
    selected_checkpointer = checkpointer if checkpointer is not None else _MEMORY

    prompt = (
        "你是公司的 IT 数字员工，擅长回答内部流程与政策问题。"
        "遇到流程或政策问题时，请调用 lookup_policy 工具；"
        "如需升级为人工处理，请调用 escalate_ticket 并告知员工。"
        "所有回复均使用简洁专业的中文。"
    )

    return create_react_agent(
        model=selected_model,
        tools=[lookup_policy, escalate_ticket],
        prompt=prompt,
        checkpointer=selected_checkpointer,
    )


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content")
                if text:
                    parts.append(str(text))
                else:
                    parts.append(str(chunk))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    return str(content)


def _format_messages(messages: Iterable[Any]) -> Iterable[str]:
    for message in messages:
        role = getattr(message, "type", getattr(message, "role", "message"))
        content = _normalize_content(getattr(message, "content", message))
        yield f"{role}: {content}"


def _print_memory_snapshot(agent: CompiledStateGraph, config: dict[str, Any]) -> None:
    snapshot: StateSnapshot | None
    try:
        snapshot = agent.get_state(config)
    except Exception:
        snapshot = None

    if not snapshot or not isinstance(snapshot.values, dict):
        return

    messages = snapshot.values.get("messages", [])
    if not messages:
        return

    print(
        "会话记忆快照（thread_id = {thread}）:".format(
            thread=config["configurable"]["thread_id"]
        )
    )
    for line in _format_messages(messages):
        print(f"  {line}")


def run_demo(agent: CompiledStateGraph, *, thread_id: str = DEFAULT_THREAD_ID) -> None:
    """Simulate two requests from the same employee thread."""

    config = {"configurable": {"thread_id": thread_id}}

    for question in DEFAULT_QUESTIONS:
        print(f"员工: {question}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
        ai_message = response["messages"][-1]
        print(f"助手: {_normalize_content(ai_message.content)}\n")

    _print_memory_snapshot(agent, config)


def main() -> None:
    model = os.getenv("HELPDESK_MODEL", "openai:gpt-4o-mini")

    if model.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        print(
            "⚠️  未检测到 OPENAI_API_KEY，无法调用 OpenAI 模型。"
            "如需使用其他服务商，请设置 HELPDESK_MODEL 并提供对应的凭据。"
        )
        return

    agent = create_helpdesk_agent(model=model)
    run_demo(agent)


if __name__ == "__main__":
    main()
