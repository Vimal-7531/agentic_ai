from typing import Annotated, TypedDict
import operator

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    user_query: str

    next: str

    agent_context: Annotated[list[str], operator.add]

    completed_workers: Annotated[list[str], operator.add]

    execution_trace: Annotated[list[dict], operator.add]

    final_response: str

