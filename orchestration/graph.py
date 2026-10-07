import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END

from orchestration.state import AgentState


# ==================================================
# ENVIRONMENT
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


# ==================================================
# ROUTING MODEL
# ==================================================

class RouteDecision(BaseModel):
    next: Literal[
        "policy_rag",
        "network_analytics",
        "network_diagnostics_adk",
        "billing_resolution_adk",
        "customer_comms_crew",
        "FINISH",
    ]


model_name = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5",
)

anthropic_key = os.getenv(
    "ANTHROPIC_API_KEY"
)


supervisor_llm = ChatAnthropic(
    model=model_name,
    anthropic_api_key=anthropic_key,
    max_tokens=300,
)


supervisor_router = supervisor_llm.with_structured_output(
    RouteDecision
)


# ==================================================
# FALLBACK ROUTING
# ==================================================

def fallback_route(
    user_query: str,
    completed: list[str],
) -> str:
    """
    Deterministic backup router.

    Used when:
    - Supervisor LLM is unavailable
    - API credits are exhausted
    - LLM selects an already completed worker

    The router can execute multiple specialists
    sequentially before CustomerCommsCrew.
    """

    query = user_query.lower()


    # --------------------------------------------------
    # FINISH
    # --------------------------------------------------

    if "customer_comms_crew" in completed:
        return "FINISH"


    # --------------------------------------------------
    # COMBINED OUTAGE + SLA / POLICY QUERY
    # --------------------------------------------------

    outage_terms = [
        "outage",
        "downtime",
        "service disruption",
        "service interruption",
    ]

    sla_terms = [
        "sla",
        "eligible",
        "eligibility",
        "policy",
        "compensation",
    ]

    is_outage_query = any(
        term in query
        for term in outage_terms
    )

    is_sla_query = any(
        term in query
        for term in sla_terms
    )


    if is_outage_query and is_sla_query:

        if (
            "network_analytics"
            not in completed
        ):
            return "network_analytics"

        if (
            "policy_rag"
            not in completed
        ):
            return "policy_rag"

        return "customer_comms_crew"


    # --------------------------------------------------
    # BILLING
    # --------------------------------------------------

    billing_terms = [
        "bill",
        "billing",
        "charge",
        "charged",
        "duplicate",
        "credit",
        "dispute",
        "refund",
        "invoice",
    ]

    if (
        any(
            term in query
            for term in billing_terms
        )
        and "billing_resolution_adk"
        not in completed
    ):
        return "billing_resolution_adk"


    # --------------------------------------------------
    # NETWORK ANALYTICS / SQL
    # --------------------------------------------------

    analytics_terms = [
        "packet loss",
        "latency",
        "throughput",
        "performance",
        "network summary",
        "regional summary",
        "summarize the",
        "region",
        "outage duration",
        "6-hour outage",
        "hours outage",
        "current balance",
        "latest",
    ]

    if (
        any(
            term in query
            for term in analytics_terms
        )
        and "network_analytics"
        not in completed
    ):
        return "network_analytics"


    # --------------------------------------------------
    # NETWORK DIAGNOSTICS
    # --------------------------------------------------

    network_terms = [
        "tower",
        "5g",
        "network issue",
        "network problem",
        "session drop",
        "session drops",
        "connectivity",
        "signal",
        "incident",
        "connection",
        "internet issue",
    ]

    if (
        any(
            term in query
            for term in network_terms
        )
        and "network_diagnostics_adk"
        not in completed
    ):
        return "network_diagnostics_adk"


    # --------------------------------------------------
    # POLICY RAG
    # --------------------------------------------------

    policy_terms = [
        "policy",
        "roaming",
        "travel pass",
        "sla",
        "eligible",
        "eligibility",
        "japan",
        "canada",
        "mexico",
        "europe",
        "upgrade",
        "faq",
        "coverage",
    ]

    if (
        any(
            term in query
            for term in policy_terms
        )
        and "policy_rag"
        not in completed
    ):
        return "policy_rag"


    # --------------------------------------------------
    # FINAL CUSTOMER RESPONSE
    # --------------------------------------------------

    return "customer_comms_crew"


# ==================================================
# SUPERVISOR NODE
# ==================================================

def supervisor_node(
    state: AgentState,
):
    user_query = state[
        "user_query"
    ]

    completed = state.get(
        "completed_workers",
        [],
    )

    context = state.get(
        "agent_context",
        [],
    )

    # --------------------------------------------------
    # FAST ROUTING
    # --------------------------------------------------

    # If customer communications already ran,
    # finish immediately without another LLM call.
    if "customer_comms_crew" in completed:
        return {
            "next": "FINISH"
        }


    # Use deterministic routing for obvious requests.
    fast_route = fallback_route(
        user_query,
        completed,
    )


    clear_specialists = {
        "policy_rag",
        "network_analytics",
        "network_diagnostics_adk",
        "billing_resolution_adk",
    }


    # First worker can be chosen without calling
    # the supervisor LLM when the intent is obvious.
    if (
        not completed
        and fast_route in clear_specialists
    ):
        return {
            "next": fast_route
        }


    # If specialist work is already complete and
    # customer communication is the next step,
    # skip another supervisor LLM call.
    if (
        completed
        and fast_route == "customer_comms_crew"
    ):
        return {
            "next": "customer_comms_crew"
        }


    # Once communications has completed,
    # the graph must finish.
    if (
        "customer_comms_crew"
        in completed
    ):
        return {
            "next": "FINISH"
        }


    prompt = f"""
You are the supervisor of a telecom multi-agent support system.

USER QUERY:
{user_query}

WORKERS ALREADY COMPLETED:
{completed}

ACCUMULATED AGENT CONTEXT:
{context}


AVAILABLE WORKERS:


policy_rag

Use for:
- telecom policies
- roaming
- SLA eligibility
- billing dispute policy
- 5G FAQ
- outage procedures
- device upgrade policies


network_analytics

Use for:
- SQL/database questions
- tower status
- packet loss
- latency
- throughput
- performance
- customer balances
- operational analytics
- regional database summaries


network_diagnostics_adk

Use for:
- network troubleshooting
- tower diagnostics
- connectivity problems
- 5G session drops
- network incidents
- signal problems


billing_resolution_adk

Use for:
- billing investigations
- duplicate charges
- billing disputes
- billing account lookup
- billing credits


customer_comms_crew

Use only after all required specialist workers have completed.

It creates the final customer-facing response.


FINISH

Use only after customer_comms_crew has completed.


ROUTING RULES:

1. Never call the same specialist worker twice.

2. A user query may require multiple specialist workers.

3. For policy or SLA questions use policy_rag.

4. For database / analytics questions use network_analytics.

5. For troubleshooting or tower issues use network_diagnostics_adk.

6. For billing investigation or resolution use billing_resolution_adk.

7. For combined questions, continue calling required specialist
   workers one at a time.

8. Do NOT call customer_comms_crew until all required specialists
   have completed.

9. customer_comms_crew must always be the final worker.

10. Once customer_comms_crew appears in WORKERS ALREADY COMPLETED,
    choose FINISH.

11. Never choose FINISH before customer_comms_crew.

12. Choose exactly one next worker.
"""


    try:
        decision = (
            supervisor_router.invoke(
                prompt
            )
        )

        selected = decision.next


        # Prevent FINISH before communications
        if (
            selected == "FINISH"
            and "customer_comms_crew"
            not in completed
        ):
            selected = fallback_route(
                user_query,
                completed,
            )


        # Prevent duplicate specialist execution
        if (
            selected in completed
            and selected
            != "customer_comms_crew"
        ):
            selected = fallback_route(
                user_query,
                completed,
            )


        return {
            "next": selected
        }


    except Exception as exc:
        print(
            "\nSupervisor LLM unavailable."
        )

        print(
            "Using fallback routing."
        )

        print(
            f"Reason: {exc}"
        )


        selected = fallback_route(
            user_query,
            completed,
        )


        return {
            "next": selected
        }
# ==================================================
# POLICY RAG NODE
# ==================================================

def policy_rag_node(
    state: AgentState,
):
    total_start = time.perf_counter()

    # Measure lazy import
    import_start = time.perf_counter()

    from llamaindex_rag.document_rag import (
        answer_policy_question,
    )

    import_elapsed = (
        time.perf_counter()
        - import_start
    )

    print(
        f"[TIMING] PolicyRAG IMPORT: "
        f"{import_elapsed:.2f}s"
    )

    question = state[
        "user_query"
    ]

    execution_start = time.perf_counter()

    try:
        result = (
            answer_policy_question(
                question
            )
        )

        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            f"[TIMING] PolicyRAG EXECUTION: "
            f"{execution_elapsed:.2f}s"
        )

    except Exception as exc:
        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            f"[TIMING] PolicyRAG FAILED AFTER: "
            f"{execution_elapsed:.2f}s"
        )

        result = (
            "Policy RAG could not complete "
            f"the request: {exc}"
        )

    context_entry = (
        "POLICY RAG RESULT:\n"
        f"{result}"
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        f"[TIMING] PolicyRAG TOTAL: "
        f"{total_elapsed:.2f}s"
    )

    return {
        "messages": [
            AIMessage(
                content=context_entry
            )
        ],

        "agent_context": [
            context_entry
        ],

        "completed_workers": [
            "policy_rag"
        ],

        "execution_trace": [
            {
                "worker":
                    "PolicyRAG",

                "output":
                    str(result),
            }
        ],
    }


# ==================================================
# NETWORK ANALYTICS NODE
# ==================================================

def network_analytics_node(
    state: AgentState,
):
    total_start = time.perf_counter()

    # Measure lazy import
    import_start = time.perf_counter()

    from llamaindex_rag.sql_semantic_search import (
        answer_sql_question,
    )

    import_elapsed = (
        time.perf_counter()
        - import_start
    )

    print(
        f"[TIMING] NetworkAnalytics IMPORT: "
        f"{import_elapsed:.2f}s"
    )

    question = state[
        "user_query"
    ]

    execution_start = time.perf_counter()

    try:
        result = (
            answer_sql_question(
                question
            )
        )

        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            f"[TIMING] NetworkAnalytics EXECUTION: "
            f"{execution_elapsed:.2f}s"
        )

    except Exception as exc:
        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] NetworkAnalytics "
            f"FAILED AFTER: "
            f"{execution_elapsed:.2f}s"
        )

        result = (
            "Network analytics could not "
            f"complete the request: {exc}"
        )

    context_entry = (
        "NETWORK / SQL ANALYTICS RESULT:\n"
        f"{result}"
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        f"[TIMING] NetworkAnalytics TOTAL: "
        f"{total_elapsed:.2f}s"
    )

    return {
        "messages": [
            AIMessage(
                content=context_entry
            )
        ],

        "agent_context": [
            context_entry
        ],

        "completed_workers": [
            "network_analytics"
        ],

        "execution_trace": [
            {
                "worker":
                    "NetworkAnalytics",

                "output":
                    str(result),
            }
        ],
    }


# ==================================================
# NETWORK DIAGNOSTICS ADK NODE
# ==================================================

def network_diagnostics_adk_node(
    state: AgentState,
):
    total_start = time.perf_counter()

    # Measure lazy import
    import_start = time.perf_counter()

    from orchestration.adk_remote_client import (
        run_network_diagnostics_remote,
    )

    import_elapsed = (
        time.perf_counter()
        - import_start
    )

    print(
        "[TIMING] NetworkDiagnosticsADK "
        f"IMPORT: {import_elapsed:.2f}s"
    )

    question = state[
        "user_query"
    ]

    execution_start = time.perf_counter()

    try:
        result = (
            run_network_diagnostics_remote(
                question
            )
        )

        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] NetworkDiagnosticsADK "
            f"EXECUTION: "
            f"{execution_elapsed:.2f}s"
        )

    except Exception as exc:
        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] NetworkDiagnosticsADK "
            f"FAILED AFTER: "
            f"{execution_elapsed:.2f}s"
        )

        result = (
            "Network Diagnostics remote "
            "service could not complete "
            f"the request: {exc}"
        )

    context_entry = (
        "NETWORK DIAGNOSTICS ADK RESULT:\n"
        f"{result}"
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        "[TIMING] NetworkDiagnosticsADK "
        f"TOTAL: {total_elapsed:.2f}s"
    )

    return {
        "messages": [
            AIMessage(
                content=context_entry
            )
        ],

        "agent_context": [
            context_entry
        ],

        "completed_workers": [
            "network_diagnostics_adk"
        ],

        "execution_trace": [
            {
                "worker":
                    "NetworkDiagnosticsADK",

                "output":
                    str(result),
            }
        ],
    }


# ==================================================
# BILLING RESOLUTION ADK NODE
# ==================================================

def billing_resolution_adk_node(
    state: AgentState,
):
    total_start = time.perf_counter()

    # Measure lazy import
    import_start = time.perf_counter()

    from orchestration.adk_remote_client import (
        run_billing_resolution_remote,
    )

    import_elapsed = (
        time.perf_counter()
        - import_start
    )

    print(
        "[TIMING] BillingResolutionADK "
        f"IMPORT: {import_elapsed:.2f}s"
    )

    question = state[
        "user_query"
    ]

    execution_start = time.perf_counter()

    try:
        result = (
            run_billing_resolution_remote(
                question
            )
        )

        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] BillingResolutionADK "
            f"EXECUTION: "
            f"{execution_elapsed:.2f}s"
        )

    except Exception as exc:
        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] BillingResolutionADK "
            f"FAILED AFTER: "
            f"{execution_elapsed:.2f}s"
        )

        result = (
            "Billing Resolution remote "
            "service could not complete "
            f"the request: {exc}"
        )

    context_entry = (
        "BILLING RESOLUTION ADK RESULT:\n"
        f"{result}"
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        "[TIMING] BillingResolutionADK "
        f"TOTAL: {total_elapsed:.2f}s"
    )

    return {
        "messages": [
            AIMessage(
                content=context_entry
            )
        ],

        "agent_context": [
            context_entry
        ],

        "completed_workers": [
            "billing_resolution_adk"
        ],

        "execution_trace": [
            {
                "worker":
                    "BillingResolutionADK",

                "output":
                    str(result),
            }
        ],
    }


# ==================================================
# CUSTOMER COMMS CREW NODE
# ==================================================

def customer_comms_crew_node(
    state: AgentState,
):
    total_start = time.perf_counter()

    # Measure CrewAI module import
    import_start = time.perf_counter()

    from orchestration.crew_nodes import (
        create_customer_response,
    )

    import_elapsed = (
        time.perf_counter()
        - import_start
    )

    print(
        f"[TIMING] CustomerCommsCrew IMPORT: "
        f"{import_elapsed:.2f}s"
    )

    question = state[
        "user_query"
    ]

    context = "\n\n".join(
        state.get(
            "agent_context",
            [],
        )
    )

    execution_start = time.perf_counter()

    try:
        response = (
            create_customer_response(
                question,
                context,
            )
        )

        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] CustomerCommsCrew "
            f"EXECUTION: "
            f"{execution_elapsed:.2f}s"
        )

    except Exception as exc:
        execution_elapsed = (
            time.perf_counter()
            - execution_start
        )

        print(
            "[TIMING] CustomerCommsCrew "
            f"FAILED AFTER: "
            f"{execution_elapsed:.2f}s"
        )

        print(
            "\nCustomerCommsCrew unavailable."
        )

        print(
            f"Reason: {exc}"
        )

        fallback_start = time.perf_counter()

        response = (
            fallback_customer_response(
                question,
                context,
            )
        )

        fallback_elapsed = (
            time.perf_counter()
            - fallback_start
        )

        print(
            "[TIMING] CustomerCommsFallback: "
            f"{fallback_elapsed:.2f}s"
        )

    context_entry = (
        "CUSTOMER COMMS RESULT:\n"
        + response
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        f"[TIMING] CustomerCommsCrew TOTAL: "
        f"{total_elapsed:.2f}s"
    )

    return {
        "messages": [
            AIMessage(
                content=response
            )
        ],

        "agent_context": [
            context_entry
        ],

        "completed_workers": [
            "customer_comms_crew"
        ],

        "execution_trace": [
            {
                "worker":
                    "CustomerCommsCrew",

                "output":
                    str(response),
            }
        ],

        "final_response":
            response,
    }
# ==================================================
# FALLBACK CUSTOMER RESPONSE
# ==================================================

def fallback_customer_response(
    question: str,
    context: str,
) -> str:
    """
    Used only if CrewAI / LLM cannot run.

    This keeps the application usable during
    temporary API quota problems.
    """

    if not context.strip():
        return (
            "The request could not be fully "
            "processed because no specialist "
            "information was available."
        )


    return (
        "The automated customer-response service "
        "is currently unavailable.\n\n"
        "The specialist investigation produced "
        "the following information:\n\n"
        f"{context}"
    )


# ==================================================
# CUSTOMER COMMS CREW NODE
# ==================================================
def customer_comms_crew_node(
    state: AgentState,
):
    # Lazy import
    from orchestration.crew_nodes import (
        create_customer_response,
    )

    question = state[
        "user_query"
    ]

    context = "\n\n".join(
        state.get(
            "agent_context",
            [],
        )
    )

    start = time.perf_counter()

    try:
        response = (
            create_customer_response(
                question,
                context,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"[TIMING] CustomerCommsCrew: "
            f"{elapsed:.2f}s"
        )

    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"[TIMING] CustomerCommsCrew "
            f"failed after: {elapsed:.2f}s"
        )

        print(
            "\nCustomerCommsCrew unavailable."
        )

        print(
            f"Reason: {exc}"
        )

        fallback_start = time.perf_counter()

        response = (
            fallback_customer_response(
                question,
                context,
            )
        )

        fallback_elapsed = (
            time.perf_counter()
            - fallback_start
        )

        print(
            f"[TIMING] CustomerCommsFallback: "
            f"{fallback_elapsed:.2f}s"
        )

    context_entry = (
        "CUSTOMER COMMS RESULT:\n"
        + response
    )

    return {
        "messages": [
            AIMessage(
                content=response
            )
        ],

        "agent_context": [
            context_entry
        ],

        "completed_workers": [
            "customer_comms_crew"
        ],

        "execution_trace": [
            {
                "worker":
                    "CustomerCommsCrew",

                "output":
                    str(response),
            }
        ],

        "final_response":
            response,
    }
# ==================================================
# ROUTE FUNCTION
# ==================================================

def route_from_supervisor(
    state: AgentState,
):
    return state[
        "next"
    ]


# ==================================================
# BUILD LANGGRAPH
# ==================================================

builder = StateGraph(
    AgentState
)


builder.add_node(
    "supervisor",
    supervisor_node,
)


builder.add_node(
    "policy_rag",
    policy_rag_node,
)


builder.add_node(
    "network_analytics",
    network_analytics_node,
)


builder.add_node(
    "network_diagnostics_adk",
    network_diagnostics_adk_node,
)


builder.add_node(
    "billing_resolution_adk",
    billing_resolution_adk_node,
)


builder.add_node(
    "customer_comms_crew",
    customer_comms_crew_node,
)


# --------------------------------------------------
# START
# --------------------------------------------------

builder.add_edge(
    START,
    "supervisor",
)


# --------------------------------------------------
# SUPERVISOR ROUTING
# --------------------------------------------------

builder.add_conditional_edges(
    "supervisor",

    route_from_supervisor,

    {
        "policy_rag":
            "policy_rag",

        "network_analytics":
            "network_analytics",

        "network_diagnostics_adk":
            "network_diagnostics_adk",

        "billing_resolution_adk":
            "billing_resolution_adk",

        "customer_comms_crew":
            "customer_comms_crew",

        "FINISH":
            END,
    },
)


# --------------------------------------------------
# EVERY WORKER RETURNS TO SUPERVISOR
# --------------------------------------------------

builder.add_edge(
    "policy_rag",
    "supervisor",
)


builder.add_edge(
    "network_analytics",
    "supervisor",
)


builder.add_edge(
    "network_diagnostics_adk",
    "supervisor",
)


builder.add_edge(
    "billing_resolution_adk",
    "supervisor",
)


builder.add_edge(
    "customer_comms_crew",
    "supervisor",
)


# --------------------------------------------------
# COMPILE
# --------------------------------------------------

graph = builder.compile()


# ==================================================
# PUBLIC ENTRY POINT
# ==================================================
def run_support_graph(
    question: str,
):
    initial_state = {

        "messages": [
            HumanMessage(
                content=question
            )
        ],

        "user_query":
            question,

        "next":
            "",

        "agent_context":
            [],

        "completed_workers":
            [],

        "execution_trace":
            [],

        "final_response":
            "",
    }


    graph_start = time.perf_counter()


    result = graph.invoke(
        initial_state,

        config={
            "recursion_limit": 20
        },
    )


    graph_elapsed = (
        time.perf_counter()
        - graph_start
    )


    print(
        "\n========================================"
    )

    print(
        f"[TIMING] COMPLETE LANGGRAPH REQUEST: "
        f"{graph_elapsed:.2f}s"
    )

    print(
        "========================================\n"
    )


    return result


# ==================================================
# CLI TEST
# ==================================================

if __name__ == "__main__":

    question = input(
        "Ask a telecom support question: "
    )


    result = run_support_graph(
        question
    )


    print(
        "\n================================"
    )

    print(
        "AGENT EXECUTION TRACE"
    )

    print(
        "================================"
    )


    trace = result.get(
        "execution_trace",
        [],
    )


    if trace:

        for index, step in enumerate(
            trace,
            start=1,
        ):
            print()

            print(
                f"Step {index} - "
                f"{step.get('worker', 'Unknown')}"
            )

            print(
                step.get(
                    "output",
                    "",
                )
            )

    else:
        print(
            "No worker trace generated."
        )


    print(
        "\n================================"
    )

    print(
        "COMPLETED WORKERS"
    )

    print(
        "================================"
    )


    for worker in result.get(
        "completed_workers",
        [],
    ):
        print(
            f"✓ {worker}"
        )


    print(
        "\n================================"
    )

    print(
        "FINAL RESPONSE"
    )

    print(
        "================================"
    )


    print(
        result.get(
            "final_response",
            "No final response generated.",
        )
    )