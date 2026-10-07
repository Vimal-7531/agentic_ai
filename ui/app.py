import sys
from pathlib import Path

import requests
import streamlit as st


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "telecom_ops.db"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "policy_index"
)


NETWORK_AGENT_CARD = (
    "http://localhost:8001/"
    ".well-known/agent-card.json"
)

BILLING_AGENT_CARD = (
    "http://localhost:8002/"
    ".well-known/agent-card.json"
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Prodapt AI Operations Center",
    page_icon="📡",
    layout="wide",
)

@st.cache_resource
def get_support_runner():
    from orchestration.graph import run_support_graph
    return run_support_graph

@st.cache_resource
def preload_ai_resources():
    # Preload Semantic SQL
    from llamaindex_rag.sql_semantic_search import (
        answer_sql_question,
    )

    # Preload Policy RAG
    from llamaindex_rag.document_rag import (
        answer_policy_question,
    )

    return {
        "sql": answer_sql_question,
        "policy": answer_policy_question,
    }

with st.spinner(
    "Initializing AI resources..."
):
    preload_ai_resources()

@st.cache_data(ttl=10)

def service_running(url: str) -> bool:
    try:
        response = requests.get(
            url,
            timeout=0.5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False
# ==================================================
# SERVICE HEALTH CHECK
# ==================================================



# ==================================================
# CHECK SYSTEM STATUS
# ==================================================

database_ready = DB_PATH.exists()

index_ready = INDEX_PATH.exists()

network_adk_running = service_running(
    NETWORK_AGENT_CARD
)

billing_adk_running = service_running(
    BILLING_AGENT_CARD
)


# ==================================================
# HEADER
# ==================================================

st.title(
    "📡 Prodapt AI Operations Center"
)

st.caption(
    "Agentic Telecom Support using "
    "LangGraph, LlamaIndex, Google ADK, "
    "A2A and CrewAI"
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header(
        "System Status"
    )


    # --------------------------------------------------
    # DATABASE
    # --------------------------------------------------

    if database_ready:
        st.success(
            "SQLite Database: Ready"
        )

    else:
        st.error(
            "SQLite Database: Missing"
        )


    # --------------------------------------------------
    # POLICY INDEX
    # --------------------------------------------------

    if index_ready:
        st.success(
            "Policy Vector Index: Ready"
        )

    else:
        st.error(
            "Policy Vector Index: Missing"
        )


    # --------------------------------------------------
    # NETWORK ADK
    # --------------------------------------------------

    if network_adk_running:
        st.success(
            "Network ADK: Running"
        )

    else:
        st.error(
            "Network ADK: Offline"
        )


    # --------------------------------------------------
    # BILLING ADK
    # --------------------------------------------------

    if billing_adk_running:
        st.success(
            "Billing ADK: Running"
        )

    else:
        st.error(
            "Billing ADK: Offline"
        )


    # --------------------------------------------------
    # SERVICE START HELP
    # --------------------------------------------------

    if (
        not network_adk_running
        or not billing_adk_running
    ):

        with st.expander(
            "How to start ADK services"
        ):

            if not network_adk_running:

                st.markdown(
                    "**Network Diagnostics**"
                )

                st.code(
                    "uvicorn "
                    "adk_services.network_diagnostics.agent:"
                    "a2a_app "
                    "--host localhost "
                    "--port 8001",
                    language="powershell",
                )


            if not billing_adk_running:

                st.markdown(
                    "**Billing Resolution**"
                )

                st.code(
                    "uvicorn "
                    "adk_services.billing_resolution.agent:"
                    "a2a_app "
                    "--host localhost "
                    "--port 8002",
                    language="powershell",
                )


    st.divider()


    # --------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------

    st.header(
        "Architecture"
    )


    st.markdown(
        """
**LangGraph**
- Supervisor
- Agent routing
- Multi-agent workflow

**LlamaIndex**
- Policy RAG
- Semantic SQL

**Google ADK + A2A**
- Network Diagnostics
- Billing Resolution

**CrewAI**
- Customer response
- Quality review

**Streamlit**
- User interface
- Execution trace
"""
    )



# ==================================================
# MAIN UI
# ==================================================

st.subheader(
    "Customer Support Assistant"
)


query = st.text_area(
    "Ask a telecom support question",
    height=120,
    placeholder=(
        "Example: Why am I experiencing repeated "
        "5G session drops near tower TX-512?"
    ),
)


submit = st.button(
    "Submit Inquiry",
    type="primary",
    use_container_width=True,
)


# ==================================================
# EXECUTION
# ==================================================

if submit:

    # --------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a telecom support question."
        )

        st.stop()


    # --------------------------------------------------
    # START WORKFLOW
    # --------------------------------------------------

    st.info(
        "Starting multi-agent workflow..."
    )


    try:

        # Lazy import prevents large AI libraries
        # from blocking initial Streamlit rendering.

        run_support_graph = get_support_runner()


        with st.spinner(
            "Agents are investigating the request..."
        ):

            result = run_support_graph(
                query.strip()
            )


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as exc:

        error_message = str(
            exc
        )

        error_lower = (
            error_message.lower()
        )


        # --------------------------------------------------
        # LLM QUOTA / BILLING
        # --------------------------------------------------

        if any(
            phrase in error_lower
            for phrase in [
                "credit balance is too low",
                "insufficient credits",
                "insufficient",
                "quota",
                "rate limit",
                "billing limit",
            ]
        ):

            st.error(
                "LLM API limit or billing issue detected."
            )

            st.warning(
                "The external LLM provider currently "
                "cannot complete this request because "
                "of insufficient credits, quota, or "
                "account limits."
            )


        # --------------------------------------------------
        # A2A CONNECTION FAILURE
        # --------------------------------------------------

        elif any(
            phrase in error_lower
            for phrase in [
                "connection refused",
                "failed to connect",
                "localhost:8001",
                "localhost:8002",
                "connecterror",
            ]
        ):

            st.error(
                "A2A service connection failed."
            )

            st.warning(
                "Make sure the Network Diagnostics "
                "service on port 8001 and Billing "
                "Resolution service on port 8002 "
                "are running."
            )


        # --------------------------------------------------
        # AUTHENTICATION
        # --------------------------------------------------

        elif any(
            phrase in error_lower
            for phrase in [
                "api key",
                "authentication",
                "unauthorized",
                "invalid_api_key",
            ]
        ):

            st.error(
                "API authentication failed."
            )

            st.warning(
                "Check the API key configured "
                "in your .env file."
            )


        # --------------------------------------------------
        # OTHER
        # --------------------------------------------------

        else:

            st.error(
                "The workflow could not complete."
            )


        with st.expander(
            "Technical Error Details"
        ):

            st.code(
                error_message
            )


        st.stop()


    # ==================================================
    # FINAL CUSTOMER RESPONSE
    # ==================================================

    st.divider()

    st.subheader(
        "💬 Customer Response"
    )


    final_response = result.get(
        "final_response",
        "",
    )


    if final_response:

        st.success(
            final_response
        )

    else:

        st.warning(
            "No final response was generated."
        )


    # ==================================================
    # AGENT EXECUTION TRACE
    # ==================================================

    st.divider()

    st.subheader(
        "🔎 Agent Execution Trace"
    )


    execution_trace = result.get(
        "execution_trace",
        [],
    )


    if execution_trace:

        for number, step in enumerate(
            execution_trace,
            start=1,
        ):

            worker = step.get(
                "worker",
                "Unknown Worker",
            )

            output = step.get(
                "output",
                "",
            )


            st.markdown(
                f"### Step {number} — {worker}"
            )


            if output:

                st.write(
                    output
                )

            else:

                st.caption(
                    "No output returned."
                )


            if number < len(
                execution_trace
            ):
                st.divider()


    else:

        st.info(
            "No execution trace was returned."
        )


    # ==================================================
    # COMPLETED WORKERS
    # ==================================================

    completed_workers = result.get(
        "completed_workers",
        [],
    )


    with st.expander(
        "Completed Workers"
    ):

        if completed_workers:

            for worker in completed_workers:

                st.write(
                    f"✅ {worker}"
                )

        else:

            st.write(
                "No completed workers recorded."
            )


    # ==================================================
    # SPECIALIST OUTPUT
    # ==================================================

    contexts = result.get(
        "agent_context",
        [],
    )


    with st.expander(
        "View Specialist Agent Outputs"
    ):

        if contexts:

            for number, context in enumerate(
                contexts,
                start=1,
            ):

                st.markdown(
                    f"#### Agent Output {number}"
                )

                st.code(
                    str(context)
                )


                if number < len(
                    contexts
                ):
                    st.divider()


        else:

            st.write(
                "No agent context available."
            )


    # ==================================================
    # LANGGRAPH MESSAGES
    # ==================================================

    messages = result.get(
        "messages",
        [],
    )


    with st.expander(
        "View LangGraph Messages"
    ):

        if messages:

            for number, message in enumerate(
                messages,
                start=1,
            ):

                content = getattr(
                    message,
                    "content",
                    str(message),
                )


                st.markdown(
                    f"**Message {number}**"
                )

                st.write(
                    content
                )


        else:

            st.write(
                "No graph messages available."
            )