import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

from llama_index.core import (
    Settings,
    SQLDatabase,
    VectorStoreIndex,
)

from llama_index.core.indices.struct_store import (
    SQLTableRetrieverQueryEngine,
)

from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

from llama_index.llms.anthropic import Anthropic


# ==================================================
# PATHS / ENVIRONMENT
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"

DB_PATH = (
    BASE_DIR
    / "data"
    / "telecom_ops.db"
)


load_dotenv(
    dotenv_path=ENV_PATH
)


print(
    "DATABASE:",
    DB_PATH,
)

print(
    "DATABASE EXISTS:",
    DB_PATH.exists(),
)

print(
    "ANTHROPIC KEY LOADED:",
    bool(
        os.getenv(
            "ANTHROPIC_API_KEY"
        )
    ),
)


# ==================================================
# EMBEDDING MODEL
# ==================================================

embedding_model = HuggingFaceEmbedding(
    model_name=(
        "BAAI/bge-small-en-v1.5"
    )
)


# ==================================================
# LLM
# ==================================================

MODEL_NAME = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5",
)


llm = Anthropic(
    model=MODEL_NAME,
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    ),
    max_tokens=700,
)


Settings.embed_model = (
    embedding_model
)

Settings.llm = llm


# ==================================================
# SQL DATABASE
# ==================================================

engine = create_engine(
    f"sqlite:///{DB_PATH.as_posix()}"
)


sql_database = SQLDatabase(
    engine
)


# ==================================================
# TABLE NODE MAPPING
# ==================================================

table_node_mapping = (
    SQLTableNodeMapping(
        sql_database
    )
)


# ==================================================
# TABLE SCHEMA DESCRIPTIONS
# ==================================================

table_schemas = [

    # --------------------------------------------------
    # NETWORK TOWERS
    # --------------------------------------------------

    SQLTableSchema(
        table_name="network_towers",

        context_str="""
Contains telecom tower/site information.

Important fields include:
- tower_id
- tower_name
- region
- technology
- status

The status field describes current tower operational state.

Important interpretation rules:

OPERATIONAL:
The tower is functioning normally.

DEGRADED:
The tower is functioning but experiencing reduced or impaired performance.
A DEGRADED tower is NOT the same as an inactive tower.

DOWN or OUT_OF_SERVICE, if present:
The tower is unavailable.

Do NOT describe OPERATIONAL or DEGRADED towers as inactive.

For regional summaries, report the actual status value for each tower.

Example:
TX-512 with status OPERATIONAL means the tower is operating normally.
TX-208 with status DEGRADED means the tower is operating with degraded service.

Never infer that all towers are inactive merely because none has a literal
status value called ACTIVE.
""",
    ),


    # --------------------------------------------------
    # TOWER PERFORMANCE
    # --------------------------------------------------

    SQLTableSchema(
        table_name="tower_performance",

        context_str="""
Contains time-series performance measurements for telecom towers.

Important fields include:
- tower_id
- measurement timestamp
- packet_loss_pct
- latency_ms
- downlink throughput
- uplink throughput
- signal strength
- active connections

Use the most recent measurement when the user asks for the latest
performance or current performance.

Do not infer tower operational status purely from performance metrics.
Use network_towers.status for tower operational state.
""",
    ),


    # --------------------------------------------------
    # NETWORK OUTAGES
    # --------------------------------------------------

    SQLTableSchema(
        table_name="network_outages",

        context_str="""
Contains recorded network outage events.

Includes information such as:
- affected tower or region
- outage start time
- outage end time
- severity
- outage status
- duration
- operational details

An outage record should not automatically be interpreted as proof that
every tower in the region is currently offline.

Use network_towers.status to describe current tower state.
Use this table when the question specifically asks about outage history,
duration, outage events or affected areas.
""",
    ),


    # --------------------------------------------------
    # OPEN INCIDENTS
    # --------------------------------------------------

    SQLTableSchema(
        table_name="open_incidents",

        context_str="""
Contains currently open network incidents.

Includes:
- incident ID
- tower ID
- severity
- incident status
- issue description
- suspected cause
- investigation information

An open incident does not necessarily mean the associated tower is down.

Always distinguish:
- tower operational status
from
- incident status.

For tower operational state, use network_towers.status.
""",
    ),


    # --------------------------------------------------
    # BILLING ACCOUNTS
    # --------------------------------------------------

    SQLTableSchema(
        table_name="billing_accounts",

        context_str="""
Contains customer billing account information.

Includes:
- customer_id
- customer name
- current balance
- account type
- billing cycle
- account status
- autopay information

Use current_balance for questions about the customer's current amount due.
""",
    ),


    # --------------------------------------------------
    # BILLING CHARGES
    # --------------------------------------------------

    SQLTableSchema(
        table_name="billing_charges",

        context_str="""
Contains individual customer billing charges.

Includes:
- charge_id
- customer_id
- description
- amount
- billing_period
- charge_date
- charge_type
- is_duplicate_flag
- invoice_status

A charge with is_duplicate_flag = 1 is explicitly marked as a duplicate.

Charges may also be potential duplicates when the same customer has
multiple charges with the same description and billing period.
""",
    ),


    # --------------------------------------------------
    # BILLING CREDITS
    # --------------------------------------------------

    SQLTableSchema(
        table_name="billing_credits",

        context_str="""
Contains customer billing credits.

Includes:
- credit information
- customer ID
- credit amount
- reason
- status
- creation timestamp

Possible statuses may include:
- APPLIED
- PENDING_APPROVAL

Do not treat PENDING_APPROVAL as an already applied credit.
""",
    ),


    # --------------------------------------------------
    # BILLING DISPUTES
    # --------------------------------------------------

    SQLTableSchema(
        table_name="billing_disputes",

        context_str="""
Contains billing disputes raised by customers.

Includes:
- dispute information
- customer ID
- charge ID
- reason
- dispute status
- opened date
- resolution information

Use this table when determining whether a particular billing charge
has already been disputed or resolved.
""",
    ),


    # --------------------------------------------------
    # CUSTOMER SUBSCRIPTIONS
    # --------------------------------------------------

    SQLTableSchema(
        table_name="customer_subscriptions",

        context_str="""
Contains customer subscription and service-plan information.

Includes:
- customer ID
- subscribed plan
- service information
- plan status
- related account information

Use this table for questions about the customer's current subscription
or telecom service plan.
""",
    ),
]


# ==================================================
# OBJECT INDEX
# ==================================================

object_index = (
    ObjectIndex.from_objects(
        table_schemas,
        table_node_mapping,
        VectorStoreIndex,
    )
)


# ==================================================
# TABLE RETRIEVER
# ==================================================

table_retriever = (
    object_index.as_retriever(
        similarity_top_k=3
    )
)


# ==================================================
# SEMANTIC SQL QUERY ENGINE
# ==================================================

query_engine = (
    SQLTableRetrieverQueryEngine(
        sql_database=sql_database,
        table_retriever=table_retriever,
        llm=llm,
    )
)


# ==================================================
# QUERY GUIDANCE
# ==================================================

QUERY_GUIDANCE = """
Important telecom database interpretation rules:

1. For tower state, always use the actual status from network_towers.

2. OPERATIONAL means the tower is functioning normally.

3. DEGRADED means the tower is functioning with impaired performance.
   It must NOT be described as inactive.

4. Do not count only a literal status named ACTIVE.
   The database uses operational-state labels such as OPERATIONAL
   and DEGRADED.

5. An open incident does not necessarily mean the tower is offline.

6. An outage record does not prove that all towers in the same region
   are currently unavailable.

7. For regional network summaries:
   - list the towers
   - report their exact status values
   - count statuses accurately
   - describe DEGRADED separately from OPERATIONAL
   - do not invent an 'active/inactive' classification unless the
     database itself contains that classification.

8. Never invent operational or billing values.
Use only values obtained from the database.
"""


# ==================================================
# PUBLIC FUNCTION
# ==================================================

def answer_sql_question(
    question: str,
) -> str:
    enhanced_question = f"""
{QUERY_GUIDANCE}

USER QUESTION:

{question}
"""

    response = query_engine.query(
        enhanced_question
    )

    return str(
        response
    )


# ==================================================
# CLI TEST
# ==================================================

if __name__ == "__main__":

    question = input(
        "Ask an operational data question: "
    )

    answer = answer_sql_question(
        question
    )

    print(
        "\nANSWER:"
    )

    print(
        answer
    )