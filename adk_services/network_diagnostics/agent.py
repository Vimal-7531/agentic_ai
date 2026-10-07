import os
import sys
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a


# ==================================================
# PROJECT / ENVIRONMENT SETUP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH
)


DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "telecom_ops.db"
)


from agents.database_tools import (
    get_tower,
    get_tower_performance,
    get_open_incidents,
)


# ==================================================
# TOOL 1 - CHECK TOWER STATUS
# ==================================================

def check_tower_status(
    tower_id: str,
) -> dict:
    """
    Return basic information for a tower.

    Operational information is retrieved
    from the SQLite-backed database tools.
    """

    tower = get_tower(
        tower_id
    )

    if not tower:
        return {
            "status": "error",
            "tower_id": tower_id,
            "message": (
                f"Tower {tower_id} was not found."
            ),
        }


    performance = get_tower_performance(
        tower_id
    )

    incidents = get_open_incidents(
        tower_id
    )


    return {
        "status": "success",
        "tower_id": tower_id,
        "tower": tower,
        "latest_performance": performance,
        "open_incidents": incidents,
    }


# ==================================================
# TOOL 2 - CONNECTIVITY DIAGNOSTICS
# ==================================================

def run_connectivity_diagnostics(
    tower_id: str,
    symptom: str,
) -> dict:
    """
    Investigate connectivity problems for
    a specific tower using operational data.
    """

    tower = get_tower(
        tower_id
    )

    if not tower:
        return {
            "status": "error",
            "tower_id": tower_id,
            "message": (
                f"Tower {tower_id} was not found."
            ),
        }


    performance = get_tower_performance(
        tower_id
    )

    incidents = get_open_incidents(
        tower_id
    )


    recommendations = []


    # --------------------------------------------------
    # BASIC DATA-DRIVEN RECOMMENDATIONS
    # --------------------------------------------------

    if performance:

        # Support either dictionary or list responses.
        if isinstance(
            performance,
            list,
        ):
            latest = (
                performance[0]
                if performance
                else {}
            )

        else:
            latest = performance


        packet_loss = latest.get(
            "packet_loss_pct"
        )

        latency = latest.get(
            "latency_ms"
        )

        signal = latest.get(
            "signal_strength_dbm",
            latest.get(
                "signal_dbm"
            ),
        )


        if (
            packet_loss is not None
            and packet_loss > 2
        ):
            recommendations.append(
                "Packet loss is elevated. "
                "Check radio scheduling and backhaul congestion."
            )


        if (
            latency is not None
            and latency > 50
        ):
            recommendations.append(
                "Latency is elevated. "
                "Inspect transport/backhaul performance."
            )


        if (
            signal is not None
            and signal < -85
        ):
            recommendations.append(
                "Signal strength is weak. "
                "Check radio coverage and interference."
            )


    if incidents:
        recommendations.append(
            "Review the active incident associated "
            "with this tower before performing "
            "additional remediation."
        )


    if not recommendations:
        recommendations.append(
            "No obvious issue was identified from "
            "the available operational metrics. "
            "Continue standard connectivity checks."
        )


    return {
        "status": "success",
        "tower_id": tower_id,
        "symptom": symptom,
        "tower": tower,
        "performance": performance,
        "incidents": incidents,
        "recommendations": recommendations,
    }


# ==================================================
# TOOL 3 - REGIONAL NETWORK SUMMARY
# ==================================================

def get_regional_network_summary(
    region: str,
) -> dict:
    """
    Return all towers in a region together with
    their operational status.

    Region information is obtained directly from
    the network_towers table rather than incidents.
    """

    if not DB_PATH.exists():
        return {
            "status": "error",
            "message": (
                "Telecom operations database "
                "was not found."
            ),
        }


    connection = None

    try:

        connection = sqlite3.connect(
            DB_PATH
        )

        connection.row_factory = (
            sqlite3.Row
        )

        cursor = connection.cursor()


        # --------------------------------------------------
        # DISCOVER TABLE COLUMNS
        # --------------------------------------------------
        #
        # This makes the tool tolerant of status
        # column naming differences such as:
        # status
        # operational_status
        #
        # without hardcoding a schema assumption.
        # --------------------------------------------------

        cursor.execute(
            """
            PRAGMA table_info(network_towers)
            """
        )

        columns = {
            row["name"]
            for row in cursor.fetchall()
        }


        if "region" not in columns:
            return {
                "status": "error",
                "message": (
                    "network_towers does not contain "
                    "a region column."
                ),
            }


        if "operational_status" in columns:
            status_column = (
                "operational_status"
            )

        elif "status" in columns:
            status_column = "status"

        else:
            status_column = None


        # Optional descriptive columns
        select_columns = [
            "tower_id"
        ]


        if "tower_name" in columns:
            select_columns.append(
                "tower_name"
            )

        elif "name" in columns:
            select_columns.append(
                "name"
            )


        select_columns.append(
            "region"
        )


        if "technology" in columns:
            select_columns.append(
                "technology"
            )


        if status_column:
            select_columns.append(
                status_column
            )


        sql = f"""
            SELECT
                {", ".join(select_columns)}
            FROM network_towers
            WHERE LOWER(region) = LOWER(?)
            ORDER BY tower_id
        """


        cursor.execute(
            sql,
            (region,),
        )


        towers = [
            dict(row)
            for row in cursor.fetchall()
        ]


        if not towers:
            return {
                "status": "not_found",
                "region": region,
                "tower_count": 0,
                "towers": [],
                "message": (
                    f"No towers were found "
                    f"for region '{region}'."
                ),
            }


        # --------------------------------------------------
        # NORMALIZE STATUS FIELD
        # --------------------------------------------------

        if (
            status_column
            and status_column
            != "status"
        ):

            for tower in towers:
                tower["status"] = (
                    tower.pop(
                        status_column,
                        None,
                    )
                )


        # --------------------------------------------------
        # REGION SUMMARY
        # --------------------------------------------------

        status_summary = {}


        for tower in towers:

            tower_status = tower.get(
                "status",
                "UNKNOWN",
            )

            status_summary[
                tower_status
            ] = (
                status_summary.get(
                    tower_status,
                    0,
                )
                + 1
            )


        return {
            "status": "success",
            "region": region,
            "tower_count": len(
                towers
            ),
            "status_summary": (
                status_summary
            ),
            "towers": towers,
        }


    except sqlite3.Error as exc:

        return {
            "status": "error",
            "region": region,
            "message": (
                "Database error while retrieving "
                f"regional network summary: {exc}"
            ),
        }


    finally:

        if connection:
            connection.close()


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_NAME = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5",
)


# ==================================================
# GOOGLE ADK AGENT
# ==================================================

root_agent = Agent(

    name="network_diagnostics",

    model=LiteLlm(
        model=f"anthropic/{MODEL_NAME}"
        if not MODEL_NAME.startswith(
            "anthropic/"
        )
        else MODEL_NAME
    ),

    description=(
        "Remote telecom Network Operations "
        "Center diagnostics agent."
    ),

    instruction="""
You are the Network Diagnostics specialist for a telecom operations system.

You have access to SQLite-backed operational tools.

Use the tools whenever operational facts are required.

Responsibilities:

- Check tower status.
- Diagnose connectivity issues.
- Investigate packet loss.
- Investigate latency.
- Investigate throughput.
- Investigate signal issues.
- Check open incidents.
- Summarize network health by region.

Important rules:

1. Never invent operational values.

2. Tower status, performance metrics, incidents and regional
   summaries must come from the supplied tools.

3. When troubleshooting a tower, consider:
   - tower operational status
   - latest performance metrics
   - packet loss
   - latency
   - throughput
   - signal strength
   - active incidents

4. For regional questions, use get_regional_network_summary.

5. If the data is unavailable, clearly state that it is unavailable.

6. Keep responses concise and suitable for telecom/NOC support staff.
""",

    tools=[
        check_tower_status,
        run_connectivity_diagnostics,
        get_regional_network_summary,
    ],
)


# ==================================================
# A2A SERVICE
# ==================================================

a2a_app = to_a2a(
    root_agent,
    host="localhost",
    port=8001,
)