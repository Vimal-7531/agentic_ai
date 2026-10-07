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

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "telecom_ops.db"
)

ENV_PATH = PROJECT_ROOT / ".env"

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

load_dotenv(
    dotenv_path=ENV_PATH
)


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():
    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn


# ==================================================
# TOOL 1 - BILLING ACCOUNT LOOKUP
# ==================================================

def lookup_billing_account(
    customer_id: str,
) -> dict:
    conn = get_connection()

    try:
        account = conn.execute(
            """
            SELECT *
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,),
        ).fetchone()

        if not account:
            return {
                "status": "error",
                "customer_id": customer_id,
                "message": (
                    f"Customer {customer_id} "
                    "was not found."
                ),
            }

        charges = conn.execute(
            """
            SELECT *
            FROM billing_charges
            WHERE customer_id = ?
            ORDER BY charge_date DESC
            LIMIT 5
            """,
            (customer_id,),
        ).fetchall()

        credits = conn.execute(
            """
            SELECT *
            FROM billing_credits
            WHERE customer_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (customer_id,),
        ).fetchall()

        disputes = conn.execute(
            """
            SELECT *
            FROM billing_disputes
            WHERE customer_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (customer_id,),
        ).fetchall()

        return {
            "status": "success",
            "account": dict(account),
            "recent_charges": [
                dict(row)
                for row in charges
            ],
            "recent_credits": [
                dict(row)
                for row in credits
            ],
            "recent_disputes": [
                dict(row)
                for row in disputes
            ],
        }

    finally:
        conn.close()


# ==================================================
# TOOL 2 - DUPLICATE CHARGE DETECTION
# ==================================================

def check_duplicate_charges(
    customer_id: str,
) -> dict:
    """
    A charge is treated as duplicate if:

    1. is_duplicate_flag = 1

    OR

    2. Another charge exists for the same customer,
       same description and same billing period.
    """

    conn = get_connection()

    try:
        duplicates = conn.execute(
            """
            SELECT DISTINCT
                bc.*
            FROM billing_charges bc
            WHERE bc.customer_id = ?
              AND (
                    bc.is_duplicate_flag = 1

                    OR EXISTS (
                        SELECT 1
                        FROM billing_charges other
                        WHERE other.customer_id =
                              bc.customer_id

                          AND other.description =
                              bc.description

                          AND other.billing_period =
                              bc.billing_period

                          AND other.charge_id !=
                              bc.charge_id
                    )
              )
            ORDER BY bc.charge_date DESC
            """,
            (customer_id,),
        ).fetchall()

        duplicate_rows = [
            dict(row)
            for row in duplicates
        ]

        return {
            "status": "success",
            "customer_id": customer_id,
            "duplicate_found": (
                len(duplicate_rows) > 0
            ),
            "duplicate_count": (
                len(duplicate_rows)
            ),
            "duplicate_charges": (
                duplicate_rows
            ),
        }

    finally:
        conn.close()


# ==================================================
# TOOL 3 - APPLY BILLING CREDIT
# ==================================================

def apply_billing_credit(
    customer_id: str,
    amount: float,
    reason: str,
) -> dict:
    """
    Billing rules:

    - amount <= 0:
        invalid

    - amount <= 50:
        APPLIED immediately
        account balance reduced

    - amount > 50:
        PENDING_APPROVAL
        account balance unchanged

    - equivalent APPLIED credit already exists:
        do not apply again
    """

    if amount <= 0:
        return {
            "status": "error",
            "message": (
                "Credit amount must be "
                "greater than zero."
            ),
        }

    conn = get_connection()

    try:
        account = conn.execute(
            """
            SELECT *
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,),
        ).fetchone()

        if not account:
            return {
                "status": "error",
                "customer_id": customer_id,
                "message": (
                    f"Customer {customer_id} "
                    "was not found."
                ),
            }


        # ------------------------------------------
        # IDEMPOTENCY CHECK
        # ------------------------------------------

        existing_credit = conn.execute(
            """
            SELECT *
            FROM billing_credits
            WHERE customer_id = ?
              AND amount = ?
              AND LOWER(reason) = LOWER(?)
              AND status = 'APPLIED'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                customer_id,
                amount,
                reason,
            ),
        ).fetchone()


        if existing_credit:
            return {
                "status": "success",
                "customer_id": customer_id,
                "credit_status": (
                    "ALREADY_APPLIED"
                ),
                "credit_id": (
                    existing_credit["credit_id"]
                    if "credit_id"
                    in existing_credit.keys()
                    else None
                ),
                "amount": amount,
                "reason": reason,
                "message": (
                    "An equivalent billing credit "
                    "has already been applied. "
                    "No additional credit was created."
                ),
            }


        # ------------------------------------------
        # CREDIT APPROVAL RULE
        # ------------------------------------------

        if amount <= 50:
            credit_status = "APPLIED"
        else:
            credit_status = (
                "PENDING_APPROVAL"
            )


        cursor = conn.execute(
            """
            INSERT INTO billing_credits (
                customer_id,
                amount,
                reason,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                customer_id,
                amount,
                reason,
                credit_status,
            ),
        )


        # ------------------------------------------
        # UPDATE BALANCE ONLY IF APPLIED
        # ------------------------------------------

        if credit_status == "APPLIED":
            conn.execute(
                """
                UPDATE billing_accounts
                SET current_balance =
                    current_balance - ?,
                    last_updated =
                    CURRENT_TIMESTAMP
                WHERE customer_id = ?
                """,
                (
                    amount,
                    customer_id,
                ),
            )


        conn.commit()

        credit_id = cursor.lastrowid


        updated_account = conn.execute(
            """
            SELECT *
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,),
        ).fetchone()


        return {
            "status": "success",
            "customer_id": customer_id,
            "credit_id": credit_id,
            "amount": amount,
            "credit_status": (
                credit_status
            ),
            "reason": reason,
            "current_balance": (
                updated_account[
                    "current_balance"
                ]
                if updated_account
                else None
            ),
        }


    except sqlite3.Error as exc:
        conn.rollback()

        return {
            "status": "error",
            "customer_id": customer_id,
            "message": (
                "Database error while processing "
                f"billing credit: {exc}"
            ),
        }


    finally:
        conn.close()


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_NAME = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5",
)


# ==================================================
# GOOGLE ADK BILLING AGENT
# ==================================================

root_agent = Agent(

    name="billing_resolution",

    model=LiteLlm(
        model=(
            f"anthropic/{MODEL_NAME}"
            if not MODEL_NAME.startswith(
                "anthropic/"
            )
            else MODEL_NAME
        )
    ),

    description=(
        "Remote telecom billing dispute "
        "and credit resolution agent."
    ),

    instruction="""
You are the Billing Resolution specialist
for a telecom support system.

Use the available SQLite-backed tools for all
billing facts and billing actions.

Responsibilities:

- Look up customer billing accounts.
- Review recent charges.
- Detect duplicate charges.
- Review recent billing credits.
- Review billing disputes.
- Apply valid billing credits.
- Respect billing approval limits.

Billing rules:

1. Credits of 50 USD or less may be APPLIED immediately.

2. Credits above 50 USD must be recorded as
   PENDING_APPROVAL.

3. PENDING_APPROVAL credits must NOT reduce the
   customer's current balance.

4. If an equivalent APPLIED credit already exists,
   do not create or apply the same credit again.

5. Duplicate charges may be identified either by:
   - is_duplicate_flag = 1
   OR
   - same customer, description and billing period.

6. Never invent billing information.

7. Always use tool results.

8. Keep responses concise and suitable for
   billing support staff.
""",

    tools=[
        lookup_billing_account,
        check_duplicate_charges,
        apply_billing_credit,
    ],
)


# ==================================================
# A2A SERVICE
# ==================================================

a2a_app = to_a2a(
    root_agent,
    host="localhost",
    port=8002,
)