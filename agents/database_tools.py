import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def get_tower(tower_id):
    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM network_towers
            WHERE tower_id = ?
            """,
            (tower_id,)
        ).fetchone()

        return dict(row) if row else None

    finally:
        connection.close()


def get_tower_performance(tower_id):
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM tower_performance
            WHERE tower_id = ?
            ORDER BY recorded_at DESC
            """,
            (tower_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_open_incidents(tower_id=None):
    connection = get_connection()

    try:
        if tower_id:
            rows = connection.execute(
                """
                SELECT *
                FROM open_incidents
                WHERE tower_id = ?
                """,
                (tower_id,)
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM open_incidents
                """
            ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_customer_account(customer_id):
    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,)
        ).fetchone()

        return dict(row) if row else None

    finally:
        connection.close()


def get_customer_charges(customer_id):
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM billing_charges
            WHERE customer_id = ?
            ORDER BY charge_date DESC
            """,
            (customer_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()
def get_customer(customer_id):
    connection = get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM billing_accounts
        WHERE customer_id = ?
        """,
        (customer_id,)
    ).fetchone()

    connection.close()

    return dict(row) if row else None


def get_billing_charges(customer_id):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM billing_charges
        WHERE customer_id = ?
        ORDER BY charge_date DESC
        """,
        (customer_id,)
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_billing_credits(customer_id):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM billing_credits
        WHERE customer_id = ?
        ORDER BY created_at DESC
        """,
        (customer_id,)
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_billing_disputes(customer_id):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM billing_disputes
        WHERE customer_id = ?
        ORDER BY opened_at DESC
        """,
        (customer_id,)
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    print("Testing database tools...\n")

    tower = get_tower("TX-512")
    print("Tower:")
    print(tower)

    print("\nCustomer:")
    customer = get_customer_account("CUST-10002")
    print(customer)

    print("\nOpen incidents:")
    incidents = get_open_incidents("TX-512")
    print(incidents)