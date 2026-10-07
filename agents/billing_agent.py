from database_tools import (
    get_customer,
    get_billing_charges,
    get_billing_credits,
    get_billing_disputes
)


def extract_customer_id(question):
    words = question.upper().replace("?", "").split()

    for word in words:
        if word.startswith("CUST-"):
            return word

    return None


def answer_billing_question(question):
    customer_id = extract_customer_id(question)

    if not customer_id:
        return {
            "status": "error",
            "message": "No customer ID found in the question."
        }

    customer = get_customer(customer_id)

    if not customer:
        return {
            "status": "error",
            "message": f"Customer {customer_id} was not found."
        }

    text = question.lower()

    if "balance" in text:
        return {
            "status": "success",
            "answer": (
                f"{customer_id} has a current balance of "
                f"{customer['current_balance']:.2f} {customer['currency']}."
            )
        }

    if "auto-pay" in text or "autopay" in text or "auto pay" in text:
        enabled = bool(customer["auto_pay_enabled"])

        return {
            "status": "success",
            "answer": (
                f"Auto-pay is {'enabled' if enabled else 'disabled'} "
                f"for {customer_id}."
            )
        }

    if "charge" in text or "charges" in text:
        charges = get_billing_charges(customer_id)

        if not charges:
            answer = f"No billing charges were found for {customer_id}."
        else:
            total = sum(charge["amount"] for charge in charges)

            answer = (
                f"{customer_id} has {len(charges)} billing charge(s) "
                f"totaling {total:.2f} {customer['currency']}."
            )

        return {
            "status": "success",
            "charges": charges,
            "answer": answer
        }

    if "credit" in text or "credits" in text:
        credits = get_billing_credits(customer_id)

        if not credits:
            answer = f"No billing credits were found for {customer_id}."
        else:
            total = sum(credit["amount"] for credit in credits)

            answer = (
                f"{customer_id} has {len(credits)} billing credit(s) "
                f"totaling {total:.2f} {customer['currency']}."
            )

        return {
            "status": "success",
            "credits": credits,
            "answer": answer
        }

    if "dispute" in text or "disputes" in text:
        disputes = get_billing_disputes(customer_id)

        if not disputes:
            answer = f"No billing disputes were found for {customer_id}."
        else:
            answer = (
                f"{customer_id} has {len(disputes)} billing dispute(s)."
            )

        return {
            "status": "success",
            "disputes": disputes,
            "answer": answer
        }

    return {
        "status": "success",
        "answer": (
            f"{customer_id} is an active "
            f"{customer['account_type']} account."
        )
    }


if __name__ == "__main__":
    questions = [
        "What is the balance of CUST-10002?",
        "Is auto-pay enabled for CUST-10002?",
        "What charges does CUST-10002 have?",
        "Does CUST-10002 have any credits?",
        "Are there any billing disputes for CUST-10002?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        result = answer_billing_question(question)

        print("Answer:")
        print(result["answer"])