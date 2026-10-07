from adk_services.network_diagnostics.agent import (
    check_tower_status,
    run_connectivity_diagnostics
)


def extract_tower_id(question):
    words = question.upper().replace("?", "").split()

    for word in words:
        if word.startswith("TX-"):
            return word

    return None


def answer_network_question(question):
    tower_id = extract_tower_id(question)

    if not tower_id:
        return {
            "status": "error",
            "message": "No tower ID found in the question."
        }

    question_lower = question.lower()

    if (
        "incident" in question_lower
        or "issue" in question_lower
        or "problem" in question_lower
        or "drop" in question_lower
        or "dropping" in question_lower
        or "why" in question_lower
    ):
        result = run_connectivity_diagnostics(
            tower_id,
            question
        )

        if result.get("status") != "success":
            return result

        tower = result.get("tower", {})
        incidents = result.get("incidents", [])
        performance = result.get("performance", [])

        if not incidents:
            return {
                "status": "success",
                "answer": (
                    f"Tower {tower_id} is currently {tower.get('status', 'UNKNOWN')}. "
                    f"No open incidents are currently associated with this tower."
                ),
                "data": result
            }

        incident = incidents[0]

        latest = performance[0] if performance else {}

        packet_loss = latest.get("packet_loss_pct")
        throughput = latest.get("downlink_throughput_mbps")
        signal = latest.get("signal_strength_dbm")

        answer = (
            f"Tower {tower_id} is currently {tower.get('status', 'UNKNOWN')}, "
            f"but there is a {incident.get('severity', 'UNKNOWN')} incident "
            f"under investigation. "
        )

        if packet_loss is not None:
            answer += f"Packet loss has increased to {packet_loss}%. "

        if throughput is not None:
            answer += (
                f"5G downlink throughput has dropped to "
                f"{throughput} Mbps. "
            )

        if signal is not None:
            answer += f"The latest signal strength is {signal} dBm. "

        description = incident.get("description")

        if description:
            if "Suspected" in description:
                suspected = description.split("Suspected", 1)[1]

                if "." in suspected:
                    suspected = suspected.split(".", 1)[0]

                answer += f"The suspected cause is {suspected.strip()}. "

        answer += (
            "The tower is not experiencing a full site outage."
        )

        return {
            "status": "success",
            "answer": answer,
            "data": result
        }

    result = check_tower_status(tower_id)

    if result.get("status") != "success":
        return result

    tower = result.get("tower", {})

    return {
        "status": "success",
        "answer": (
            f"Tower {tower_id} is currently "
            f"{tower.get('status', 'UNKNOWN')}."
        ),
        "data": result
    }


if __name__ == "__main__":
    questions = [
        "Is tower TX-512 operational?",
        "Why are there 5G session drops at TX-512?",
        "Are there any incidents for TX-512?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        result = answer_network_question(question)

        print("Answer:")
        print(result["answer"])