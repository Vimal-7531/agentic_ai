import os
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


llm = LLM(
    model="anthropic/claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


communications_specialist = Agent(
    role="Customer Communications Specialist",

    goal=(
        "Turn technical telecom investigation results into a clear, accurate, "
        "customer-friendly response."
    ),

    backstory=(
        "You specialize in explaining telecom network, billing and policy issues "
        "to customers without exposing unnecessary internal implementation details."
    ),

    llm=llm,
    verbose=False,
    allow_delegation=False,
)


quality_reviewer = Agent(
    role="Customer Response Quality Reviewer",

    goal=(
        "Review customer responses for factual accuracy, clarity, professionalism "
        "and compliance with the supplied agent context."
    ),

    backstory=(
        "You are responsible for ensuring that customer-facing telecom responses "
        "do not invent facts, contradict policy, or expose unnecessary internal details."
    ),

    llm=llm,
    verbose=False,
    allow_delegation=False,
)


def create_customer_response(
    user_query: str,
    agent_context: str,
) -> str:

    draft_task = Task(
        description=f"""
Create a concise customer-ready response.

ORIGINAL CUSTOMER QUERY:
{user_query}

AGENT CONTEXT:
{agent_context}

Rules:
- Use only facts contained in AGENT CONTEXT.
- Do not invent network, billing or policy information.
- Do not mention internal frameworks such as LangGraph, ADK, CrewAI or LlamaIndex.
- Explain technical information in clear customer-friendly language.
- Preserve important prices, amounts, IDs and operational values accurately.
- If the context does not contain enough information, state that clearly.
""",

        expected_output=(
            "A clear and concise customer-facing response based only on "
            "the supplied agent context."
        ),

        agent=communications_specialist,
    )


    review_task = Task(
        description="""
Review the drafted customer response.

Check:
- factual accuracy
- consistency with the supplied context
- clear professional tone
- no hallucinated facts
- no unnecessary internal system details
- correct prices, amounts, IDs and technical values

Return ONLY the corrected final customer response.
Do not include review notes or commentary.
""",

        expected_output=(
            "The final corrected customer-facing response only."
        ),

        agent=quality_reviewer,
        context=[draft_task],
    )


    crew = Crew(
        agents=[
            communications_specialist,
            quality_reviewer,
        ],

        tasks=[
            draft_task,
            review_task,
        ],

        process=Process.sequential,
        verbose=False,
    )


    result = crew.kickoff()

    return str(result)


if __name__ == "__main__":

    test_query = (
        "Why am I experiencing repeated 5G session drops near tower TX-512?"
    )

    test_context = """
Tower TX-512 is currently OPERATIONAL.

Latest network measurements:
- Packet loss: 3.8%
- Latency: 58 ms
- Downlink throughput: 85 Mbps
- Signal strength: -72 dBm

There is an open MAJOR incident INC-8841 currently under investigation.
The suspected cause is radio scheduler or backhaul congestion.
The issue is not classified as a full tower outage.
"""

    response = create_customer_response(
        test_query,
        test_context,
    )

    print("\nFINAL CUSTOMER RESPONSE:")
    print(response)