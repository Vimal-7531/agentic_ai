import asyncio
from pathlib import Path
import truststore
truststore.inject_into_ssl()

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


NETWORK_AGENT_CARD = (
    "http://localhost:8001/.well-known/agent-card.json"
)

BILLING_AGENT_CARD = (
    "http://localhost:8002/.well-known/agent-card.json"
)


network_remote_agent = RemoteA2aAgent(
    name="network_diagnostics_remote",
    description="Remote network diagnostics A2A agent.",
    agent_card=NETWORK_AGENT_CARD,
)


billing_remote_agent = RemoteA2aAgent(
    name="billing_resolution_remote",
    description="Remote billing resolution A2A agent.",
    agent_card=BILLING_AGENT_CARD,
)


network_proxy_agent = Agent(
    name="network_proxy",

    model=LiteLlm(
        model="anthropic/claude-sonnet-4-5"
    ),

    description="Proxy for the remote network diagnostics service.",

    instruction="""
You are a proxy for the remote Network Diagnostics agent.

For network operational questions, delegate the request to
network_diagnostics_remote.

Return the remote agent's result clearly and do not invent operational data.
""",

    sub_agents=[
        network_remote_agent
    ],
)


billing_proxy_agent = Agent(
    name="billing_proxy",

    model=LiteLlm(
        model="anthropic/claude-sonnet-4-5"
    ),

    description="Proxy for the remote billing resolution service.",

    instruction="""
You are a proxy for the remote Billing Resolution agent.

For billing account, charge, duplicate-charge or credit questions,
delegate the request to billing_resolution_remote.

Return the remote agent's result clearly and do not invent billing data.
""",

    sub_agents=[
        billing_remote_agent
    ],
)


async def run_remote_agent(agent, message: str) -> str:

    runner = InMemoryRunner(
        agent=agent,
        app_name="prodapt_remote_client",
    )

    session = await runner.session_service.create_session(
        app_name="prodapt_remote_client",
        user_id="prodapt_user",
    )

    user_message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=message
            )
        ],
    )

    final_parts = []

    async for event in runner.run_async(
        user_id="prodapt_user",
        session_id=session.id,
        new_message=user_message,
    ):

        if event.content and event.content.parts:

            for part in event.content.parts:

                if part.text:
                    final_parts.append(part.text)

    if not final_parts:
        return "No response received from remote agent."

    return final_parts[-1]


async def run_network_diagnostics_async(
    question: str
) -> str:

    return await run_remote_agent(
        network_proxy_agent,
        question
    )


async def run_billing_resolution_async(
    question: str
) -> str:

    return await run_remote_agent(
        billing_proxy_agent,
        question
    )


def run_network_diagnostics_remote(
    question: str
) -> str:

    return asyncio.run(
        run_network_diagnostics_async(question)
    )


def run_billing_resolution_remote(
    question: str
) -> str:

    return asyncio.run(
        run_billing_resolution_async(question)
    )


if __name__ == "__main__":

    print("\nNETWORK REMOTE TEST")

    network_result = run_network_diagnostics_remote(
        "Check tower TX-512 and explain why customers may be experiencing repeated 5G session drops."
    )

    print(network_result)

    print("\nBILLING REMOTE TEST")

    billing_result = run_billing_resolution_remote(
        "Does customer CUST-10002 have any duplicate billing charges?"
    )

    print(billing_result)