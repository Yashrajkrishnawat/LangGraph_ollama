"""A privacy-first LangGraph onboarding chatbot.

This demo deliberately does not collect Aadhaar numbers, PAN numbers, document
images, passwords, OTPs, or bank details. It keeps the onboarding data only in
memory while the program runs. Replace the placeholder KYC link with a
compliance-approved provider before using this with real customers.
"""

import re
from typing import Literal
from typing_extensions import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from ollama import chat


PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){11}\d(?!\d)")


class State(TypedDict, total=False):
    message: str
    stage: Literal["welcome", "contact", "consent", "verified"]
    name: str
    phone: str
    email: str
    consent_given: bool
    reply: str
    kyc_link: str


def contains_sensitive_id(message: str) -> bool:
    """Catch common Aadhaar and PAN formats before they reach the LLM."""
    return bool(AADHAAR_PATTERN.search(message) or PAN_PATTERN.search(message))


def safety_gate(state: State) -> dict:
    message = state["message"].strip()
    if contains_sensitive_id(message):
        return {
            "reply": (
                "For your safety, please do not send Aadhaar or PAN numbers, "
                "document images, passwords, OTPs, or bank details in chat. "
                "A secure verification link will be provided after consent."
            )
        }
    # Clear a previous blocked-message reply so the customer can continue safely.
    return {"reply": ""}


def route_message(state: State) -> dict:
    """Read only explicit onboarding commands; do not ask the LLM to extract PII."""
    message = state["message"].strip()
    lower = message.lower()

    if state.get("stage") == "welcome":
        return {
            "stage": "contact",
            "reply": (
                "Welcome. I can help begin your onboarding. Please provide only "
                "your name, mobile number, and email using: "
                "name: … | phone: … | email: …"
            ),
        }

    if state.get("stage") == "contact":
        fields = dict(re.findall(r"\b(name|phone|email)\s*:\s*([^|]+)", message, re.I))
        cleaned = {key.lower(): value.strip() for key, value in fields.items()}
        required = {"name", "phone", "email"}
        if not required.issubset(cleaned):
            return {
                "reply": (
                    "I still need name, phone, and email. Example: "
                    "name: Priya Sharma | phone: 9876543210 | email: priya@example.com"
                )
            }
        return {
            "name": cleaned["name"],
            "phone": cleaned["phone"],
            "email": cleaned["email"],
            "stage": "consent",
            "reply": (
                "Thanks. We will use these details only to begin your application "
                "and send a secure KYC link. Type exactly: consent: yes"
            ),
        }

    if state.get("stage") == "consent":
        if lower == "consent: yes":
            session_id = uuid4().hex
            link = f"https://kyc.example.com/session/{session_id}"
            return {
                "consent_given": True,
                "stage": "verified",
                "kyc_link": link,
                "reply": (
                    "Consent recorded. Open this demo-only secure KYC link: "
                    f"{link}\n\nIn a real product, this must be a link from your "
                    "approved KYC provider—not a page built into this chatbot."
                ),
            }
        return {"reply": "No consent was recorded. Type exactly: consent: yes"}

    return {"reply": "Your onboarding session is complete. Type quit to exit."}


def use_llm_for_general_questions(state: State) -> dict:
    message = state["message"].strip()

    response = chat(
        model="gemma3",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a warm, helpful financial-services onboarding chatbot. "
                    "Reply naturally, like ChatGPT, in 2 to 4 concise sentences. "
                    "Answer general questions and explain the onboarding process. "
                    "Never ask for or repeat Aadhaar, PAN, document images, passwords, "
                    "OTPs, bank details, or card numbers. Explain that secure KYC "
                    "happens through a separate verification link."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    return {
        "reply": response.message.content,
        "stage": "contact" if state.get("stage") == "welcome" else state.get("stage"),
    }


def choose_next_step(state: State) -> Literal["llm", "router", "end"]:
    if state.get("reply", "").startswith("For your safety"):
        return "end"

    message = state["message"].strip().lower()

    # Keep structured contact collection and consent deterministic.
    if "name:" in message and "phone:" in message and "email:" in message:
        return "router"

    if state.get("stage") == "consent" and message == "consent: yes":
        return "router"

    # All ordinary conversation goes to Ollama.
    return "llm"


builder = StateGraph(State)
builder.add_node("safety_gate", safety_gate)
builder.add_node("route_message", route_message)
builder.add_node("general_question", use_llm_for_general_questions)
builder.add_edge(START, "safety_gate")
builder.add_conditional_edges(
    "safety_gate",
    choose_next_step,
    {"llm": "general_question", "router": "route_message", "end": END},
)
builder.add_edge("route_message", END)
builder.add_edge("general_question", END)
graph = builder.compile()


if __name__ == "__main__":
    session: State = {"stage": "welcome"}
    print("Privacy-first onboarding chatbot. Type 'quit' to exit.\n")

    while True:
        message = input("You: ").strip()
        if message.lower() in {"quit", "exit"}:
            break

        session = {**session, "message": message}
        result = graph.invoke(session)
        session = {**session, **result}
        print(f"\nBot: {result['reply']}\n")
