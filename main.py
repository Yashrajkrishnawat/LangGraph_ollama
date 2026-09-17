from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from ollama import chat


class State(TypedDict):
    topic: str
    summary: str


def write_summary(state: State):
    response = chat(
        model="gemma3",
        messages=[
            {
                "role": "system",
                "content": "You write concise, accurate explanations for beginners.",
            },
            {
                "role": "user",
                "content": f"Explain {state['topic']} in 2 short sentences.",
            },
        ],
    )

    return {"summary": response.message.content}


builder = StateGraph(State)
builder.add_node("write_summary", write_summary)
builder.add_edge(START, "write_summary")
builder.add_edge("write_summary", END)

graph = builder.compile()

result = graph.invoke({"topic": "What is an LLM"})
print(result["summary"])