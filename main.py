from urllib import response

from typing_extensions import TypedDict
from langgraph.graph import START,END, StateGraph
from openai import OpenAI

class State(TypedDict):
    topic: str
    summary: str

client = OpenAI() # Reads OPENAI_API_KEY automatically

def write_summary(state: State):
    response = client.responses.create(
        model = 'gpt-5',
        instructions = 'You write consise, accurate explanations for beginners.',
        input = f'Explain {state['topic']} in 2 short sentences.',
    )

    return {"summary": response.output_text}

builder = StateGraph(State)
builder.add_node("write_summary", write_summary)
builder.add_edge(START, "write_summary")
builder.add_edge("write_summary", END)

graph = builder.compile()

result = graph.invoke({"topic":"How langgraph manages state"})

print(result["summary"])