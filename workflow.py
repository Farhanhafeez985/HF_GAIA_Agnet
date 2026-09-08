import mimetypes
import re
from typing import get_args

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import (
    AgentState,
    Label,
    settings
)
from gaia_files import get_file
from helpers import  sniff_excel_type
from prompts import (
    ROUTER_PROMPT,
    FINAL_LLM_SYSTEM_PROMPT,
    FINAL_LLM_USER_PROMPT,
)
from tools import (
    analyze_excel_file,
    calculator,
    run_py,
    transcribe_via_whisper,
    vision_task,
    web_multi_search,
    wiki_search,
    youtube_transcript,
)

_llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    # temperature=settings.TEMPERATURE,
)


def route_question(state: AgentState) -> AgentState:
    """Classify the question into a tool category."""

    label_values = get_args(Label)

    prompt = ROUTER_PROMPT.format(
        question=state["question"],
        labels=", ".join(repr(label) for label in label_values),
    )

    response = _llm.invoke(prompt).content.strip().lower()
    print(f"[DEBUG] ROUTER RAW RESPONSE >>> {response}")
    print(f"[DEBUG] ALLOWED LABELS >>> {label_values}")

    if response in label_values:
        state["label"] = response
    else:
        state["label"] = "reason"
    print(f"[DEBUG] FINAL LABEL >>> {state['label']}")
    return state


def invoke_tools(state: AgentState) -> AgentState:
    """Execute the tool required for the classified task."""
    print(f"[DEBUG]  STATE IN INVOKE_TOOLS >>> {state}")
    question = state["question"]
    label = state["label"]
    task_id = state["task_id"]

    # Attachment
    if task_id:
        file_path = get_file(task_id)

        if file_path:
            print(f"[DEBUG] attachment path={file_path}")

            with open(file_path, "rb") as file:
                blob = file.read()

            content_type = (
                    mimetypes.guess_type(file_path)[0]
                    or "application/octet-stream"
            )

            print(f"[DEBUG] attachment type={content_type}")

            if "python" in content_type:
                print("[TOOL] run_py")

                state["answer"] = run_py.invoke(
                    {"code": blob.decode("utf-8")}
                )
                state["label"] = "code"
                return state

            file_type = sniff_excel_type(blob)

            if (
                    any(key in content_type for key in ("excel", "sheet", "csv"))
                    or file_type in {"xlsx", "xls", "csv"}
            ):
                print("[TOOL] analyze_excel_file")

                state["answer"] = analyze_excel_file.invoke(
                    {
                        "xls_bytes": blob,
                        "question": question,
                    }
                )
                state["label"] = "excel"
                return state

            if "audio" in content_type:
                print("[TOOL] transcribe_via_whisper")

                state["context"] = transcribe_via_whisper.invoke(
                    {"audio_bytes": blob}
                )
                state["label"] = "audio"
                return state

            if "image" in content_type:
                print("[TOOL] vision_task")

                state["answer"] = vision_task.invoke(
                    {
                        "img_bytes": blob,
                        "question": question,
                    }
                )
                state["label"] = "image"
                return state

    # Math
    if label == "math":
        print("[TOOL] calculator")

        expression = re.sub(r"\s+", "", question)

        state["answer"] = calculator.invoke(
            {"expression": expression}
        )
        return state

    # YouTube
    if label == "youtube":
        url_match = re.search(r"https?://\S+", question)

        if url_match:
            print("[TOOL] youtube_transcript")

            state["context"] = youtube_transcript.invoke(
                {"url": url_match.group(0)}
            )

        return state

    # Web search
    if label == "search":
        print("[TOOL] web search")

        # search_result = web_multi_search.invoke(
        #     {"query": question}
        # )

        wiki_result = wiki_search.invoke(
            {"query": question}
        )

        # state["context"] = f"{search_result}\n\n{wiki_result}"
        state["context"] = wiki_result
        return state

    print("[TOOL] reasoning only")
    state["context"] = ""

    return state


def synthesize_response(state: AgentState) -> AgentState:
    """Generate the final response when a tool has not already produced it."""

    if state["label"] in {"code", "excel", "image", "math"}:
        print(
            f"[DEBUG] ANSWER ({state['label']}) >>> "
            f"{state['answer']}"
        )
        return state

    prompt = [
        SystemMessage(
            content=FINAL_LLM_SYSTEM_PROMPT
        ),
        HumanMessage(
            content=FINAL_LLM_USER_PROMPT.format(
                question=state["question"],
                context=state["context"],
            )
        ),
    ]

    state["answer"] = _llm.invoke(prompt).content.strip()
    return state


def format_output(state: AgentState) -> AgentState:
    """Clean and normalize the final answer."""

    answer = re.sub(
        r"^final answer:?\s*",
        "",
        state["answer"],
        flags=re.IGNORECASE,
    ).strip()

    if any(
            keyword in state["question"].lower()
            for keyword in ("first name", "single word")
    ):
        answer = answer.split()[0]

    state["answer"] = answer.rstrip(".")

    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route_question", route_question)
    graph.add_node("invoke_tools", invoke_tools)
    graph.add_node("synthesize_response", synthesize_response)
    graph.add_node("format_output", format_output)

    graph.set_entry_point("route_question")

    graph.add_edge("route_question", "invoke_tools")
    graph.add_edge("invoke_tools", "synthesize_response")
    graph.add_edge("synthesize_response", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
