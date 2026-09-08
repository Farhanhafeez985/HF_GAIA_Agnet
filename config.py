import ast
import operator
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "o4-mini",
    )

    TEMPERATURE = float(
        os.getenv("TEMPERATURE", "0.1")
    )

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    GAIA_API_URL = os.getenv(
        "GAIA_API_URL",
        "https://agents-course-unit4-scoring.hf.space",
    )

    SPACE_HOST = os.getenv("SPACE_HOST")
    SPACE_ID = os.getenv("SPACE_ID")


settings = Settings()


Label = Literal[
    "math",
    "youtube",
    "image",
    "code",
    "excel",
    "audio",
    "search",
    "reason",
]

ALLOWED_AST_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

class AgentState(TypedDict):
    question: str
    label: Label
    context: str
    answer: str
    task_id: str | None