import ast
import json
import re
import os
import numpy as np
import pandas as pd
import tempfile
import whisper
import subprocess
from base64 import b64encode
from functools import lru_cache
from io import BytesIO
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from helpers import  error_traceback
from prompts import (
    VISION_SYSTEM_PROMPT,
    EXCEL_SYSTEM_PROMPT,
)
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from config import (
    settings,
    ALLOWED_AST_OPS
)
import wikipedia
wikipedia.set_user_agent(
    "GAIA-Agent/1.0 (https://huggingface.co/)"
)

whisper_model = whisper.load_model("base")

# ============================= ARITHMETIC OPERATIONS ===================== #
def _safe_eval(node: ast.AST) -> float | int | complex:
    """Recursively evaluate a *restricted* AST expression tree."""
    if isinstance(node, ast.Constant):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_AST_OPS:
        return ALLOWED_AST_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_AST_OPS:
        return ALLOWED_AST_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    raise ValueError("Unsafe or unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate basic arithmetic expressions (no variables, functions)."""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree.body)
        return str(value)
    except Exception as exc:
        error_traceback(exc, "Calculator")
        return f"calc_error:{exc}"

# ============================= WEB AND WIKI SEARCH ===================== #
@lru_cache(maxsize=256)
def _ddg_search(query: str, k: int = 6) -> list[dict[str, str]]:
    """Cached DuckDuckGo JSON search."""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=k)
    hits = wrapper.results(query)
    return [
        {
            "title": hit.get("title", "")[:500],
            "snippet": hit.get("snippet", "")[:750],
            "link": hit.get("link", "")[:300],
        }
        for hit in hits[:k]
    ]


@tool
def web_multi_search(query: str, k: int = 6) -> str:
    """Run DuckDuckGo → Tavily fallback search. Returns JSON list[dict]."""
    try:
        hits = _ddg_search(query, k)
        if hits:
            return json.dumps(hits, ensure_ascii=False)
    except Exception:  # fall through to Tavily
        pass

    try:
        tavily_results = TavilySearchResults(
            max_results=5,
            tavily_api_key=settings.TAVILY_API_KEY
            # include_answer=True,
            # search_depth="advanced",
        )
        search_result = tavily_results.invoke({"query": query})
        print(
            f"[TOOL] TAVILY search is triggered with following response: {search_result}"
        )
        formatted = [
            {
                "title": d.get("title", "")[:500],
                "snippet": d.get("content", "")[:750],
                "link": d.get("url", "")[:300],
            }
            for d in search_result
        ]
        return json.dumps(formatted, ensure_ascii=False)
    except Exception as exc:
        error_traceback(exc, "Multi Search")
        return f"search_error:{exc}"


@tool
def wiki_search(query: str, max_pages: int = 2) -> str:
    """Lightweight wrapper on WikipediaLoader; returns concatenated page texts."""
    print(f"[TOOL] wiki_search called with query: {query}")
    docs = WikipediaLoader(query=query, load_max_docs=max_pages).load()
    joined = "\n\n---\n\n".join(d.page_content for d in docs)
    return joined[:8_000]


# ============================= OUTUBE  TRANSCRIPT ===================== #
# @tool
# def youtube_transcript(url: str, chars: int = 10_000) -> str:
#     """Fetch full YouTube transcript (first *chars* characters)."""
#     video_id_match = re.search(r"[?&]v=([A-Za-z0-9_\-]{11})", url)
#     if not video_id_match:
#         return "yt_error:id_not_found"
#
#     try:
#         video_id = video_id_match.group(1)
#
#         api = YouTubeTranscriptApi()
#         transcript = api.fetch(video_id)
#
#         text = " ".join(piece.text for piece in transcript)
#
#         return text[:chars]
#
#     except Exception as exc:
#         error_traceback(exc, "YouTube")
#         return f"yt_error:{exc}"

@tool
def youtube_transcript(url: str, chars: int = 10_000) -> str:
    """Fetch YouTube transcript. Falls back to audio transcription when captions are disabled."""

    video_id_match = re.search(
        r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_\-]{11})",
        url
    )

    if not video_id_match:
        return "yt_error:id_not_found"

    video_id = video_id_match.group(1)

    # ---------------------------------------------------------
    # 1. Try YouTube captions first
    # ---------------------------------------------------------
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        text = " ".join(piece.text for piece in transcript)

        return text[:chars]

    except Exception as exc:
        print(f"[YouTube] Caption unavailable: {exc}")
        print("[YouTube] Falling back to audio transcription...")

    # ---------------------------------------------------------
    # 2. Captions unavailable → download audio
    # ---------------------------------------------------------
    audio_path = None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            output_template = os.path.join(
                temp_dir,
                "youtube_audio.%(ext)s"
            )

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = ydl.prepare_filename(info)

            # -------------------------------------------------
            # 3. Whisper → speech to text
            # -------------------------------------------------
            result = whisper_model.transcribe(
                audio_path,
                fp16=False
            )

            text = result["text"].strip()

            if not text:
                return "yt_error:audio_transcription_empty"

            return text[:chars]

    except Exception as exc:
        error_traceback(exc, "YouTube Audio")
        return f"yt_error:audio_transcription_failed:{exc}"



@tool
def vision_task(img_bytes: bytes, question: str) -> str:
    """
    Pass the user's question AND the referenced image to a multimodal LLM and
    return its first line of text as the answer.  No domain assumptions made.
    """
    vision_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,  # Ensures deterministic output for chess moves
        max_tokens=2048,  # Fine for single moves; bump to 300+ if asking broader vision questions
        api_key=settings.OPENAI_API_KEY
    )
    try:
        image_b64 = b64encode(img_bytes).decode()
        messages = [
            SystemMessage(content=VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                {"type": "text", "text": question.strip()},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high"
                    },
                },
            ]
            ),
        ]
        raw_response = vision_llm.invoke(messages).content
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        clean_answer = re.sub(r'^(Answer:|\s*)+', '', clean_answer).strip()
        return clean_answer
    except Exception as exc:
        error_traceback(exc, "vision")
        return f"img_error:{exc}"

#============================= FILE  UTILS =====================#
@tool
def run_py(code: str) -> str:
    """Execute Python code in a sandboxed subprocess and return last stdout line."""
    try:
        with NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
            f.write(code)
            path = f.name
        proc = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=45
        )
        out = proc.stdout.strip().splitlines()
        return out[-1] if out else ""
    except Exception as exc:
        error_traceback(exc, "run_py")
        return f"py_error:{exc}"


@tool
def transcribe_via_whisper(audio_bytes: bytes) -> str:
    """Transcribe audio with Whisper (CPU)."""
    with NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        import whisper  # openai-whisper

        model = whisper.load_model("base")
        output = model.transcribe(path)["text"].strip()
        print(f"[DEBUG] Whisper transcript (first 200 chars): {output[:200]}")
        return output
    except Exception as exc:
        error_traceback(exc, "Whisper")
        return f"asr_error:{exc}"


@tool
def analyze_excel_file(xls_bytes: bytes, question: str) -> str:
    "Analyze Excel or CSV file by passing the data preview to LLM and getting the Python Pandas operation to run"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=64,api_key=settings.OPENAI_API_KEY)

    try:
        df = pd.read_excel(BytesIO(xls_bytes))
    except Exception:
        df = pd.read_csv(BytesIO(xls_bytes))

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].astype(float)

    # Ask the LLM for a single expression
    prompt = EXCEL_SYSTEM_PROMPT.format(
        question=question,
        preview=df.head(5).to_dict(orient="list"),
    )
    expr = llm.invoke(prompt).content.strip()

    # Run generated Pandas' one-line expression
    try:
        result = eval(expr, {"df": df, "pd": pd, "__builtins__": {}})
        # Normalize scalars to string
        if isinstance(result, np.generic):
            result = float(result)  # → plain Python float
            return f"{result:.2f}"  # or str(result) if no decimals needed

        # DataFrame / Series → single-line string
        return (
            result.to_string(index=False)
            if hasattr(result, "to_string")
            else str(result)
        )
    except Exception as e:
        error_traceback(e, "Excel")
        return f"eval_error:{e}"


__all__ = [
    "calculator",
    "web_multi_search",
    "wiki_search",
    "youtube_transcript",
    "vision_task",
    "run_py",
    "transcribe_via_whisper",
    "analyze_excel_file",
]