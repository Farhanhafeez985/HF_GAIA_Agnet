import os

import gradio as gr
import pandas as pd
import requests

from config import settings, AgentState
from workflow import build_graph


class GAIAAgent:
    """Wrapper around the LangGraph agent."""

    def __init__(self) -> None:
        self.graph = build_graph()

    def __call__(
            self,
            question: str,
            task_id: str | None = None,
    ) -> str:
        state: AgentState = {
            "question": question,
            "label": "reason",
            "context": "",
            "answer": "",
            "task_id": task_id,
        }

        final = self.graph.invoke(state)

        print(
            f"[DEBUG] route='{final['label']}' "
            f"| LLM_used={final['label'] not in {'math', 'code', 'excel', 'image'} }"
        )

        return final["answer"]


def run_and_submit_all(
        profile: gr.OAuthProfile | None,
) -> tuple[str, pd.DataFrame | None]:
    if not profile:
        return (
            "Please Login to Hugging Face with the button.",
            None,
        )

    username = profile.username

    questions_url = f"{settings.GAIA_API_URL}/questions"
    submit_url = f"{settings.GAIA_API_URL}/submit"

    # ------------------------------------------------------------------
    # Initialize agent
    # ------------------------------------------------------------------

    try:
        agent = GAIAAgent()
        print("[DEBUG] GAIA Agent initialized successfully")

    except Exception as exc:
        print(f"[ERROR] Agent initialization failed: {exc}")
        return f"Error initializing agent: {exc}", None

    # ------------------------------------------------------------------
    # Fetch questions
    # ------------------------------------------------------------------

    print(f"[DEBUG] Fetching questions from: {questions_url}")

    try:
        response = requests.get(
            questions_url,
            timeout=15,
        )
        response.raise_for_status()

        questions = response.json()

        if not questions:
            return "No questions received from GAIA.", None

        print(f"[DEBUG] Fetched {len(questions)} questions")

    except requests.RequestException as exc:
        return f"Error fetching questions: {exc}", None

    except ValueError as exc:
        return f"Invalid questions response: {exc}", None

    # ------------------------------------------------------------------
    # Run agent
    # ------------------------------------------------------------------

    answers = []
    results = []

    for item in questions:
        task_id = item.get("task_id")
        question = item.get("question")

        if not task_id or question is None:
            print(f"[DEBUG] Skipping invalid task: {item}")
            continue

        print(f"\n[DEBUG] Processing task: {task_id}")
        print(f"[QUESTION] {question}")

        try:
            answer = agent(
                question=question,
                task_id=task_id,
            )

            answers.append(
                {
                    "task_id": task_id,
                    "submitted_answer": answer,
                }
            )

            results.append(
                {
                    "Task ID": task_id,
                    "Question": question,
                    "Submitted Answer": answer,
                }
            )

        except Exception as exc:
            print(
                f"[ERROR] Task {task_id} failed: {exc}"
            )

            results.append(
                {
                    "Task ID": task_id,
                    "Question": question,
                    "Submitted Answer": f"AGENT ERROR: {exc}",
                }
            )

    if not answers:
        return (
            "Agent did not produce any answers.",
            pd.DataFrame(results),
        )

    # ------------------------------------------------------------------
    # Submit answers
    # ------------------------------------------------------------------

    space_id = os.getenv("SPACE_ID")

    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else ""
    )

    submission = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers,
    }

    print(
        f"[DEBUG] Submitting {len(answers)} answers"
    )

    try:
        response = requests.post(
            submit_url,
            json=submission,
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        status = (
            f"Submission Successful!\n"
            f"User: {result.get('username')}\n"
            f"Overall Score: {result.get('score', 'N/A')}% "
            f"({result.get('correct_count', '?')}/"
            f"{result.get('total_attempted', '?')} correct)\n"
            f"Message: {result.get('message', 'No message received.')}"
        )

        return status, pd.DataFrame(results)

    except requests.HTTPError as exc:
        response = exc.response

        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text[:500]

        return (
            f"Submission Failed: "
            f"{response.status_code} - {detail}",
            pd.DataFrame(results),
        )

    except requests.Timeout:
        return (
            "Submission Failed: request timed out.",
            pd.DataFrame(results),
        )

    except requests.RequestException as exc:
        return (
            f"Submission Failed: {exc}",
            pd.DataFrame(results),
        )


# ----------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation")

    gr.Markdown(
        """
        1. Login to your Hugging Face account.
        2. Click **Run Evaluation & Submit All Answers**.
        3. The agent will process all GAIA tasks and submit the answers.
        """
    )

    gr.LoginButton()

    run_button = gr.Button(
        "Run Evaluation & Submit All Answers"
    )

    status_output = gr.Textbox(
        label="Run Status / Submission Result",
        lines=5,
        interactive=False,
    )

    results_table = gr.DataFrame(
        label="Questions and Agent Answers",
        wrap=True,
    )

    run_button.click(
        fn=run_and_submit_all,
        outputs=[
            status_output,
            results_table,
        ],
    )

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)


# For Testing The Agent Locally....

# if __name__ == "__main__":
#     agent = GAIAAgent()
#
#     while True:
#         try:
#             task = {
#                 "task_id": "f918266a-b3e0-4914-865d-4faa564f1aef",
#                 "question": "What is the final numeric output from the attached Python code?",
#                 "Level": "1",
#                 "file_name": "f918266a-b3e0-4914-865d-4faa564f1aef.py"
#             }
#             if not task["task_id"]:
#                 break
#             if not task["question"]:
#                 continue
#
#             print("Answer:", agent(
#                 question=task["question"],
#                 task_id=task["task_id"],
#             ))
#
#         except KeyboardInterrupt:
#             break
