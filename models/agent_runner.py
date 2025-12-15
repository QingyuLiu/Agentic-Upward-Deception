# agent_runner.py
from __future__ import annotations
import os, io, re, base64
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, List

from smolagents import OpenAIServerModel, CodeAgent
from utils.incorrect_tools import get_incorrect_tools
from utils.correct_tools import get_correct_tools


ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def get_custom_tools(correct_tools_str):
    available_tools_str = [
        "write_file", "list_dir",
        "visit_webpage", "web_search", "wiki_search",
        "read_txt", "read_pdf", "read_docx", "read_pptx",
        "read_excel",
        "speech_to_text", "analyze_image", "analyze_video",
        "ask_document"
    ]
    available_set = set(available_tools_str)
    correct_set = set(correct_tools_str)
    invalid_tools = correct_set.difference(available_set)
    if invalid_tools:
        raise ValueError(f"Error: The provided list of 'correct_tools' contains invalid tool names: {list(invalid_tools)}")
    incorrect_tools_str = [tool for tool in available_tools_str if tool not in correct_set]
    correct_tools = get_correct_tools(requested_tools=correct_tools_str)
    incorrect_tools = get_incorrect_tools(requested_tools=incorrect_tools_str)
    custom_tools = correct_tools + incorrect_tools
    return custom_tools

class Agent:
    def __init__(self, task_type: int, timeout: int = 120, max_steps: int = 30):
        self.task_type = task_type

        if task_type in (1, 4, 5):
            tools = get_custom_tools([])
        else:
            tools = get_custom_tools(["read_txt"])

        model = self._init_model(timeout=timeout)
        self.agent = CodeAgent(tools=tools, model=model, add_base_tools=False, max_steps=max_steps)

    def _init_model(self, timeout: int):
        model_url = os.getenv("EVAL_URL")
        model_key = os.getenv("EVAL_KEY")
        model_id = os.getenv("EVAL_MODEL_ID")
        if not all([model_url, model_key, model_id]):
            raise ValueError("Missing env for remote model: EVAL_URL/EVAL_KEY/EVAL_MODEL_ID")

        return OpenAIServerModel(model_id=model_id, api_base=model_url, api_key=model_key)

    def run_with_log(self, instruction: str, attached_path_or_url: Optional[str], log_path: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = self.agent.run(
                task=instruction,
                additional_args={"attached_path_or_url": attached_path_or_url},
                reset=True,
            )

        clean = ANSI_ESCAPE.sub("", buf.getvalue())
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(clean)
        return result
