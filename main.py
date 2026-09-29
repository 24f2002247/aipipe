import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout = StringIO()
    stderr = StringIO()

    sys.stdout = stdout
    sys.stderr = stderr

    try:
        exec(code)

        output = stdout.getvalue() + stderr.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def analyze_error_with_ai(code: str, error_output: str) -> List[int]:

    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze the following Python code and traceback.

Identify the exact source-code line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{error_output}

Return ONLY JSON in this format:

{{
    "error_lines": [3]
}}

If there are multiple relevant error lines, include them all.
"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # If code succeeds, do NOT call the AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # If code fails, ask AI for error line numbers
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }


@app.get("/")
def root():
    return {
        "status": "ok"
    }