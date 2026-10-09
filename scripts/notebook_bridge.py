"""Run an interactive script (scripts/translate_chat_pipeline.py) in ANOTHER Python (the Python 3.11 environment of the Kaggle notebooks)
while the questions are asked in the notebook cell. Kaggle's own kernel runs a newer Python than the pinned libraries support, so the
translator cannot be imported into the kernel; it runs as a child process and this function relays: child output -> cell output, and
every PROMPT line of the child -> input() in the cell -> child stdin.

    from notebook_bridge import run_interactive
    run_interactive(["/kaggle/temp/venv311/bin/python", "-u", "scripts/translate_chat_pipeline.py", "--adapter", ADAPTER])
"""
import os
import subprocess
import sys

PROMPT_MARK = "\x01PROMPT"


def run_interactive(cmd, env=None, ask=input, show=print):
    child_env = {**os.environ, **(env or {}), "CAT_BRIDGE": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env, text=True, bufsize=1,
                         encoding="utf-8", errors="replace")
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            if line == PROMPT_MARK:
                try:
                    answer = ask("> ")
                except (EOFError, KeyboardInterrupt):
                    answer = "quit"
                p.stdin.write(answer.replace("\n", " ") + "\n")
                p.stdin.flush()
            else:
                show(line)
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
        rc = p.wait()
    if rc:
        show(f"(the translator stopped with exit code {rc}; the lines above say why)")
    return rc
