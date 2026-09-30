"""Minimal OpenAI-compatible /v1/chat/completions server around a local
HuggingFace causal LM (greedy decoding). Useful to run
scripts/run_context_llm.py (GEMBA/ContextMQM) with a free model when vLLM's
own OpenAI server (`vllm serve <model>`, recommended) is not available.

  python scripts/serve_openai_compatible.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
  python scripts/run_context_llm.py --base_url http://localhost:8000/v1 --judge_model Qwen/Qwen2.5-7B-Instruct ...
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import HFBackend  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--default_max_tokens", type=int, default=256)
    args = p.parse_args()

    import uvicorn
    from fastapi import FastAPI

    llm = HFBackend(args.model)
    tok = llm.get_tokenizer()
    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat(req: dict):
        messages = req["messages"]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        max_tokens = int(req.get("max_tokens") or args.default_max_tokens)
        text = llm.generate([prompt], temperature=0.0, max_tokens=max_tokens, use_tqdm=False)[0]
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", args.model),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
