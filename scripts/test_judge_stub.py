"""Test of scripts/run_context_llm.py against a FAKE OpenAI-compatible server (no network, no API key): rate-limit errors (429) are retried,
answers in different formats (plain, markdown headers, numbered lists, inline) are parsed like GPT-4's, an unparsable answer gives NaN (never a
perfect score), and a second run is answered completely from the cache.

  python scripts/test_judge_stub.py <run_root>        # run_root contains results_zh/bmeld/test.en-zh.csv (scripts/consolidate_zh_results.py)
"""
import hashlib, json, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, pandas as pd

ANSWERS = [
    ('Critical:\nno-error\nMajor:\naccuracy/mistranslation - "x"\nMinor:\nno-error', -5),
    ('**Critical:**\n- accuracy/omission - "a"\n**Major:**\nno-error\n**Minor:**\nstyle/awkward - "b"\n- fluency/grammar - "c"', -12),
    ('Critical: no-error\nMajor: no-error\nMinor: no-error', 0),
    ('I cannot evaluate this.', np.nan),
    ('Here is my analysis:\n\nMajor:\n1. accuracy/mistranslation - "q"\n2. fluency/register - "r"\nMinor:\nno-error', -10),
]
seen, lock, served = {}, threading.Lock(), []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "stub" and "top_p" not in body, body.keys()      # minimal parameters for non-OpenAI endpoints
        h = hashlib.sha1(json.dumps(body["messages"], sort_keys=True).encode()).hexdigest()
        with lock:
            n = seen.get(h, 0); seen[h] = n + 1
            idx = len(set(served)); 
            if n >= 1: served.append(h)
        if n == 0:                                         # every prompt is rejected once with a rate limit error
            self.send_response(429); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "rate limited", "code": 429}}).encode()); return
        order = list(dict.fromkeys(served)).index(h)
        ans = ANSWERS[order % len(ANSWERS)][0]
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": "stub",
               "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ans}}]}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(out).encode())

srv = ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

R, data_dir = sys.argv[1], sys.argv[1] + "/results_zh"
cmd = [sys.executable, str(__import__("pathlib").Path(__file__).resolve().parent / "run_context_llm.py"), "--base_url", f"http://127.0.0.1:{port}/v1", "--judge_model", "stub", "--model_name", "stub",
       "--lp", "zh", "--dataset", "bmeld", "--split", "test", "--data_dir", data_dir, "--tgt_col", "mbr-source", "--max_workers", "1",
       "--cache_file", f"{R}/judge_cache.jsonl", "--context_size", "4"]
import os
env = {**os.environ, "OPENAI_API_KEY": "test-key-not-real"}
r = subprocess.run(cmd, capture_output=True, text=True, env=env)
print((r.stdout + r.stderr)[-700:])
assert r.returncode == 0, "judge failed"
out = pd.read_csv(f"{data_dir}/bmeld/test.en-zh-mbr-source.gemba-stub.csv", keep_default_na=False)
scores = pd.to_numeric(out["stub-score"], errors="coerce").tolist()
exp = [ANSWERS[i % len(ANSWERS)][1] for i in range(len(out))]
print("scores  :", scores); print("expected:", exp)
assert all((np.isnan(a) and np.isnan(b)) or a == b for a, b in zip(scores, exp)), "scores differ"
print("JUDGE TEST OK:", len(out), "rows; every first request was rate limited and retried; markdown / numbered / inline answers parsed; unparsable -> NaN")
# resume: a second run must need no requests at all (all answers are cached)
before = len(seen)
r2 = subprocess.run(cmd, capture_output=True, text=True, env=env); assert r2.returncode == 0
assert len(seen) == before and all(v == 2 for v in seen.values()), "cache not used"
print("RESUME TEST OK: the second run answered everything from the cache")
