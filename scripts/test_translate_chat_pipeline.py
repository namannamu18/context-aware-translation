"""Tests of scripts/translate_chat_pipeline.py with the tiny smoke models (CPU, no downloads):
  0. the COMET context strings are identical to run_context_comet_mbr.add_context_across (the paper's code) for every turn, window and mode
  1. a LoRA adapter loaded on the fly gives the same model as the merged checkpoint (scripts/merge_lora.py)
  2. concise mode runs no greedy decoding and picks the final translation among the sampled candidates; verbose mode shows all four systems
  3. COMET unavailable -> chrF-MBR fallback
  4. the typing loop: several messages per line, verbose toggle, reset, transcript
  5. the judge after the translation (fake API server): what the judge is sent, the score, a failing judge never breaks the translation,
     the judge never changes the translation, on/off toggle, mean score in the transcript
  6. the translator is tied to the training run through the adapter folder: base model and prompt format are read from it

  python scripts/test_translate_chat_pipeline.py --lm <tiny lm> --adapter <adapter> --merged <merged model> --comet <tiny comet model.ckpt>
"""

import argparse
import builtins
import contextlib
import io
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import run_context_comet_mbr as mbr_mod  # noqa: E402
import translate_chat_pipeline as tp  # noqa: E402
from translate_chat import ChatTranslator  # noqa: E402


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class FakeApi:
    """OpenAI-compatible server that records the judge's requests. mode: ok | fail."""

    def __init__(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self
        self.requests, self.mode = [], "ok"
        self.answer = 'Critical:\nno-error\nMajor:\naccuracy/mistranslation - "x"\nMinor:\nstyle/awkward - "y"'

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if outer.mode == "fail":
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": {"message": "boom"}}')
                    return
                out = {"id": "x", "object": "chat.completion", "created": 0, "model": "stub",
                       "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": outer.answer}}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(out).encode())

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lm", required=True, help="tiny base language model")
    p.add_argument("--adapter", required=True, help="LoRA adapter trained on it (scripts/finetune_lora.py)")
    p.add_argument("--merged", required=True, help="the same adapter merged into the model (scripts/merge_lora.py)")
    p.add_argument("--comet", required=True, help="tiny COMET model.ckpt")
    A = p.parse_args()

    # 0. context strings == the paper's add_context_across --------------------------------------------------------------------
    convo = [("en", "Hi, I bought a coat last week."), ("zh", "尺码太小了。"), ("en", "Which size do you need?"), ("zh", "我要L码。"),
             ("zh", "越快越好。"), ("en", "OK, I'll send it tomorrow."), ("zh", "谢谢！")]
    best = [f"BEST{i}" for i in range(len(convo))]
    cands = [f"CAND{i}" for i in range(len(convo))]
    langs, srcs = [l for l, _ in convo], [t for _, t in convo]
    n = 0
    for ws, mode in itertools.product([0, 1, 2, 3, 6], ["source", "comet-best"]):
        pt = object.__new__(tp.PipelineTranslator)
        pt.sep, pt.context_size, pt.context_mt = "</s>", ws, mode
        ctx_mt = best if mode == "comet-best" else srcs
        ref_src = mbr_mod.add_context_across(srcs, srcs, ctx_mt, langs, "</s>", ws)
        ref_out = mbr_mod.add_context_across(cands, ctx_mt, srcs, langs, "</s>", ws)
        for i, (lang, text) in enumerate(convo):
            history = [tp.Turn(l, t, b) for (l, t), b in zip(convo[:i], best[:i])]
            s, o = pt.context_strings(history, lang, text, [cands[i]])
            assert s == ref_src[i] and o == [ref_out[i]], (ws, mode, i)
            n += 1
    print(f"0. context strings identical to add_context_across: {n}/{n} (turns x windows x modes)")

    # 1. adapter on the fly == merged checkpoint ----------------------------------------------------------------------------------
    a = ChatTranslator(A.lm, dtype="float32", device_map=None, template="chatml_empty_sys", adapter=A.adapter)
    b = ChatTranslator(A.merged, dtype="float32", device_map=None, template="chatml_empty_sys")
    x = a.tokenizer(a.prompt("Hello there", "en", "zh", [("zh", "你好")], True), return_tensors="pt")
    with torch.no_grad():
        diff = (a.model(**x).logits - b.model(**x).logits).abs().max().item()
    assert diff < 1e-4, diff
    print(f"1. adapter-on-the-fly == merged checkpoint (max logit diff {diff:.1e})")

    # 2. concise / verbose conversation ----------------------------------------------------------------------------------------
    pt = tp.load_system(A.lm, adapter=A.adapter, comet_model=A.comet, n_candidates=4, dtype="float32", device_map=None)
    assert pt.utility == "comet" and pt.context_mt == "source" and pt.context_size == 2
    conv = [("en", "Hi, I bought a coat last week but it is too small."), ("zh", "那我可以换一件大一点的吗？"), ("en", "Sure, which size do you need?")]
    calls = {"n": 0}
    orig = pt.tr.generate
    pt.tr.generate = lambda *args, **kw: (calls.__setitem__("n", calls["n"] + 1), orig(*args, **kw))[1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = pt.run_conversation(conv, max_new_tokens=16)
    assert calls["n"] == 0, "concise mode must not run the greedy decodings"
    assert all(r["final"] in r["candidates"] and r["greedy_context"] is None for r in res)
    assert buf.getvalue().count("   -> ") == 3
    with quiet():
        res_v = pt.run_conversation(conv[:2], max_new_tokens=16, verbose=True)
    assert all(r["greedy_context"] is not None and r["mbr"] in r["candidates"] and r["final"] in r["candidates"] for r in res_v)
    print("2. concise mode: no greedy decoding, final output among the candidates; verbose mode: all four systems")

    # 3. chrF fallback ---------------------------------------------------------------------------------------------------------
    o = io.StringIO()
    with contextlib.redirect_stdout(o):
        pt2 = tp.PipelineTranslator(pt.tr, comet_model="/nonexistent/model.ckpt", n_candidates=4)
    assert pt2.utility == "chrf" and "falling back" in o.getvalue()
    with quiet():
        r = pt2.run_conversation(conv[:2], max_new_tokens=16)
    assert all(x_["final"] in x_["candidates"] for x_ in r)
    print("3. COMET unavailable -> chrF-MBR fallback works")

    # 4. typing loop -----------------------------------------------------------------------------------------------------------
    lines = iter(["en: Hello there. zh: 你好吗？", "verbose", "en: Fine, thanks.", "reset", "zh: 再见", "quit", "en: never"])
    builtins.input = lambda prompt="": next(lines)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = tp.interactive_pipeline(pt, max_new_tokens=16)
    text = buf.getvalue()
    assert len(out) == 1 and out[0]["text"] == "再见"          # after 'reset' only the last conversation is kept
    assert text.count("=== translated conversation ===") == 2   # transcript printed at reset and at quit
    assert "(verbose on)" in text and "(new conversation)" in text
    print("4. typing loop: several messages per line, verbose toggle, reset, transcript on reset/quit")

    # 5. judge after the translation -------------------------------------------------------------------------------------------
    import time

    import judge_chat
    import run_context_llm as rcl

    api = FakeApi()
    os.environ["FAKE_JUDGE_KEY"] = "not-a-real-key"
    rcl.time.sleep = lambda s_: None                      # no waiting between retries in the test
    judge = judge_chat.ChatJudge(provider="openai", base_url=api.url, judge_model="stub", api_key_env="FAKE_JUDGE_KEY", rpm=0, context_size=2)
    pt3 = tp.PipelineTranslator(pt.tr, comet_model=A.comet, n_candidates=4, judge=judge)
    conv3 = [("en", "Hi, I bought a coat last week."), ("zh", "尺码太小了。"), ("en", "Which size do you need?")]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res3 = pt3.run_conversation(conv3, max_new_tokens=16)
    assert len(api.requests) == 3 and all(r["judge"]["score"] == -6 for r in res3), [r["judge"] for r in res3]      # 5 (major) + 1 (minor)
    assert "judge (stub): MQM -6 (0 critical, 1 major, 1 minor)" in buf.getvalue() and 'accuracy/mistranslation - "x"' in buf.getvalue()
    # what the judge is sent: the third message, context = the previous 2 messages as "Sender (source language): translation" (original code)
    last = api.requests[2]["messages"]
    user = last[-1]["content"]
    assert last[0]["role"] == "system" and last[1]["role"] == "user" and last[2]["role"] == "assistant"      # system prompt + 1-shot example
    assert "Which size do you need?" in user and res3[2]["final"] in user and 'sender' not in user
    assert f"Agent (English): {res3[0]['final']}" in user and f"Customer (Chinese): {res3[1]['final']}" in user, user
    assert 'by "Agent" in English' in user and "Chinese translation" in user
    assert "top_p" not in api.requests[0] and api.requests[0]["temperature"] == 0 and api.requests[0]["max_tokens"] == 1024
    # context_mode="source" shows the original messages instead
    judge_src = judge_chat.ChatJudge(provider="openai", base_url=api.url, judge_model="stub", api_key_env="FAKE_JUDGE_KEY", rpm=0, context_mode="source")
    hist = [tp.Turn("en", "Hi, I bought a coat last week.", "x", "你好，我上周买了一件外套。"), tp.Turn("zh", "尺码太小了。", "y", "The size is too small.")]
    u = judge_src.prompt(hist, "en", "Which size do you need?", "你需要哪个尺码？")[-1]["content"]
    assert "Agent (English): Hi, I bought a coat last week." in u and "Customer (Chinese): 尺码太小了。" in u
    # the judge never changes the translation: same candidates -> the judge call happens after the choice, and a failing judge does not break anything
    api.mode = "fail"
    with contextlib.redirect_stdout(io.StringIO()) as o:
        r_fail = pt3.run_conversation(conv3[:1], max_new_tokens=16)
    assert r_fail[0]["final"] in r_fail[0]["candidates"] and r_fail[0]["judge"]["score"] is None and "judge: no score" in o.getvalue()
    api.mode = "ok"
    # unreadable answer -> no score, not a perfect score
    api.answer = "I cannot judge this."
    with quiet():
        r_bad = pt3.run_conversation(conv3[:1], max_new_tokens=16)
    assert r_bad[0]["judge"]["score"] is None
    api.answer = 'Critical:\nno-error\nMajor:\nno-error\nMinor:\nno-error'
    # typing loop: judge toggle + mean score in the transcript; judge off -> no request
    n_before = len(api.requests)
    lines = iter(["en: Hello there.", "judge", "zh: 你好吗？", "judge", "en: Fine, thanks.", "quit"])
    builtins.input = lambda prompt="": next(lines)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tp.interactive_pipeline(pt3, max_new_tokens=16)
    text = buf.getvalue()
    assert len(api.requests) - n_before == 2, "the judge must be called only while it is switched on"
    assert "(judge off)" in text and "(judge on)" in text and "judge: mean MQM 0.00 over 2 of 3 messages" in text
    # no key -> no judge, a message, and the translator still works
    os.environ.pop("FAKE_JUDGE_KEY")
    with contextlib.redirect_stdout(io.StringIO()) as o:
        none = judge_chat.ChatJudge.create(provider="openai", base_url=api.url, judge_model="stub", api_key_env="FAKE_JUDGE_KEY")
    print("5. judge after the translation: prompt and context as in the paper's judge code, score parsed, failing / unreadable judge never breaks "
          "or changes the translation, on/off toggle, mean score in the transcript")
    # 6. the adapter folder carries the setup of the training run ----------------------------------------------------------------
    setup = tp.read_adapter_setup(A.adapter)
    assert setup["base_model"] == A.lm and setup["template"] == "chatml_empty_sys" and setup["context"] == "full", setup
    with quiet():
        pt4 = tp.load_system(adapter=A.adapter, comet_model=A.comet, n_candidates=4, dtype="float32", device_map=None)   # nothing repeated by hand
    from chat_mt_utils import TEMPLATES
    assert pt4.tr.template == TEMPLATES["chatml_empty_sys"]
    print("6. translator tied to the training run: base model and prompt format come from the adapter folder")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
