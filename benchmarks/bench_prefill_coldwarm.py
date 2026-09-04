"""Real-usage prefill: COLD (fresh prompt) vs WARM (prefix-cache hit).
Warm covers the two real patterns: full re-send (100% hit) and agent-turn
growing prefix (~99% hit). Reports wall time, cached_tokens, effective t/s."""
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:9016"
MODEL = "DeepSeek-V4-Flash-Vision-Exp"
KEY = "sk_344303"
WORDS = ("system model tensor compute memory network pipeline attention cache token "
         "context prompt inference performance kernel matrix vector layer expert route "
         "sequence parallel stage communication bandwidth latency throughput benchmark").split()


def post(path, payload, timeout=1200):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return urllib.request.urlopen(req, timeout=timeout)


def tok_count(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}, timeout=300) as r:
        return json.load(r)["count"]


def build(seed, target):
    rng = random.Random(seed)
    lo, hi = target - 2000, target + 1000
    while lo <= hi:
        n = (lo + hi) // 2
        rng.seed(seed)
        text = "Summarize:\n" + " ".join(rng.choice(WORDS) for _ in range(n)) + "\nAnswer:"
        c = tok_count(text)
        if c == target:
            return text
        if c < target:
            lo = n + 1
        else:
            hi = n - 1
    raise RuntimeError("build failed")


def send(p, max_tokens=1):
    t0 = time.time()
    with post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": max_tokens,
                                  "temperature": 0.0, "ignore_eos": True}) as r:
        d = json.load(r)
    dt = time.time() - t0
    u = d["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return u["prompt_tokens"], cached, dt


def main():
    print("== COLD 30k (fresh) x2 ==", flush=True)
    p = build(40001, 30000)
    for i in range(2):
        pt, cached, dt = send(build(41000 + i, 30000))
        print(f"  {pt} tok, cached={cached}: {dt:.2f}s = {pt/dt:.0f} t/s", flush=True)

    print("== WARM full-hit 30k (same prompt resent) x3 ==", flush=True)
    for i in range(3):
        pt, cached, dt = send(p)
        print(f"  {pt} tok, cached={cached}: {dt:.2f}s = {pt/dt:.0f} t/s eff", flush=True)

    print("== WARM agent-turn (30k base + ~500 tok/turn, 5 turns) ==", flush=True)
    rng = random.Random(777)
    stem = p[: p.rindex("\nAnswer:")]
    for t in range(1, 6):
        text = stem + "\nTurn %d: %s\nAnswer:" % (
            t, " ".join(rng.choice(WORDS) for _ in range(400)))
        pt, cached, dt = send(text)
        print(f"  turn{t}: {pt} tok, cached={cached} ({cached*100//pt}%): "
              f"{dt:.2f}s = {pt/dt:.0f} t/s eff", flush=True)

    print("== COLD 60k (fresh) ==", flush=True)
    pt, cached, dt = send(build(40002, 60000))
    print(f"  {pt} tok, cached={cached}: {dt:.2f}s = {pt/dt:.0f} t/s", flush=True)
    print("PREFILL_COLDWARM_DONE", flush=True)


if __name__ == "__main__":
    main()
