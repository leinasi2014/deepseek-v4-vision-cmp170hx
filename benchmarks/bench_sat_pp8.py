"""PP8 saturation with configurable concurrency: python3 bench_sat_pp8.py [conc] [gen] [model]"""
import json
import random
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:9016"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "DeepSeek-V4-Flash-Vision-Exp"
KEY = "sk_344303"
WORDS = ("system model tensor compute memory network pipeline attention cache token "
         "context prompt inference performance kernel matrix vector layer expert route "
         "sequence parallel stage communication bandwidth latency throughput benchmark").split()


def post(payload, timeout=2400):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return urllib.request.urlopen(req, timeout=timeout)


out = []
lock = threading.Lock()


def one(i, gen):
    rng = random.Random(7000 + i * 31)
    p = "Summarize:\n" + " ".join(rng.choice(WORDS) for _ in range(1800)) + "\nAnswer:"
    t0 = time.time()
    try:
        with post({"model": MODEL, "prompt": p, "max_tokens": gen,
                   "temperature": 1.0, "ignore_eos": True}) as r:
            d = json.load(r)
        with lock:
            out.append((d["usage"]["completion_tokens"], time.time() - t0))
    except Exception as e:
        with lock:
            out.append((0, time.time() - t0, str(e)[:80]))


def main():
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    threads = [threading.Thread(target=one, args=(i, gen)) for i in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
        time.sleep(0.08)
    for t in threads:
        t.join()
    wall = time.time() - t0
    ct = sum(o[0] for o in out)
    errs = [o for o in out if len(o) > 2]
    print(f"SAT(conc={conc},gen={gen}): gen_total={ct} wall={wall:.0f}s "
          f"AGG={ct/wall:.1f} t/s per_req={ct/conc/max(wall-2,0.1):.1f} errs={len(errs)}",
          flush=True)


if __name__ == "__main__":
    main()
