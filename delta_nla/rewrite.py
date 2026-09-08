"""Grounded LLM rewrite: evidence.jsonl -> descriptions.jsonl via OpenRouter.

Resumable (skips ids already in the output file), concurrent, tracks token usage and cost.
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, time
from pathlib import Path

from openai import AsyncOpenAI

from .prompts import build_messages


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path) if l.strip()}


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j < 0:
            return None
        try:
            obj = json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or "description" not in obj:
        return None
    return obj


async def one(client, sem, model, ev, out_f, stats, max_tokens, temperature, lock, reasoning):
    msgs = build_messages(ev)
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens,
                                                         temperature=temperature, response_format={"type": "json_object"},
                                                         extra_body={"reasoning": {"effort": reasoning}})
                content = r.choices[0].message.content or ""
                obj = parse_json(content)
                if obj is not None and not obj.get("description", "").strip():
                    obj = None
                u = r.usage
                cost = 0.0
                if u is not None:
                    stats["prompt_tokens"] += u.prompt_tokens or 0
                    stats["completion_tokens"] += u.completion_tokens or 0
                    extra = getattr(u, "model_extra", None) or {}
                    cost = float(extra.get("cost") or 0) or float((extra.get("cost_details") or {}).get("upstream_inference_cost") or 0)
                    stats["cost"] += cost
                if obj is None:
                    stats["parse_fail"] += 1
                    if attempt < 3:
                        continue
                    stats["failed"] += 1
                    return
                rec = {"id": ev["id"], "layer": ev["layer"], "model": model, "description": obj.get("description", "").strip(),
                       "short": (obj.get("short") or "").strip(), "cost": cost}
                async with lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
                    stats["done"] += 1
                return
            except Exception as e:  # rate limits, transient errors
                stats["errors"] += 1
                await asyncio.sleep(min(30, 2 ** attempt + random.random()))
        stats["failed"] += 1


async def run(args):
    key = os.environ.get("OPENROUTER_API_KEY") or open(os.path.expanduser("~/.openrouter_key")).read().strip()
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         default_headers={"HTTP-Referer": "https://github.com/syvb/metamodelling", "X-Title": "delta-nla-warmstart"})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    evs = [json.loads(l) for l in open(args.evidence)]
    if args.ids:
        want = set(args.ids.split(",")); evs = [e for e in evs if e["id"] in want]
    evs = [e for e in evs if e["id"] not in done]
    if args.limit:
        evs = evs[: args.limit]
    print(f"{len(evs)} to do ({len(done)} already done)", flush=True)
    stats = {"done": 0, "errors": 0, "failed": 0, "parse_fail": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    sem = asyncio.Semaphore(args.concurrency); lock = asyncio.Lock()
    t0 = time.time()
    with open(out, "a") as out_f:
        tasks = [one(client, sem, args.model, ev, out_f, stats, args.max_tokens, args.temperature, lock, args.reasoning) for ev in evs]
        async def progress():
            while True:
                await asyncio.sleep(15)
                print(f"  {stats['done']}/{len(evs)} done, cost=${stats['cost']:.3f}, tok={stats['prompt_tokens']}+{stats['completion_tokens']}, errors={stats['errors']}, {time.time()-t0:.0f}s", flush=True)
        p = asyncio.create_task(progress())
        await asyncio.gather(*tasks)
        p.cancel()
    print(json.dumps(stats), flush=True)
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, job_type="rewrite", config=vars(args))
        wb.log(stats); wb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="data/evidence.jsonl")
    ap.add_argument("--out", default="data/descriptions.jsonl")
    ap.add_argument("--model", default="openai/gpt-5.6-luna")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--reasoning", default="low", help="OpenRouter reasoning effort: minimal/low/medium/high")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--wandb", default="")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
