"""Ask the writer model how to improve the prompt, given its own explanation for a record."""
import asyncio, json, os, sys
from openai import AsyncOpenAI
from delta_nla import prompts_v2

QUESTION = sys.argv[3] if len(sys.argv) > 3 else "hi! how could we improve this prompt?"
ids = sys.argv[1].split(","); desc_path = sys.argv[2]
key = open(os.path.expanduser("~/.openrouter_key")).read().strip()
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
evs = {}
for l in open("data/raw/evidence.jsonl"):
    e = json.loads(l)
    if e["id"] in ids: evs[e["id"]] = e
descs = {json.loads(l)["id"]: json.loads(l) for l in open(desc_path)}

async def one(i):
    msgs = prompts_v2.build_messages(evs[i]) + [
        {"role": "assistant", "content": "<explanation>" + descs[i]["description"] + "</explanation>"},
        {"role": "user", "content": QUESTION}]
    r = await client.chat.completions.create(model="openai/gpt-5.6-luna", messages=msgs, max_tokens=3000,
                                             extra_body={"reasoning": {"effort": "medium"}})
    return i, r.choices[0].message.content

async def main():
    for i, txt in await asyncio.gather(*[one(i) for i in ids]):
        print("=" * 100); print(i, "| ...", repr(evs[i]["context"][-70:])); print(txt)
asyncio.run(main())
