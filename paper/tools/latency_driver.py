"""Drive live turns over the UI WebSocket to collect latency records headlessly.

Mutes the mic (so ambient noise doesn't trigger turns), then for each engine
mode sends a batch of short text prompts, waiting for each turn to complete
(signalled by the "Atlas:" response log). The running app's instrumentation
appends one JSONL line per turn to eval_work/live_latency.jsonl.

Run (app must be running):  python tools/latency_driver.py
Then aggregate:             python tools/latency_live.py
"""
import asyncio
import json

import websockets

WS = "ws://localhost:7862"

PROMPTS = [
    "Reply with one short friendly sentence.",
    "Say a quick hello in one sentence.",
    "Tell me a very short fun fact in one sentence.",
    "Give a one sentence motivational quote.",
    "Name your favourite colour in one short sentence.",
    "Say goodbye in one short sentence.",
    "In one sentence, what is the weather like on the moon?",
    "Give one short sentence of advice.",
]


async def set_mode(ws, sys1, sys2):
    await ws.send(json.dumps({"type": "toggle", "system": "sys1", "value": sys1}))
    await asyncio.sleep(0.2)
    await ws.send(json.dumps({"type": "toggle", "system": "sys2", "value": sys2}))
    await asyncio.sleep(0.4)


async def drain_until_atlas(ws, timeout=45):
    """Wait until an 'Atlas:' response log arrives (turn complete)."""
    try:
        async with asyncio.timeout(timeout):
            while True:
                raw = await ws.recv()
                m = json.loads(raw)
                if m.get("type") == "log" and str(m.get("text", "")).startswith("Atlas:"):
                    return True
    except (asyncio.TimeoutError, Exception):
        return False


async def run_batch(ws, label, sys1, sys2, post_wait):
    print(f"\n=== {label} (sys1={sys1}, sys2={sys2}) ===")
    await set_mode(ws, sys1, sys2)
    for i, p in enumerate(PROMPTS):
        await ws.send(json.dumps({"type": "text", "text": p}))
        ok = await drain_until_atlas(ws)
        print(f"  [{i+1}/{len(PROMPTS)}] {'ok' if ok else 'TIMEOUT'}: {p[:40]}")
        # let post-turn work (HuBERT inference + JSONL write) finish
        await asyncio.sleep(post_wait)


async def main():
    async with websockets.connect(WS, max_size=None) as ws:
        # Mute mic so only our text prompts drive turns
        await ws.send(json.dumps({"type": "mute"}))
        await asyncio.sleep(0.5)
        # SYS1-only: live audio + real-time visemes (fast post-turn)
        await run_batch(ws, "SYS1 LIVE VISEME", True, False, post_wait=1.5)
        # SYS2: HuBERT — needs seconds of CPU inference after each turn
        await run_batch(ws, "SYS2 HuBERT", False, True, post_wait=14.0)
        print("\nDone. Aggregate with: python tools/latency_live.py")


if __name__ == "__main__":
    asyncio.run(main())
