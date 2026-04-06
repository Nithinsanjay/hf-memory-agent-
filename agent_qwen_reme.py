import os
import json
import re
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

load_dotenv()

MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Put HF_TOKEN=... in your .env file.")

hf_client = InferenceClient(token=HF_TOKEN)

# ---------------- Storage (ReMe-style files) ----------------
WORKING_DIR = Path("working_dir")
DIALOG_DIR = WORKING_DIR / "dialog"
DAILY_DIR = WORKING_DIR / "memory"
MEMORY_FILE = WORKING_DIR / "MEMORY.md"
STATE_FILE = WORKING_DIR / "state.json"
TOOL_RESULT_DIR = WORKING_DIR / "tool_result"

MAX_TOOL_CHARS = 200000  # keep large; we store to file anyway

# Context compaction controls
MAX_PROMPT_CHARS = 20000     # compact when prompt exceeds this
KEEP_LAST_TURNS = 8          # keep last N user+assistant turns (~2N messages)

TOOL_PATTERN = re.compile(r"^TOOL:\s*(\w+)\s*(\{.*\})\s*$", re.DOTALL)

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def ensure_dirs():
    WORKING_DIR.mkdir(exist_ok=True)
    DIALOG_DIR.mkdir(exist_ok=True)
    DAILY_DIR.mkdir(exist_ok=True)
    TOOL_RESULT_DIR.mkdir(exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# Long-term Memory (curated)\n\n"
            "This file stores stable, important facts and preferences.\n\n",
            encoding="utf-8",
        )

def dialog_path() -> Path:
    return DIALOG_DIR / f"{today_str()}.jsonl"

def daily_path() -> Path:
    return DAILY_DIR / f"{today_str()}.md"

def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_md(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text)

def load_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

# ---------------- Tool output caching ----------------
def maybe_store_tool_output(tool_name: str, output: str, preview_chars: int = 2000) -> str:
    """
    Store large tool outputs in working_dir/tool_result and return a compact reference.
    """
    if output is None:
        return ""

    if len(output) <= preview_chars:
        return output

    TOOL_RESULT_DIR.mkdir(exist_ok=True)
    file_id = uuid.uuid4().hex
    path = TOOL_RESULT_DIR / f"{tool_name}_{file_id}.txt"
    path.write_text(output, encoding="utf-8", errors="ignore")

    preview = output[:preview_chars]
    return (
        f"[TOOL_OUTPUT_STORED]\n"
        f"tool={tool_name}\n"
        f"file={path.as_posix()}\n"
        f"preview(first_{preview_chars}_chars)=\n{preview}\n"
        f"...(truncated; full output saved in file)"
    )

# ---------------- Simple retrieval (keyword-based) ----------------
def simple_retrieve(query: str, k: int = 8) -> str:
    query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 3]
    if not query_terms:
        return ""

    sources: List[Tuple[str, str]] = []
    sources.append(("MEMORY.md", load_text_if_exists(MEMORY_FILE)))
    sources.append((f"memory/{today_str()}.md", load_text_if_exists(daily_path())))

    hits: List[Tuple[int, str, str]] = []
    for name, content in sources:
        for line in content.splitlines():
            l = line.strip()
            if not l:
                continue
            score = sum(1 for t in query_terms if t in l.lower())
            if score > 0:
                hits.append((score, name, l))

    hits.sort(key=lambda x: x[0], reverse=True)
    top = hits[:k]
    if not top:
        return ""

    out_lines = ["Relevant memory snippets:"]
    for score, name, line in top:
        out_lines.append(f"- ({name}) {line}")
    return "\n".join(out_lines)

# ---------------- Memory extraction (curated) ----------------
MEMORY_EXTRACTOR_SYSTEM = """You extract long-term memory from a conversation.

Return ONLY a JSON array. Each item must be an object with:
- "fact": a short, stable fact worth remembering long-term
- "category": one of ["preference","identity","project","constraint","decision","other"]

Rules:
- Only include stable, useful facts (not chit-chat).
- If nothing is worth saving, return [].
"""

def extract_memories(user_text: str, assistant_text: str) -> List[dict]:
    messages = [
        {"role": "system", "content": MEMORY_EXTRACTOR_SYSTEM},
        {"role": "user", "content": f"USER:\n{user_text}\n\nASSISTANT:\n{assistant_text}"},
    ]
    resp = hf_client.chat_completion(
        model=MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()

    m = re.search(r"\[\s*{.*}\s*\]|\[\s*\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            cleaned = []
            for item in data:
                if isinstance(item, dict) and "fact" in item:
                    cleaned.append(
                        {
                            "fact": str(item["fact"]).strip(),
                            "category": str(item.get("category", "other")).strip(),
                        }
                    )
            return cleaned
    except Exception:
        return []
    return []

def append_to_memory_md(items: List[dict]):
    if not items:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n## {ts}\n"]
    for it in items:
        fact = it["fact"]
        cat = it.get("category", "other")
        if fact:
            lines.append(f"- [{cat}] {fact}\n")
    append_md(MEMORY_FILE, "".join(lines))

# ---------------- Context compaction ----------------
COMPACTOR_SYSTEM = """You are a conversation compactor.

Given:
- an existing summary (may be empty)
- a chunk of older conversation messages

Produce an UPDATED summary that preserves:
- user identity & preferences
- ongoing goals/projects
- decisions made
- important constraints
- important tool outcomes only if necessary

Keep it concise and factual. Bullet points are OK.
"""

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def get_summary_from_state() -> str:
    state = load_state()
    return (state.get("conversation_summary") or "").strip()

def set_summary_in_state(summary: str):
    state = load_state()
    state["conversation_summary"] = summary.strip()
    state["updated_at"] = datetime.now().isoformat()
    save_state(state)

def approx_prompt_len(messages: List[Dict[str, str]]) -> int:
    return sum(len((m.get("content") or "")) for m in messages)

def summarize_messages(existing_summary: str, old_messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in old_messages:
        role = m["role"]
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            continue
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")

    chunk = "\n".join(lines[-120:])

    messages = [
        {"role": "system", "content": COMPACTOR_SYSTEM},
        {"role": "user", "content": f"EXISTING_SUMMARY:\n{existing_summary}\n\nOLDER_CONVERSATION:\n{chunk}"},
    ]
    resp = hf_client.chat_completion(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def compact_if_needed(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if approx_prompt_len(messages) <= MAX_PROMPT_CHARS:
        return messages

    existing_summary = get_summary_from_state()

    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    keep_n = 2 * KEEP_LAST_TURNS
    to_keep = non_system[-keep_n:] if len(non_system) > keep_n else non_system
    to_summarize = non_system[:-keep_n] if len(non_system) > keep_n else []

    new_summary = existing_summary
    if to_summarize:
        new_summary = summarize_messages(existing_summary, to_summarize)
        set_summary_in_state(new_summary)

    summary_msg = {
        "role": "system",
        "content": "Conversation summary (auto-generated, keep consistent):\n" + (new_summary or "(empty)")
    }

    base_system = system_msgs[:1] if system_msgs else []
    return base_system + [summary_msg] + to_keep

# ---------------- Agent tools + chat ----------------
SYSTEM_PROMPT = """You are a tool-using agent.

When you need a tool, respond ONLY in exactly this format:
TOOL: tool_name {"arg":"value"}

Available tools:
1) get_time {}
2) http_get {"url":"https://example.com"}

If no tool is needed, answer normally.
"""

def get_time(_: dict) -> str:
    return datetime.now().isoformat()

def http_get(args: dict) -> str:
    url = args["url"]
    r = requests.get(url, timeout=30)
    text = r.text
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + "\n...TRUNCATED..."
    return f"Status: {r.status_code}\n\n{text}"

async def run_tool(name: str, args: dict) -> str:
    if name == "get_time":
        return get_time(args)
    if name == "http_get":
        return http_get(args)
    return f"Unknown tool: {name}"

def hf_chat(messages: List[Dict[str, str]]) -> str:
    resp = hf_client.chat_completion(
        model=MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------------- Main agent loop with memory + compaction + tool caching ----------------
async def agent_turn(messages: List[Dict[str, str]], user_text: str) -> Tuple[List[Dict[str, str]], str]:
    mem = simple_retrieve(user_text)
    if mem:
        messages = messages + [{"role": "system", "content": mem}]

    messages = compact_if_needed(messages)

    assistant = hf_chat(messages)
    messages.append({"role": "assistant", "content": assistant})

    for _ in range(2):
        m = TOOL_PATTERN.match(assistant.strip())
        if not m:
            break

        tool_name = m.group(1)
        tool_args = json.loads(m.group(2))

        tool_out_full = await run_tool(tool_name, tool_args)
        tool_out = maybe_store_tool_output(tool_name, tool_out_full, preview_chars=2000)

        messages.append({"role": "user", "content": f"TOOL_RESULT: {tool_name}\n{tool_out}"})

        messages = compact_if_needed(messages)

        assistant = hf_chat(messages)
        messages.append({"role": "assistant", "content": assistant})

    return messages, assistant

async def main():
    ensure_dirs()

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        append_jsonl(dialog_path(), {"ts": ts, "role": "user", "content": user_text})
        append_md(daily_path(), f"\n### {ts}\n**User:** {user_text}\n")

        messages.append({"role": "user", "content": user_text})

        messages, assistant_text = await agent_turn(messages, user_text)
        print("\nAssistant:", assistant_text)

        append_jsonl(dialog_path(), {"ts": ts, "role": "assistant", "content": assistant_text})
        append_md(daily_path(), f"**Assistant:** {assistant_text}\n")

        items = extract_memories(user_text, assistant_text)
        append_to_memory_md(items)

if __name__ == "__main__":
    asyncio.run(main())
