
# hf-memory-agent

An async conversational agent powered by Hugging Face models.  
It integrates tool execution, context compaction, and long‑term memory persistence.  
Conversations are logged, curated into memory files, and enriched with retrieval for continuity, enabling adaptive, tool‑aware dialogue.

---

## ✨ Features
- Hugging Face Inference API integration
- Tool execution (`get_time`, `http_get`)
- Long‑term memory extraction and storage
- Context compaction for efficient prompts
- Async agent loop with conversation logging
- Structured working directory for memory and dialog files

---

## 📂 Project Structure
- `agent_qwen_reme.py` → main agent loop
- `requirements.txt` → dependencies
- `working_dir/`  
  - `dialog/` → conversation logs in JSONL format  
  - `memory/` → extracted memory files in Markdown  
- `logs/` → runtime logs with timestamps  
- `.env.example` → template for environment variables (safe to commit)  
- `.env` → **local only**, contains your Hugging Face token (ignored by Git)

---

