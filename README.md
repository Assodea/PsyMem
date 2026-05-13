```markdown
# PsyMem: Longitudinal Psychotherapy Dialogue Synthesis Framework

**Paper**: *[arXiv link will be added after publication]*  
**Code & Data**: *[Repository URL will be made public upon acceptance]*

> **PsyMem** is a framework for synthesizing **longitudinal (multi-session) counseling dialogues** via memory-augmented role-playing.  
> It models therapy as a state-evolution process across sessions, combining:
> - Psychologically grounded persona construction  
> - Reflection-guided utterance-level quality control  
> - **Role-asymmetric bidirectional memory** (client problem state vs. counselor disclosure-based memory)

This repository contains the core synthesis pipeline: dual-agent (client & counselor) interaction, session-level memory extraction, and automatic continuity evaluation (MCAF).

---

## 📌 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Environment & Dependencies](#environment--dependencies)
- **Important: Network Requirements** ⚠️
- [Configuration (You Must Edit)](#configuration-you-must-edit)
- [Running the Synthesis Pipeline](#running-the-synthesis-pipeline)
- [Output Artifacts](#output-artifacts)
- [Evaluation & Fine-tuning](#evaluation--fine-tuning)
- [Citation](#citation)
- [License & Safeguards](#license--safeguards)

---

## Overview

Real psychotherapy dialogues are privacy-sensitive and costly to collect. Existing synthetic corpora are mostly **single-session** and **session-isolated**, lacking cross-session therapeutic continuity.

**PsyMem** addresses this by generating multi-session client-counselor continuums (e.g., 4 sessions per client, 60+ turns per session). Key innovations:

| Component | Description |
|-----------|-------------|
| **Persona Grounding** | Expands sparse client profiles into structured hidden persona (stable across sessions). |
| **Reflection-Guided Generation** | Client utterances are verified for completeness, concrete detail, and persona consistency; auto-rework if needed. |
| **Role-Asymmetric Memory** | Client maintains evolving problem state (using hidden persona + past suggestions). Counselor maintains memory only from disclosed dialogue (no access to hidden profile). |
| **MCAF** | Memory Continuity Analysis Framework to score cross-session coherence (4 dimensions, 1-5 scale). |

The pipeline generates **150 clients × 4 sessions = 600 dialogues** (~240 turns per continuum).

---

## Quick Start

1. **Clone & install dependencies**
   ```bash
   git clone <your-repo-url>
   cd PsyMem
   pip install -r requirements.txt
   ```

2. **Set up Azure OpenAI credentials** – see [Configuration](#configuration-you-must-edit)

3. **Place client profile files** – modify the file path inside `PsyMem.py` to point to your own `.txt` files (one per client)

4. **Start the two agent APIs** (in separate terminals)
   ```bash
   python client_agent.py
   python counselor_agent.py
   ```

5. **Run the main synthesis script**
   ```bash
   python PsyMem.py
   ```

Generated dialogues, feedback logs, and memory summaries will be saved in `./longdialogue/`.

---

## Environment & Dependencies

- Python 3.9+
- Azure OpenAI API access (**GPT-4o** – we use `gpt-4o` model; you can change it in the agent scripts if needed)
- Required packages:

```
fastapi
uvicorn
openai
requests
ujson
```

> The code uses `AzureOpenAI` client. If you use standard OpenAI endpoint, modify `client_agent.py` and `counselor_agent.py` accordingly.

---

## ⚠️ Important: Network Requirements

The synthesis pipeline makes many sequential API calls to Azure OpenAI (often 100+ turns per session, multiple retries). **A stable and fast network connection is critical.**

If your network is slow or unstable, you will see errors like:

```
第1次尝试：请求失败: HTTPConnectionPool(host='127.0.0.1', port=8112): Read timed out. (read timeout=120)
第2次尝试：请求失败: HTTPConnectionPool(host='127.0.0.1', port=8112): Read timed out. (read timeout=120)
咨询师回应基础无效，内容: 网络错误，请稍后重试，重新请求...
```

**Recommendations:**
- Use a reliable network (wired Ethernet preferred over Wi-Fi)
- If behind a proxy, configure `HTTP_PROXY` / `HTTPS_PROXY` appropriately
- Increase timeout values in `safe_api_request()` (default 120s) if your API responds slowly
- Run the agent servers on the same machine (localhost) to avoid inter-process network latency

Poor network conditions will lead to frequent retries, slower generation, and may even break the session loop.

---

## Configuration (You Must Edit)

Before running, you **must** modify the following paths and credentials in the scripts.

### 1. Azure OpenAI Credentials

Set environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

Or replace the `os.environ.get(...)` calls in `client_agent.py` and `counselor_agent.py` with hardcoded strings (not recommended for security).

### 2. Client Profile TXT Files

In `PsyMem.py`, locate this line (around line 400):

```python
txt_file_path = rf'D:\VScodeProject\inter\interactive-agents-main\oursRoleCard\TXT{idx}.txt'
```

**Change it** to the directory where your client profile text files are stored. Each file should be named `TXT0.txt`, `TXT1.txt`, etc. (one per client). The content must contain fields like `职业：`, `婚恋状态：`, `阻抗程度：`, and `当前主要心理困扰：`.

### 3. Model Selection (Optional)

The code uses **GPT-4o** by default. If you want to use a different model (e.g., GPT-4, GPT-3.5), modify the `openai_model` variable in both `client_agent.py` and `counselor_agent.py`:

```python
openai_model = 'gpt-4o'   # change to your desired model name
```

### 4. Ports and Hosts

Agent servers run on `127.0.0.1:8111` (client) and `8112` (counselor). If these ports are occupied, change them inside each script's `uvicorn.run()` call.

---

## Running the Synthesis Pipeline

### Step 1: Start the two agent servers

```bash
# Terminal 1
python client_agent.py

# Terminal 2
python counselor_agent.py
```

Or run in background:

```bash
nohup python -u client_agent.py > ./client_agent.log &
nohup python -u counselor_agent.py > ./counselor_agent.log &
```

### Step 2: Run the multi-session generator

```bash
python PsyMem.py
```

This will:
- For each client index in `target_indices` (default `[0]`):
  - Build initial persona from `TXT{idx}.txt`
  - Generate `visit_count` sessions (default 4)
  - Apply reflection-guided rework for client utterances
  - After each session, extract dual-perspective memory
  - Use previous memory to initialize next session

**Example output:**
```
预热来访者 API...
预热咨询师 API...
开始生成第 0 位来访者的第 1 次访问对话...
第 0 位来访者第 1 次访问第 1 轮...
第 0 位来访者第 1 次访问第 1 轮检查通过
...
```

---

## Output Artifacts

All saved in `saved_dir` (default `./longdialogue/`):

| File pattern | Content |
|--------------|---------|
| `{idx}_{visit}.json` | Full dialogue session (list of `{"role": "client/counselor", "content": "..."}`) |
| `re{idx}_{visit}.json` | Reflection feedback logs per turn (includes `need_rework`, `reason`, etc.) |
| `memory_{idx}_{visit}.json` | Dual-perspective memory summary (client state & counselor memory) |

**Memory file structure example:**
```json
{
  "client_memory": { "facts": [...], "suggestions": [...], "personality": "...", "issues": "..." },
  "counselor_memory": { "facts": [...], "suggestions": [...], "personality": "...", "progress": "..." },
  "visit": 2,
  "timestamp": "2025-05-13 10:30:00"
}
```

---

## Evaluation & Fine-tuning

### Memory Continuity Analysis (MCAF)

We provide an automatic evaluation script (not included in this core release) that scores a 4-session continuum on:
- Client-side state continuity
- Counselor-side memory continuity  
- Within-session alignment
- Overall multi-session consistency

Each dimension is scored 1–5. A score ≥4.5 indicates strong cross-session coherence.  
See the paper's supplementary material for evaluation code and prompts.

### Fine-tuning downstream models

You can use the generated dialogues (`{idx}_{visit}.json`) to fine-tune any LLM (e.g., LLaMA2, Qwen2, DeepSeek-7B).  
The paper fine-tuned with LoRA (rank 8, alpha 16) on 100k samples, achieving improved WAIOs and MentalChat16K scores.

---

## Citation

If you find PsyMem useful for your research, please cite our paper (BibTeX will be updated after publication):

```bibtex
@article{psymem2025,
  title={PsyMem: Longitudinal Psychotherapy Dialogue Synthesis with Role-Asymmetric Memory},
  author={[Author Names]},
  journal={[Conference/Journal]},
  year={2025},
  url={[arXiv link]}
}
```

---

## License & Safeguards

This code and data are released for **research purposes only**.  
They are **not** intended for clinical use, crisis intervention, or as a substitute for professional mental health care.

By using this repository, you agree to:
- Not deploy generated dialogues as real therapy
- Include clear AI disclosure if building on this work
- Implement human oversight and safety checks for any downstream application

See Section 5 of the paper for broader impact and limitation discussions.

---

## Repository Structure

```
PsyMem/
├── client_agent.py       # FastAPI server for client agent (GPT-4o)
├── counselor_agent.py    # FastAPI server for counselor agent (GPT-4o)
├── PsyMem.py             # Main synthesis loop (session orchestration, memory, reflection)
├── longdialogue/         # Output directory (created at runtime)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Troubleshooting

| Issue | Likely cause & solution |
|-------|--------------------------|
| `openai.error.AuthenticationError` | Wrong Azure key/endpoint; check environment variables. |
| Connection refused on 8111/8112 | Agent servers not running; start them first. |
| `Read timed out` errors (as shown above) | **Poor network** – see [Network Requirements](#important-network-requirements). |
| `FileNotFoundError` for `TXT{idx}.txt` | You forgot to change the file path in `PsyMem.py`; edit it to your own directory. |
| Repeated rework loops (infinite retries) | Client utterances failing quality checks; lower `MAX_RETRY_TIMES` or relax `check_client_response()` heuristics. |
| JSON decode error from agent API | The `safe_api_request()` will retry; check agent logs for malformed responses. Often caused by network timeouts. |

---

## Contact

For questions, please open an issue on this repository or email the corresponding author.

**Enjoy longitudinal dialogue synthesis!** 🧠💬
```



这样读者拿到 README 后就能清楚知道哪些地方需要自己配置，以及网络问题如何应对。
