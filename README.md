# WealthWise AI — AI-Powered Personal Finance Advisor

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-LangGraph-1C3C3C.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991.svg)
![Gen AI](https://img.shields.io/badge/Gen%20AI-5%20Capabilities-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Upload your real bank statement → get an instant financial dashboard and chat with an AI advisor grounded in your actual spending.**

> *From a notebook nobody could use → to a product you open in a browser.*

---



## What it does

- 📥 **Upload a real bank statement** (CSV or Excel) — it auto-detects the format, skips the
  account-info preamble, handles separate Debit/Credit columns, day-first dates, and
  comma-formatted amounts. Tested against **State Bank of India (SBI)** statements.
- 🏷️ **Automatic categorization** — extracts the merchant/payee from messy UPI strings
  (`WDL TFR UPI/DR/.../Blinkit/...` → **Blinkit**) and categorizes it. Common Indian merchants
  match a **free keyword map**; only unknowns hit the LLM (capped per upload to bound cost).
- 💱 **Currency-aware** — reads the statement's currency (e.g. `INR`) and renders the whole
  dashboard in **₹** with Indian digit grouping.
- 📊 **Live dashboard** — KPIs, cash-flow trend, savings rate, 50/30/20 budget, spending by
  category, and an interactive investment-growth calculator.
- 💬 **AI advisor** — an always-visible chat rail grounded in *your* numbers. Ask
  *"where's my money going?"*, *"what's my biggest expense?"*, *"where can I cut back?"* and it
  answers from your real data, plus does semantic search over your transactions.
- 🔒 **Private & isolated by design** — see [Security & Privacy](#security--privacy).

---

## Architecture

```mermaid
flowchart LR
    subgraph Client["Streamlit Dashboard"]
        UI["Dashboard<br/>KPIs, charts, categories"]
        Chat["AI Advisor<br/>interactive chat"]
    end

    subgraph Backend["Backend Services (src/backend/)"]
        Session["Session Management<br/>isolated per-user"]
        Ingest["Statement ingest<br/>CSV/XLSX, debit/credit"]
        Redact["PII redaction<br/>account and ref numbers stripped"]
        Snapshot["Dashboard snapshot<br/>KPIs, budget, cash flow"]
    end

    subgraph Brain["LangGraph ReAct Agent"]
        T1["Calculators"]
        T2["RAG lookup"]
        T3["Categorize"]
        T4["Semantic search"]
    end

    OpenAI["OpenAI<br/>gpt-4o-mini + embeddings"]
    Chroma[("Chroma<br/>knowledge base")]

    UI -->|upload| Ingest
    Ingest --> Snapshot
    Snapshot --> UI
    Chat -->|chat| Session
    Session --> Brain
    Brain --> T1
    Brain --> T2
    Brain --> T3
    Brain --> T4
    Brain -->|grounded and redacted| Redact
    Redact --> OpenAI
    Ingest -.->|merchant redacted| OpenAI
    T2 --> Chroma
    T4 -->|embed search| OpenAI
```

**Flow:** The Streamlit dashboard interacts with the backend services to process uploaded statements and generate the financial dashboard. Uploads are normalized and categorized, then a dashboard snapshot drives the visualizations. The chat grounds a LangGraph agent in the user’s financial data and routes sensitive information through a redaction layer before it reaches OpenAI.

---

## The 5 Gen AI Capabilities

All five are wired into the LangGraph agent as callable tools, so the conversational advisor
can reach any of them on demand.

| # | Capability | Implementation | Agent tool |
|---|------------|----------------|------------|
| 1 | **Structured Output** | Pydantic `with_structured_output` → typed `TransactionCategory` | `categorize_transaction` |
| 2 | **RAG** | Chroma vector store over a financial knowledge base | `financial_knowledge_lookup` |
| 3 | **Embeddings** | OpenAI embeddings + cosine similarity over your statement | `find_similar_transactions` |
| 4 | **Function Calling** | 5 financial calculators (budget · emergency fund · debt · investment · loan) | `calculate_*` |
| 5 | **Agents (LangGraph)** | `create_react_agent` + memory, orchestrating all of the above | *(the advisor itself)* |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM** | OpenAI `gpt-4o-mini` (configurable) |
| **Agent framework** | LangChain + **LangGraph** (ReAct agent with tool calling + memory) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector store** | ChromaDB (`langchain-chroma`) |
| **Backend** | FastAPI + Uvicorn |
| **Data** | pandas · NumPy · scikit-learn · openpyxl |
| **Frontend** | Streamlit |

---

## Quick Start

**Prerequisites:** Python 3.10+ and an OpenAI API key.

```bash
# 1. Clone
git clone https://github.com/avijain1412/WealthWise-AI.git
cd WealthWise-AI

# 2. Install
pip install -r requirements.txt

# 3. Add your key
cp .env.example .env        # then edit .env and set OPENAI_API_KEY

# 4. Run the dashboard
streamlit run src/streamlit_ui/app.py
# open http://localhost:8000
```

Upload a bank statement through the Streamlit dashboard, and the application will process the data, generate your financial dashboard, and update the AI advisor with your financial information. If no statement is uploaded, the application runs using a built-in sample dataset.

For a terminal-based experience, you can also run:

```bash
python main.py
```

> The first run builds a local Chroma vector store from the knowledge base (cached in `.wealthwise_chroma/`).

---

## Bank Statement Support

The importer is built for **real, messy statements**, not just clean CSVs:

- **Formats:** `.csv`, `.xlsx`, `.xls`
- **Preamble handling:** automatically finds the real transaction table below account-info rows
- **Column aliases:** `Date`/`Transaction Date`, `Narration`/`Particulars`/`Details`,
  `Withdrawal Amt.`/`Deposit Amt.`, `Debit`/`Credit`, `Amount`, …
- **Debit/Credit split** → single signed amount (debit = spend, credit = income)
- **Day-first dates** (`01/03/2026` → 1 March) and **comma amounts** (`1,250.00`)
- **Encoding fallback** (UTF-8 → cp1252 for Windows-exported bank CSVs)
- **UPI/IMPS merchant extraction** + Indian merchant keyword categorization

Minimum required columns: a **date**, a **description**, and either an **amount** or
**debit/credit** columns.

---

## Security & Privacy

This is a public-demo-ready app, so isolation and privacy were designed in:

- 🔑 **Server-issued sessions** — a cryptographically strong (256-bit) **HttpOnly** cookie.
  The server never trusts a client-supplied id, so no one can guess or set another user's
  session. Each visitor's dashboard, chat, and search are fully isolated.
- 🔒 **PII redaction before the LLM** — account numbers, customer/reference IDs, and UPI VPAs
  are stripped from everything sent to OpenAI (chat grounding, categorization, and the
  semantic-search corpus). Merchant names and amounts are kept so advice stays useful; the
  account header is discarded on import and never leaves your machine.
- 🧾 **In-memory only** — uploaded statements live in server RAM keyed to your session, never
  written to disk or a database.
- ⏲️ **Auto-expiry** — a statement is purged after **30 min of inactivity** (configurable), so
  sensitive data doesn't linger and idle sessions don't leak memory.
- 🚦 **Rate limiting + daily budget cap** — per-session limits stop spam, and a global daily
  "LLM-unit" cap bounds total OpenAI spend even across many sessions. All tunable via env vars.

---


### Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Your OpenAI key (required) |
| `WEALTHWISE_CHAT_PER_MIN` | `15` | Max chat messages per session per minute |
| `WEALTHWISE_UPLOAD_PER_MIN` | `5` | Max uploads per session per minute |
| `WEALTHWISE_DAILY_LIMIT` | `600` | Global daily LLM-unit cap (chat = 1, upload = 10) |
| `WEALTHWISE_SESSION_TTL_MIN` | `30` | Minutes of inactivity before an upload is auto-purged |
| `WEALTHWISE_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |

For a tight public demo, lower `WEALTHWISE_DAILY_LIMIT` (e.g. `150`) and `WEALTHWISE_SESSION_TTL_MIN`
(e.g. `15`) — together with your OpenAI account spending cap, this keeps cost firmly bounded.

---

---

```markdown
## Testing

A lightweight test is included for statement ingestion and normalization.

Run:

```bash
python test_upload.py
```

## Project Structure

```
wealthwise-ai/
├── src/
│   ├── backend/
│   │   ├── wealthwise/                 # Core domain logic
│   │   │   ├── config.py               # Model config + LLM factories
│   │   │   ├── privacy.py              # PII redaction layer
│   │   │   ├── data/                   # Sample transactions + knowledge base
│   │   │   ├── analysis/               # Categorization & embeddings
│   │   │   ├── rag/                    # Chroma knowledge base
│   │   │   ├── tools/                  # Agent tools & calculators
│   │   │   ├── dashboard.py            # KPIs & cash flow snapshot
│   │   │   └── agent/                  # LangGraph conversational advisor
│   │   └── server.py                   # FastAPI server (API & sessions)
│   │ 
│   └── streamlit_ui/
│       └── app.py                      # Full-featured Streamlit dashboard
├── main.py                             # CLI chat entry point
├── test_upload.py                      # Statement parser tests
├── render.yaml                         # One-click deploy config
├── pyproject.toml                      # Project metadata
├── requirements.txt
├── .env.example
└── fingenius-notebook-gemini-agent.ipynb # Original Colab notebook
```

---



---


