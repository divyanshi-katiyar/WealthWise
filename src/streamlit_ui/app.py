"""WealthWise AI Streamlit web app.

A polished browser chat UI for the financial advisor agent, with a full
dashboard showing KPIs, budgets, and charts.

    streamlit run src/streamlit_ui/app.py
"""

import io
import uuid

import pandas as pd
import streamlit as st

from src.backend.wealthwise.agent import chat
from src.backend.wealthwise.data import generate_transactions
from src.backend.wealthwise.dashboard import dashboard_snapshot, snapshot_summary
from src.backend.wealthwise.ingest import normalize, strip_preamble, detect_currency, CURRENCY_SYMBOLS
from src.backend.wealthwise.tools.calculators import calculate_investment_growth
from src.backend.wealthwise.tools.transaction_tools import set_active_transactions
from src.streamlit_ui.charts import spending_chart, cashflow_chart, investment_growth_chart

st.set_page_config(page_title="WealthWise AI", page_icon="💰", layout="wide")

# --- Theme / custom styling ---------------------------------------------
ACCENT = "#10b981"        # emerald
ACCENT_DARK = "#059669"
BG_CARD = "#161b22"
BORDER = "#262d3a"

st.markdown(
    f"""
    <style>
    /* Base */
    .stApp {{ background: #0d1117; }}
    #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1100px; }}

    /* Hero header */
    .fg-hero {{ margin-bottom: 1.4rem; }}
    .fg-title {{
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg, {ACCENT}, #38bdf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .fg-sub {{ color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: #0b0e14; border-right: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; color: #e6edf3; }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 14px; padding: 14px 16px;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 700; }}
    div[data-testid="stMetricLabel"] {{ color: #8b949e; }}

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 16px; padding: 6px 14px; margin-bottom: 8px;
    }}

    /* Example-question chips */
    div[data-testid="stButton"] > button {{
        background: {BG_CARD}; color: #c9d1d9;
        border: 1px solid {BORDER}; border-radius: 12px;
        padding: 12px 14px; text-align: left; font-size: 0.9rem;
        transition: all 0.15s ease; width: 100%;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: {ACCENT}; color: #fff;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(16,185,129,0.15);
    }}

    /* Chat input */
    div[data-testid="stChatInput"] textarea {{ font-size: 0.95rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _sample_data() -> pd.DataFrame:
    return generate_transactions()


EXAMPLES = [
    "How should I budget my $5000 monthly income?",
    "How much emergency fund do I need if my expenses are $3000?",
    "Should I pay off debt or invest first?",
    "Monthly payment on a $300,000 mortgage at 4.5% for 30 years?",
]

# --- State ---------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "conn_error" not in st.session_state:
    st.session_state.conn_error = False
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "currency" not in st.session_state:
    st.session_state.currency = "USD"


def _is_connection_error(exc: Exception) -> bool:
    """True when the failure is the agent not being able to reach OpenAI."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("connection error", "apiconnection", "connecterror",
                                   "max retries", "failed to establish", "getaddrinfo"))


CONN_HINT = (
    "**Can't reach OpenAI — your machine is blocking the connection.**\n\n"
    "Your API key is fine; `python.exe` just can't get online. Fix one of these, "
    "then ask again:\n"
    "- **NetLimiter / firewall** → allow `python.exe` (and `venv\\Scripts\\python.exe`) internet access\n"
    "- **Antivirus HTTPS/SSL scanning** → disable it or whitelist `api.openai.com`\n"
    "- **Quick test** → switch to a phone hotspot; if it works there, it's your local firewall"
)


def handle_upload(uploaded_file):
    try:
        raw_bytes = uploaded_file.read()
        name = uploaded_file.name.lower()
        if name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str)
        else:
            try:
                raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str, encoding="utf-8-sig")
            except UnicodeDecodeError:
                raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str, encoding="cp1252")
        df = normalize(strip_preamble(raw))
        if df is None or df.empty:
            st.sidebar.error("Couldn't find a Date + Debit/Credit table.")
            return
        st.session_state.uploaded_df = df
        st.session_state.currency = detect_currency(raw)
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# --- Header --------------------------------------------------------------
st.markdown(
    '<div class="fg-hero">'
    '<p class="fg-title">WealthWise AI</p>'
    '<p class="fg-sub">AI personal finance advisor — budgeting, saving, debt, loans & investing.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Persistent banner while the connection is broken.
if st.session_state.conn_error:
    st.error(CONN_HINT, icon="🚫")


# --- Sidebar dashboard ---------------------------------------------------
with st.sidebar:
    st.markdown("## 📥 Upload Data")
    uploaded_file = st.file_uploader("Bank statement (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if st.button("Process Uploaded File", type="primary", use_container_width=True):
            handle_upload(uploaded_file)
            st.rerun()

    if st.session_state.uploaded_df is not None:
        if st.button("🔄 Reset to sample data", use_container_width=True):
            st.session_state.uploaded_df = None
            st.session_state.currency = "USD"
            st.rerun()

    st.divider()

    if st.session_state.messages:
        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


# --- Dashboard Data Setup ------------------------------------------------
current_df = st.session_state.uploaded_df if st.session_state.uploaded_df is not None else _sample_data()
symbol = CURRENCY_SYMBOLS.get(st.session_state.currency, "$")
dash_data = dashboard_snapshot(current_df, symbol=symbol)

# --- Dashboard Layout ----------------------------------------------------

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Income", f"{symbol}{dash_data['income']:,.0f}")
kpi2.metric("Spend", f"{symbol}{dash_data['spend']:,.0f}")
kpi3.metric("Net", f"{symbol}{dash_data['net']:,.0f}")
kpi4.metric("Savings Rate", f"{dash_data['rate']:.1f}%")

st.markdown("---")

# Budget & Charts
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 50/30/20 Budget")
    for b in dash_data["budget"]:
        pct = min(b['actual'] / b['target'], 1.0) if b['target'] > 0 else 0
        prog_color = "normal" if not b.get("isSavings") else "green"
        st.markdown(f"**{b['name']}** - {symbol}{b['actual']:,.0f} / {symbol}{b['target']:,.0f}")
        st.progress(pct)
    
    st.markdown("### Monthly Cash Flow")
    st.plotly_chart(cashflow_chart(dash_data['cashflow']), use_container_width=True, config={"displayModeBar": False})

with col_right:
    st.markdown("### Spending by Category")
    st.plotly_chart(spending_chart(dash_data['categories']), use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

col_trans, col_calc = st.columns([1.5, 1])

with col_trans:
    st.markdown("### Recent Transactions")
    tx_df = pd.DataFrame(dash_data["transactions"])
    if not tx_df.empty:
        # Reorder and format columns for display
        tx_disp = tx_df[["date", "name", "category", "amount"]].copy()
        tx_disp.columns = ["Date", "Merchant/Description", "Category", "Amount"]
        st.dataframe(tx_disp, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions to display.")

with col_calc:
    st.markdown("### Investment Growth Calculator")
    c1, c2 = st.columns(2)
    principal = c1.number_input("Principal", value=10000, step=1000)
    monthly_contrib = c2.number_input("Monthly Contrib", value=500, step=100)
    annual_rate = c1.slider("Annual Return (%)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
    years = c2.slider("Years", min_value=1, max_value=40, value=10, step=1)
    
    # Calculate series for chart using the existing calculator logic
    history = []
    for y in range(1, years + 1):
        res = calculate_investment_growth.invoke({
            "principal": float(principal),
            "annual_return": float(annual_rate),
            "years": int(y),
            "monthly_contribution": float(monthly_contrib)
        })
        history.append({
            "year": y,
            "principal": principal,
            "total_contributions": res["total_contributions"],
            "interest_earned": res["interest_earned"]
        })
        
    st.plotly_chart(investment_growth_chart(history), use_container_width=True, config={"displayModeBar": False})
    
    final_res = calculate_investment_growth.invoke({
        "principal": float(principal),
        "annual_return": float(annual_rate),
        "years": int(years),
        "monthly_contribution": float(monthly_contrib)
    })
    st.success(f"Final Value: **{symbol}{final_res['final_value']:,.0f}**")

st.markdown("---")

# --- Welcome + example chips (only before first message) -----------------
if not st.session_state.messages:
    st.markdown("### 💬 Ask WealthWise AI")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}"):
            st.session_state.pending = ex
            st.rerun()

# --- Render history ------------------------------------------------------
for role, content in st.session_state.messages:
    avatar = "🧑" if role == "user" else "💰"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# --- Handle input --------------------------------------------------------
prompt = st.chat_input("Ask a financial question...")
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="💰"):
        with st.spinner("Thinking..."):
            try:
                # Set active transactions for the semantic search tool to hit the right data
                set_active_transactions(current_df, st.session_state.thread_id)
                # Ground the chat in the current dashboard data
                grounded = f"{snapshot_summary(current_df, symbol=symbol)}\n\nUser question: {prompt}"
                reply = chat(grounded, thread_id=st.session_state.thread_id)
                st.session_state.conn_error = False
            except Exception as exc:
                if _is_connection_error(exc):
                    st.session_state.conn_error = True
                    reply = CONN_HINT
                else:
                    reply = f"⚠️ **Error:** {exc}"
        st.markdown(reply)
    st.session_state.messages.append(("assistant", reply))
    st.rerun()
