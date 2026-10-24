from src.backend.wealthwise.tools.calculators import CALCULATOR_TOOLS
from src.backend.wealthwise.tools.knowledge_tool import financial_knowledge_lookup
from src.backend.wealthwise.tools.transaction_tools import (
    categorize_transaction,
    find_similar_transactions,
)

# All tools the agent can call.
ALL_TOOLS = [
    *CALCULATOR_TOOLS,
    financial_knowledge_lookup,
    categorize_transaction,
    find_similar_transactions,
]

__all__ = [
    "ALL_TOOLS",
    "CALCULATOR_TOOLS",
    "financial_knowledge_lookup",
    "categorize_transaction",
    "find_similar_transactions",
]
