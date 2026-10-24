import re
import pandas as pd
from src.backend.wealthwise.analysis import categorize_transaction
from src.backend.wealthwise.privacy import redact

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£",
                    "JPY": "¥", "AUD": "A$", "CAD": "C$", "SGD": "S$", "AED": "د.إ"}

def detect_currency(raw: pd.DataFrame) -> str:
    """Find the statement currency (e.g. 'Currency : INR') in the preamble.
    Defaults to USD when absent (keeps the built-in sample data in $)."""
    text = " ".join(str(v) for v in raw.to_numpy().ravel() if pd.notna(v))
    m = re.search(r"currency\s*[:\-]?\s*([A-Za-z]{3})", text, re.IGNORECASE)
    if m and m.group(1).upper() in CURRENCY_SYMBOLS:
        return m.group(1).upper()
    return "USD"

# Real bank statements name columns many ways. Map each field to its aliases.
_ALIASES = {
    "date": ("date", "transaction date", "txn date", "tran date", "value date",
             "posting date", "trans date", "date of transaction"),
    "description": ("description", "narration", "particulars", "details",
                    "remarks", "transaction details", "payee", "transaction remarks"),
    "amount": ("amount", "amt", "transaction amount", "txn amount"),
    "debit": ("debit", "debit amount", "withdrawal", "withdrawal amt",
              "withdrawal amt.", "withdrawal amount", "dr", "paid out"),
    "credit": ("credit", "credit amount", "deposit", "deposit amt",
               "deposit amt.", "deposit amount", "cr", "paid in"),
    "category": ("category", "type"),
}

def _pick(cols: dict, field: str):
    """Return the actual column name for a field, matching known aliases."""
    for alias in _ALIASES[field]:
        if alias in cols:
            return cols[alias]
    return None

def _money(series: pd.Series) -> pd.Series:
    """Parse a money column: strip currency symbols, commas, spaces, blanks."""
    cleaned = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

_KNOWN_HEADERS = {a for aliases in _ALIASES.values() for a in aliases}

def strip_preamble(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Bank statements (e.g. SBI) put account-info rows above the transaction
    table. Read with header=None, then find the real header row (the one naming
    a date column plus a debit/credit/amount column) and reframe from there."""
    for i in range(min(len(raw), 40)):
        row = [str(v).lower().strip() for v in raw.iloc[i].tolist()]
        if "date" in row and any(c in row for c in
                                 ("debit", "credit", "amount", "withdrawal amt",
                                  "deposit amt", "withdrawal amt.", "deposit amt.")):
            body = raw.iloc[i + 1:].copy()
            body.columns = [str(v).strip() for v in raw.iloc[i].tolist()]
            return body.reset_index(drop=True)
    return None

def normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map a real bank-statement CSV to Date/Description/Amount(/Category).

    Handles aliased column names, separate debit/credit columns, day-first
    dates, and comma-formatted amounts. Returns None if date+description plus
    some amount column are missing.
    """
    if df is None:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    date_c, desc_c = _pick(cols, "date"), _pick(cols, "description")
    amt_c = _pick(cols, "amount")
    deb_c, cred_c = _pick(cols, "debit"), _pick(cols, "credit")
    if date_c is None or desc_c is None or not (amt_c or deb_c or cred_c):
        return None

    if amt_c:
        amount = _money(df[amt_c])
    else:
        # Debit/credit split: debit -> money out (negative), credit -> in.
        debit = _money(df[deb_c]).fillna(0) if deb_c else 0
        credit = _money(df[cred_c]).fillna(0) if cred_c else 0
        amount = credit - debit

    out = pd.DataFrame({
        # dayfirst=True: most non-US banks use DD/MM/YYYY.
        "Date": pd.to_datetime(df[date_c], errors="coerce", dayfirst=True),
        "Description": df[desc_c].astype(str).str.strip(),
        "Amount": amount,
    })
    out["Merchant"] = out["Description"].map(_merchant)
    cat_c = _pick(cols, "category")
    if cat_c:
        out["Category"] = df[cat_c].astype(str)
    out = out.dropna(subset=["Date", "Amount"]).reset_index(drop=True)
    if "Category" not in out.columns:
        out["Category"] = _auto_categorize(out)
    return out

def _merchant(desc: str) -> str:
    """Pull the payee/merchant out of a raw bank narration.

    Indian UPI/IMPS strings look like
    'WDL TFR UPI/DR/1234567890/MerchantName/BANK/handle/...': the party name
    is the segment right after the long numeric reference. Falls back to the
    cleaned head of the description for non-UPI rows (mandates, charges, etc.).
    """
    parts = [p.strip() for p in str(desc).split("/")]
    for i, p in enumerate(parts[:-1]):
        if re.fullmatch(r"\d{6,}", p) and parts[i + 1]:
            return parts[i + 1]
    cleaned = re.sub(r"\b\d{6,}\b", "", str(desc)).strip()
    cleaned = re.sub(r"\bAT \d+ \w+$", "", cleaned).strip()
    return (cleaned or str(desc))[:40]

# Free keyword map for common Indian merchants -> category. Avoids an LLM call
# for the bulk of UPI spend; only unknown merchants fall through to the model.
_MERCHANT_KEYWORDS = {
    "blinkit": "Groceries", "zepto": "Groceries", "bigbasket": "Groceries",
    "dmart": "Groceries", "jiomart": "Groceries", "instamart": "Groceries",
    "grofers": "Groceries", "swiggy": "Dining", "zomato": "Dining",
    "domino": "Dining", "mcdonald": "Dining", "kfc": "Dining", "cafe": "Dining",
    "restaurant": "Dining", "eatclub": "Dining", "rapido": "Transportation",
    "uber": "Transportation", "ola": "Transportation", "irctc": "Transportation",
    "redbus": "Transportation", "makemytr": "Transportation",
    "goibibo": "Transportation", "ixigo": "Transportation",
    "indrive": "Transportation", "petrol": "Transportation",
    "fuel": "Transportation", "hpcl": "Transportation", "iocl": "Transportation",
    "bpcl": "Transportation", "metro": "Transportation",
    "hotstar": "Entertainment", "netflix": "Entertainment",
    "spotify": "Entertainment", "prime": "Entertainment",
    "bookmyshow": "Entertainment", "pvr": "Entertainment", "inox": "Entertainment",
    "youtube": "Entertainment", "amazon": "Shopping", "flipkart": "Shopping",
    "myntra": "Shopping", "ajio": "Shopping", "meesho": "Shopping",
    "nykaa": "Shopping", "snapdeal": "Shopping", "tatacliq": "Shopping",
    "recharge": "Utilities", "jio": "Utilities", "airtel": "Utilities",
    "vodafone": "Utilities", "electricity": "Utilities", "broadband": "Utilities",
    "cred": "Utilities", "tatapower": "Utilities", "bescom": "Utilities",
    "pharmacy": "Healthcare", "apollo": "Healthcare", "pharmeasy": "Healthcare",
    "1mg": "Healthcare", "netmeds": "Healthcare", "hospital": "Healthcare",
    "clinic": "Healthcare", "medical": "Healthcare", "udemy": "Education",
    "coursera": "Education", "byju": "Education", "unacademy": "Education",
}

def _keyword_category(merchant: str) -> str | None:
    s = merchant.lower()
    for kw, cat in _MERCHANT_KEYWORDS.items():
        if kw in s:
            return cat
    return None

def _auto_categorize(df: pd.DataFrame) -> list[str]:
    """Categorize by extracted merchant. Known Indian merchants are matched for
    free via a keyword map; unknown merchants go to the LLM (deduped, capped at
    50 to bound cost); the rest default to Income (credits) / Transfers (debits).
    """
    base = [_keyword_category(m) for m in df["Merchant"]]

    todo = sorted({m for m, c in zip(df["Merchant"], base) if c is None})
    mapping: dict[str, str] = {}
    if 0 < len(todo) <= 50:  # bound LLM calls regardless of statement size
        for m in todo:
            amt = float(df.loc[df.Merchant == m, "Amount"].iloc[0])
            try:
                mapping[m] = categorize_transaction(redact(m), amt).category
            except Exception:  # noqa: BLE001 - offline/blocked -> heuristic fallback
                pass

    out = []
    for m, amt, c in zip(df["Merchant"], df["Amount"], base):
        c = c or mapping.get(m)
        if c is None:
            c = "Income" if amt > 0 else "Transfers"
        out.append(c)
    return out
