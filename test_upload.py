"""One check for the statement-CSV parser. Run: python test_upload.py"""

import pandas as pd

from src.backend.wealthwise.ingest import normalize


def test_normalize():
    # Category provided -> no network/auto-categorize, just column mapping.
    good = pd.DataFrame({"date": ["2026-01-01"], "Description": ["Rent"],
                         "AMOUNT": [-1500], "Category": ["Housing"]})
    out = normalize(good)
    assert set(["Date", "Description", "Amount", "Category"]).issubset(out.columns)
    assert out.Amount.iloc[0] == -1500 and out.Category.iloc[0] == "Housing"

    assert normalize(pd.DataFrame({"foo": [1], "bar": [2]})) is None  # missing columns


if __name__ == "__main__":
    test_normalize()
    print("ok")
