import plotly.graph_objects as go
import pandas as pd

ACCENT = "#10b981"
ACCENT_DARK = "#059669"
BORDER = "#262d3a"

def spending_chart(categories_data) -> go.Figure:
    """A smooth horizontal bar chart of spending by category."""
    if not categories_data:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # categories_data is a list of dicts: {"name": name, "amount": amount, "colors": [c1, c2]}
    # sort by amount ascending for horizontal bar chart
    sorted_cats = sorted(categories_data, key=lambda x: x["amount"])
    names = [c["name"] for c in sorted_cats]
    amounts = [c["amount"] for c in sorted_cats]
    colors = [c.get("colors", [ACCENT_DARK, ACCENT])[1] for c in sorted_cats]

    fig = go.Figure(
        go.Bar(
            x=amounts, y=names, orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=True, gridcolor=BORDER),
        yaxis=dict(showgrid=False),
    )
    return fig

def cashflow_chart(cashflow_data) -> go.Figure:
    """Monthly net cash flow as a Plotly line/bar chart."""
    if not cashflow_data:
        fig = go.Figure()
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    labels = [c["label"] for c in cashflow_data]
    net_values = [c["net"] for c in cashflow_data]
    colors = [ACCENT if v >= 0 else "#e11d48" for v in net_values]

    fig = go.Figure(
        go.Bar(
            x=labels, y=net_values,
            marker=dict(color=colors),
            hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER),
    )
    return fig

def investment_growth_chart(history_data) -> go.Figure:
    """Plot investment growth over time."""
    if not history_data:
        return go.Figure()

    years = [d["year"] for d in history_data]
    principals = [d["principal"] for d in history_data]
    contributions = [d["total_contributions"] for d in history_data]
    interests = [d["interest_earned"] for d in history_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=principals, name="Initial", marker_color="#64748b"))
    fig.add_trace(go.Bar(x=years, y=contributions, name="Contributions", marker_color="#3b82f6"))
    fig.add_trace(go.Bar(x=years, y=interests, name="Interest", marker_color=ACCENT))

    fig.update_layout(
        barmode='stack',
        height=300, margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(title="Year", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
