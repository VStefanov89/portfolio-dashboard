import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Daily Update", layout="wide")

# ==========================================================
# Load data
# ==========================================================

live_df = pd.read_csv("live_signals_portfolio.csv")
closed_df = pd.read_csv("close_signals_historycally.csv")

daily_df = pd.read_csv("daily_pnl.csv")
realized_df = pd.read_csv("realized_pnl.csv")
unrealized_df = pd.read_csv("unrealized_pnl.csv")

# ==========================================================
# Convert dates
# ==========================================================

for df in [live_df, closed_df, daily_df, realized_df, unrealized_df]:
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        format="mixed"
    ).dt.normalize()

# ==========================================================
# Build daily summary
# ==========================================================

summary_df = (
    daily_df
    .merge(
        realized_df.rename(columns={"NetPnL": "realized_daily"}),
        on="date",
        how="left"
    )
    .merge(
        unrealized_df.rename(columns={"NetPnL": "unrealized_pnl"}),
        on="date",
        how="left"
    )
)

summary_df = summary_df.sort_values("date").reset_index(drop=True)

summary_df["realized_daily"] = summary_df["realized_daily"].fillna(0)
summary_df["unrealized_pnl"] = summary_df["unrealized_pnl"].fillna(0)

summary_df["realized_pnl"] = summary_df["realized_daily"].cumsum()

summary_df["total_pnl"] = (
    summary_df["realized_pnl"] +
    summary_df["unrealized_pnl"]
)

# ==========================================================
# Latest portfolio
# ==========================================================

latest_date = summary_df["date"].max()

latest_summary = summary_df.loc[
    summary_df["date"] == latest_date
].iloc[0]

latest_portfolio = (
    live_df[live_df["date"] == latest_date]
    .copy()
)

# ==========================================================
# Header
# ==========================================================

st.title("Daily Portfolio Update")
st.subheader(f"Date: {latest_date.date()}")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Daily PnL",
    f"${latest_summary['daily_pnl']:,.2f}"
)

col2.metric(
    "Realized PnL",
    f"${latest_summary['realized_pnl']:,.2f}"
)

col3.metric(
    "Unrealized PnL",
    f"${latest_summary['unrealized_pnl']:,.2f}"
)

col4.metric(
    "Total PnL",
    f"${latest_summary['total_pnl']:,.2f}"
)

col5.metric(
    "Total BP",
    f"${latest_summary['total_bp']:,.0f}"
)

col6.metric(
    "Open Triplets",
    f"{int(latest_summary['triplet'])}"
)

st.divider()

# ==========================================================
# Daily summary
# ==========================================================

st.subheader("Daily Summary History")

st.dataframe(
    summary_df.sort_values("date", ascending=False),
    width="stretch"
)

# ==========================================================
# Charts
# ==========================================================

st.subheader("Daily PnL")
st.line_chart(
    summary_df.set_index("date")["daily_pnl"]
)

st.subheader("Realized PnL")
st.line_chart(
    summary_df.set_index("date")["realized_pnl"]
)

st.subheader("Unrealized PnL")
st.line_chart(
    summary_df.set_index("date")["unrealized_pnl"]
)

st.subheader("Total PnL")
st.line_chart(
    summary_df.set_index("date")["total_pnl"]
)

st.subheader("Buying Power")
st.line_chart(
    summary_df.set_index("date")["total_bp"]
)

# ==========================================================
# Current Portfolio
# ==========================================================

st.subheader("Current Portfolio")

show_cols = [
    "date",
    "uuid",
    "triplet",
    "calibrated_tail_p",
    "sector_triplet",
    "industry_triplet",
    "GrossPnL",
    "total_costs",
    "NetPnL",
    "daily_pnl",
    "ticker1",
    "ticker2",
    "ticker3",
    "price1",
    "price2",
    "price3",
    "shares1",
    "shares2",
    "shares3",
    "total_bp",
    "spread",
    "half_life",
    "traceT",
    "critical_vals",
    "zscore_composed",
    "min_days_to_earnings",
    "earnings_within_3d",
    "earnings_within_5d",
    "earnings_within_10d",
    "n_earnings_next_5d"
]

show_cols = [
    c for c in show_cols
    if c in latest_portfolio.columns
]

latest_display = latest_portfolio[show_cols]

if "calibrated_tail_p" in latest_display.columns:
    latest_display = latest_display.sort_values(
        "calibrated_tail_p",
        ascending=False
    )

st.dataframe(
    latest_display,
    width="stretch"
)

# ==========================================================
# Closed Trades
# ==========================================================

st.subheader("Closed Trades")

closed_display = closed_df.copy()

closed_cols = [
    c for c in show_cols
    if c in closed_display.columns
]

closed_display = (
    closed_display[closed_cols]
    .sort_values("date", ascending=False)
)

st.dataframe(
    closed_display,
    width="stretch"
)

# ==========================================================
# Risk Alerts
# ==========================================================

st.subheader("Risk Alerts")

if "calibrated_tail_p" in latest_portfolio.columns:

    risk_df = latest_portfolio[
        latest_portfolio["calibrated_tail_p"] >= 0.96
    ]

    if len(risk_df):

        st.warning(
            f"{len(risk_df)} positions have tail probability above 0.96"
        )

        st.dataframe(
            risk_df[show_cols]
            .sort_values(
                "calibrated_tail_p",
                ascending=False
            ),
            width="stretch"
        )

    else:

        st.success("No high tail-risk positions today.")

else:

    st.info("Column calibrated_tail_p not found.")
