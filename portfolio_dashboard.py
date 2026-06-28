import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Daily Update", layout="wide")

live_df = pd.read_csv("live_signals_portfolio.csv")
closed_df = pd.read_csv("close_signals_historycally.csv")

daily_df = pd.read_csv("daily_pnl.csv")
realized_df = pd.read_csv("realized_pnl.csv")
unrealized_df = pd.read_csv("unrealized_pnl.csv")

for df in [live_df, closed_df, daily_df, realized_df, unrealized_df]:
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        format="mixed"
    ).dt.normalize()

daily_df = daily_df[["date", "daily_pnl", "total_bp"]].copy()

realized_df = realized_df[["date", "NetPnL"]].copy()
realized_df = realized_df.rename(columns={"NetPnL": "realized_daily"})

unrealized_df = unrealized_df[["date", "NetPnL"]].copy()
unrealized_df = unrealized_df.rename(columns={"NetPnL": "unrealized_pnl"})

summary_df = (
    daily_df
    .merge(realized_df, on="date", how="left")
    .merge(unrealized_df, on="date", how="left")
)

summary_df = summary_df.sort_values("date").reset_index(drop=True)

summary_df["daily_pnl"] = summary_df["daily_pnl"].fillna(0)
summary_df["total_bp"] = summary_df["total_bp"].fillna(0)
summary_df["realized_daily"] = summary_df["realized_daily"].fillna(0)
summary_df["unrealized_pnl"] = summary_df["unrealized_pnl"].fillna(0)

summary_df["realized_pnl"] = summary_df["realized_daily"].cumsum()

# Total PnL = cumulative daily PnL / equity curve
summary_df["total_pnl"] = summary_df["daily_pnl"].cumsum()

latest_date = summary_df["date"].max()
latest_summary = summary_df[summary_df["date"] == latest_date].iloc[0]

latest_portfolio = live_df[live_df["date"] == latest_date].copy()
open_triplets = latest_portfolio["uuid"].nunique()

st.title("Daily Portfolio Update")
st.subheader(f"Date: {latest_date.date()}")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total PnL", f"${latest_summary['total_pnl']:,.2f}")
col2.metric("Daily PnL", f"${latest_summary['daily_pnl']:,.2f}")
col3.metric("Unrealized PnL", f"${latest_summary['unrealized_pnl']:,.2f}")
col4.metric("Total BP", f"${latest_summary['total_bp']:,.0f}")
col5.metric("Open Triplets", f"{open_triplets:,}")

st.divider()

st.subheader("Total PnL")
st.line_chart(summary_df.set_index("date")["total_pnl"])

st.subheader("Daily Summary History")
st.dataframe(
    summary_df.sort_values("date", ascending=False),
    width="stretch"
)

st.subheader("Current Portfolio")

show_cols = [
    "date", "uuid", "triplet", "calibrated_tail_p",
    "sector_triplet", "industry_triplet",
    "GrossPnL", "total_costs", "NetPnL", "daily_pnl",
    "ticker1", "ticker2", "ticker3",
    "price1", "price2", "price3",
    "shares1", "shares2", "shares3",
    "total_bp",
    "spread", "half_life", "traceT", "critical_vals", "zscore_composed",
    "min_days_to_earnings", "earnings_within_3d",
    "earnings_within_5d", "earnings_within_10d",
    "n_earnings_next_5d"
]

show_cols = [c for c in show_cols if c in latest_portfolio.columns]

latest_display = latest_portfolio[show_cols].copy()

if "calibrated_tail_p" in latest_display.columns:
    latest_display = latest_display.sort_values(
        "calibrated_tail_p",
        ascending=False
    )

st.dataframe(latest_display, width="stretch")

st.subheader("Closed Trades")

closed_display = closed_df.copy()
closed_cols = [c for c in show_cols if c in closed_display.columns]

if len(closed_cols) > 0:
    closed_display = closed_display[closed_cols].sort_values(
        "date",
        ascending=False
    )
    st.dataframe(closed_display, width="stretch")
else:
    st.info("No matching closed trade columns found.")

st.subheader("Risk Alerts")

if "calibrated_tail_p" in latest_portfolio.columns:
    risk_df = latest_portfolio[
        latest_portfolio["calibrated_tail_p"] >= 0.96
    ].copy()

    if len(risk_df) > 0:
        st.warning(f"{len(risk_df)} positions have tail probability above 0.96")
        st.dataframe(
            risk_df[show_cols].sort_values(
                "calibrated_tail_p",
                ascending=False
            ),
            width="stretch"
        )
    else:
        st.success("No high tail-risk positions today.")
else:
    st.info("Column calibrated_tail_p not found.")
