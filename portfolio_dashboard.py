import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Daily Update", layout="wide")

bucket = "Daily_signals"

portfolio_df = pd.read_csv(f"{bucket}/live_signals_portfolio.csv")
summary_df = pd.read_csv(f"{bucket}/daily_stats.csv")

portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
summary_df["date"] = pd.to_datetime(summary_df["date"])

latest_date = summary_df["date"].max()

latest_summary = summary_df[summary_df["date"] == latest_date].iloc[0]
latest_portfolio = portfolio_df[portfolio_df["date"] == latest_date].copy()

st.title("Daily Portfolio Update")
st.subheader(f"Date: {latest_date.date()}")

col1, col2, col3 = st.columns(3)

col1.metric("Daily PnL", f"${latest_summary['daily_pnl']:,.2f}")
col2.metric("Total BP", f"${latest_summary['total_bp']:,.0f}")
col3.metric("Open Triplets", f"{len(latest_portfolio):,.0f}")

st.divider()

st.subheader("Daily Summary History")
st.dataframe(summary_df.sort_values("date", ascending=False), width="stretch")

st.subheader("PnL Chart")
st.line_chart(summary_df.set_index("date")["daily_pnl"])

st.subheader("BP Chart")
st.line_chart(summary_df.set_index("date")["total_bp"])

st.subheader("Latest Portfolio Positions")

show_cols = [
    "date", "uuid", "triplet", "calibrated_tail_p",
    "GrossPnL", "total_costs", "NetPnL", "daily_pnl",
    "ticker1", "ticker2", "ticker3",
    "price1", "price2", "price3",
    "shares1", "shares2", "shares3",
    "total_bp", "sector_triplet", "industry_triplet",
    "spread", "half_life", "traceT", "critical_vals", "zscore_composed",
    "min_days_to_earnings", "earnings_within_3d",
    "earnings_within_5d", "earnings_within_10d",
    "n_earnings_next_5d"
]

show_cols = [c for c in show_cols if c in latest_portfolio.columns]

latest_display = latest_portfolio[show_cols]

if "calibrated_tail_p" in latest_display.columns:
    latest_display = latest_display.sort_values("calibrated_tail_p", ascending=False)

st.dataframe(latest_display, width="stretch")

st.subheader("Risk Alerts")

if "calibrated_tail_p" in latest_portfolio.columns:
    risk_df = latest_portfolio[latest_portfolio["calibrated_tail_p"] >= 0.96].copy()

    if len(risk_df) > 0:
        st.warning(f"{len(risk_df)} positions have tail probability above 0.96")
        st.dataframe(
            risk_df[show_cols].sort_values("calibrated_tail_p", ascending=False),
            width="stretch"
        )
    else:
        st.success("No high tail-risk positions today.")
else:
    st.info("Column calibrated_tail_p not found.")