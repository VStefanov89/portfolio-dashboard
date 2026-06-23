import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Daily Update", layout="wide")

# =========================
# Load data
# =========================

live_df = pd.read_csv("live_signals_portfolio.csv")
closed_df = pd.read_csv("close_signals_historycally.csv")

live_df["date"] = pd.to_datetime(live_df["date"])
closed_df["date"] = pd.to_datetime(closed_df["date"])

live_df = live_df.sort_values(["date", "uuid"]).reset_index(drop=True)
closed_df = closed_df.sort_values(["date", "uuid"]).reset_index(drop=True)

# =========================
# Build daily summary from both dfs
# =========================

# Unrealized PnL from live open portfolio by date
unrealized_daily = (
    live_df
    .groupby("date", as_index=False)
    .agg(
        unrealized_pnl=("NetPnL", "sum"),
        total_bp=("total_bp", "sum"),
        open_triplets=("uuid", "nunique")
    )
)

# Daily PnL from live portfolio, if column exists
if "daily_pnl" in live_df.columns:
    daily_live_pnl = (
        live_df
        .groupby("date", as_index=False)["daily_pnl"]
        .sum()
        .rename(columns={"daily_pnl": "daily_pnl"})
    )
else:
    daily_live_pnl = unrealized_daily[["date"]].copy()
    daily_live_pnl["daily_pnl"] = 0.0

# Realized PnL from closed signals
# Take one final row per closed uuid
closed_latest_per_uuid = (
    closed_df
    .sort_values(["uuid", "date"])
    .drop_duplicates("uuid", keep="last")
)

realized_daily = (
    closed_latest_per_uuid
    .groupby("date", as_index=False)
    .agg(
        realized_pnl_daily=("NetPnL", "sum"),
        closed_triplets=("uuid", "nunique")
    )
)

# Create full date index from both live and closed
all_dates = pd.DataFrame({
    "date": sorted(
        set(live_df["date"].dropna().unique())
        | set(closed_df["date"].dropna().unique())
    )
})

summary_df = all_dates.merge(daily_live_pnl, on="date", how="left")
summary_df = summary_df.merge(unrealized_daily, on="date", how="left")
summary_df = summary_df.merge(realized_daily, on="date", how="left")

summary_df = summary_df.sort_values("date").reset_index(drop=True)

summary_df["daily_pnl"] = summary_df["daily_pnl"].fillna(0)
summary_df["unrealized_pnl"] = summary_df["unrealized_pnl"].fillna(0)
summary_df["total_bp"] = summary_df["total_bp"].fillna(0)
summary_df["open_triplets"] = summary_df["open_triplets"].fillna(0).astype(int)
summary_df["realized_pnl_daily"] = summary_df["realized_pnl_daily"].fillna(0)
summary_df["closed_triplets"] = summary_df["closed_triplets"].fillna(0).astype(int)

# Cumulative realized PnL from all closed trades over time
summary_df["realized_pnl"] = summary_df["realized_pnl_daily"].cumsum()

# Total PnL = realized closed PnL + current unrealized open PnL
summary_df["total_pnl"] = (
    summary_df["realized_pnl"] + summary_df["unrealized_pnl"]
)

# Optional cumulative daily pnl, keeping daily_pnl as-is
summary_df["cumulative_daily_pnl"] = summary_df["daily_pnl"].cumsum()

# =========================
# Latest date
# =========================

latest_date = summary_df["date"].max()

latest_summary = summary_df[summary_df["date"] == latest_date].iloc[0]
latest_portfolio = live_df[live_df["date"] == latest_date].copy()

# =========================
# Header
# =========================

st.title("Daily Portfolio Update")
st.subheader(f"Date: {latest_date.date()}")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Daily PnL", f"${latest_summary['daily_pnl']:,.2f}")
col2.metric("Realized PnL", f"${latest_summary['realized_pnl']:,.2f}")
col3.metric("Unrealized PnL", f"${latest_summary['unrealized_pnl']:,.2f}")
col4.metric("Total PnL", f"${latest_summary['total_pnl']:,.2f}")
col5.metric("Total BP", f"${latest_summary['total_bp']:,.0f}")
col6.metric("Open Triplets", f"{latest_summary['open_triplets']:,.0f}")

st.divider()

# =========================
# Summary
# =========================

st.subheader("Daily Summary History")
st.dataframe(summary_df.sort_values("date", ascending=False), width="stretch")

st.subheader("Daily PnL Chart")
st.line_chart(summary_df.set_index("date")["daily_pnl"])

st.subheader("Cumulative Daily PnL Chart")
st.line_chart(summary_df.set_index("date")["cumulative_daily_pnl"])

st.subheader("Realized PnL Chart")
st.line_chart(summary_df.set_index("date")["realized_pnl"])

st.subheader("Unrealized PnL Chart")
st.line_chart(summary_df.set_index("date")["unrealized_pnl"])

st.subheader("Total PnL Chart")
st.line_chart(summary_df.set_index("date")["total_pnl"])

st.subheader("BP Chart")
st.line_chart(summary_df.set_index("date")["total_bp"])

# =========================
# Latest portfolio
# =========================

st.subheader("Latest Portfolio Positions")

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
    latest_display = latest_display.sort_values("calibrated_tail_p", ascending=False)

st.dataframe(latest_display, width="stretch")

# =========================
# Closed trades
# =========================

st.subheader("Closed Trades History")

closed_display_cols = [c for c in show_cols if c in closed_latest_per_uuid.columns]

closed_display = closed_latest_per_uuid[closed_display_cols].sort_values(
    "date",
    ascending=False
)

st.dataframe(closed_display, width="stretch")

# =========================
# Risk alerts
# =========================

st.subheader("Risk Alerts")

if "calibrated_tail_p" in latest_portfolio.columns:
    risk_df = latest_portfolio[
        latest_portfolio["calibrated_tail_p"] >= 0.96
    ].copy()

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
