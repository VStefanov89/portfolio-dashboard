import streamlit as st
import boto3
import pandas as pd
import numpy as np
from io import BytesIO


BUCKET = st.secrets["aws"]["bucket_name"]


@st.cache_data(ttl=60)
def read_csv_from_s3(key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["aws"]["access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["secret_access_key"],
        region_name=st.secrets["aws"]["region"]
    )

    response = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    return pd.read_csv(
        BytesIO(response["Body"].read())
    )



st.set_page_config(page_title="Portfolio Daily Update", layout="wide")

# =========================
# Helper functions
# =========================

def max_drawdown(equity):
    equity = pd.Series(equity).astype(float)
    running_max = equity.cummax()
    dd = equity / running_max - 1
    dd_dollar = equity - running_max
    return dd.min(), dd, dd_dollar.min(), dd_dollar


def format_money(x):
    return f"${x:,.0f}"


def format_pct(x):
    return f"{x:.2%}"


def format_number(x):
    return f"{x:,.2f}"


def format_int(x):
    return f"{int(x):,}"


def strategy_statistics_from_df(
    df,
    starting_capital=3_000_000,
    date_col="date",
    pnl_col="daily_pnl",
    bp_col="total_bp",
    n_mc_paths=5_000,
    random_state=42
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed").dt.normalize()

    daily = (
        df.groupby(date_col)
        .agg(
            daily_pnl=(pnl_col, "sum"),
            total_bp=(bp_col, "sum")
        )
        .reset_index()
        .sort_values(date_col)
    )

    daily["return_on_capital"] = daily["daily_pnl"] / starting_capital
    daily["return_on_deployed_bp"] = daily["daily_pnl"] / daily["total_bp"].replace(0, np.nan)
    daily["bp_utilization"] = daily["total_bp"] / starting_capital
    daily["equity"] = starting_capital + daily["daily_pnl"].cumsum()

    n_days = len(daily)
    years = n_days / 252 if n_days > 0 else np.nan

    total_pnl = daily["daily_pnl"].sum()
    total_return = daily["equity"].iloc[-1] / starting_capital - 1

    cagr = (
        (daily["equity"].iloc[-1] / starting_capital) ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    ret_std = daily["return_on_capital"].std(ddof=1)
    ann_vol = ret_std * np.sqrt(252)

    sharpe = (
        daily["return_on_capital"].mean() / ret_std * np.sqrt(252)
        if ret_std != 0 and not pd.isna(ret_std)
        else np.nan
    )

    max_dd, dd_series, max_dd_dollar, dd_dollar_series = max_drawdown(daily["equity"])

    daily["drawdown"] = dd_series
    daily["drawdown_dollar"] = dd_dollar_series

    gross_profit = daily.loc[daily["daily_pnl"] > 0, "daily_pnl"].sum()
    gross_loss = daily.loc[daily["daily_pnl"] < 0, "daily_pnl"].sum()

    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.inf

    summary = pd.Series({
        "n_days": n_days,
        "starting_capital": starting_capital,
        "ending_equity": daily["equity"].iloc[-1],
        "total_pnl": total_pnl,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_dollar": max_dd_dollar,
        "win_rate_days": (daily["daily_pnl"] > 0).mean(),
        "profit_factor": profit_factor,
        "avg_daily_pnl": daily["daily_pnl"].mean(),
        "median_daily_pnl": daily["daily_pnl"].median(),
        "best_day": daily["daily_pnl"].max(),
        "worst_day": daily["daily_pnl"].min(),
        "avg_bp_utilization": daily["bp_utilization"].mean(),
        "median_bp_utilization": daily["bp_utilization"].median(),
        "max_bp_utilization": daily["bp_utilization"].max(),
        "avg_deployed_return": daily["return_on_deployed_bp"].mean(),
    })

    rng = np.random.default_rng(random_state)
    daily_returns = daily["return_on_capital"].to_numpy()

    mc_rows = []

    if n_days > 1:
        for _ in range(n_mc_paths):
            sampled_returns = rng.choice(daily_returns, size=n_days, replace=True)
            path_equity = starting_capital * np.cumprod(1 + sampled_returns)

            path_total_return = path_equity[-1] / starting_capital - 1
            path_cagr = (path_equity[-1] / starting_capital) ** (1 / years) - 1

            path_max_dd, _, path_max_dd_dollar, _ = max_drawdown(path_equity)

            path_std = sampled_returns.std(ddof=1)

            path_sharpe = (
                sampled_returns.mean() / path_std * np.sqrt(252)
                if path_std != 0
                else np.nan
            )

            mc_rows.append({
                "final_equity": path_equity[-1],
                "total_return": path_total_return,
                "cagr": path_cagr,
                "max_drawdown": path_max_dd,
                "max_drawdown_dollar": path_max_dd_dollar,
                "sharpe": path_sharpe
            })

    mc = pd.DataFrame(mc_rows)

    if len(mc) > 0:
        mc_summary = pd.DataFrame({
            "p5": mc.quantile(0.05),
            "median": mc.quantile(0.50),
            "p95": mc.quantile(0.95)
        })

        risk_probs = pd.Series({
            "prob_losing_money": (mc["total_return"] < 0).mean(),
            "prob_drawdown_worse_than_5pct": (mc["max_drawdown"] < -0.05).mean(),
            "prob_drawdown_worse_than_10pct": (mc["max_drawdown"] < -0.10).mean(),
            "prob_return_above_20pct": (mc["total_return"] > 0.20).mean(),
            "prob_return_above_30pct": (mc["total_return"] > 0.30).mean(),
        })
    else:
        mc_summary = pd.DataFrame()
        risk_probs = pd.Series(dtype=float)

    friendly_summary = pd.DataFrame({
        "Metric": [
            "Trading Days",
            "Starting Capital",
            "Ending Equity",
            "Total PnL",
            "Total Return",
            "CAGR",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Maximum Drawdown ($)",
            "Winning Days",
            "Profit Factor",
            "Average Daily PnL",
            "Median Daily PnL",
            "Best Day",
            "Worst Day",
            "Average BP Utilization",
            "Median BP Utilization",
            "Maximum BP Utilization",
            "Average Deployed Return"
        ],
        "Value": [
            format_int(summary["n_days"]),
            format_money(summary["starting_capital"]),
            format_money(summary["ending_equity"]),
            format_money(summary["total_pnl"]),
            format_pct(summary["total_return"]),
            format_pct(summary["cagr"]),
            format_pct(summary["annualized_volatility"]),
            format_number(summary["sharpe"]),
            format_pct(summary["max_drawdown"]),
            format_money(summary["max_drawdown_dollar"]),
            format_pct(summary["win_rate_days"]),
            format_number(summary["profit_factor"]) + "x",
            format_money(summary["avg_daily_pnl"]),
            format_money(summary["median_daily_pnl"]),
            format_money(summary["best_day"]),
            format_money(summary["worst_day"]),
            format_pct(summary["avg_bp_utilization"]),
            format_pct(summary["median_bp_utilization"]),
            format_pct(summary["max_bp_utilization"]),
            format_pct(summary["avg_deployed_return"]),
        ]
    })

    if len(mc_summary) > 0:
        friendly_mc_summary = mc_summary.copy().astype(object)

        for row in friendly_mc_summary.index:
            for col in friendly_mc_summary.columns:
                val = mc_summary.loc[row, col]

                if pd.isna(val):
                    friendly_mc_summary.loc[row, col] = ""
                elif row in ["final_equity", "max_drawdown_dollar"]:
                    friendly_mc_summary.loc[row, col] = format_money(val)
                elif row in ["total_return", "cagr", "max_drawdown"]:
                    friendly_mc_summary.loc[row, col] = format_pct(val)
                elif row == "sharpe":
                    friendly_mc_summary.loc[row, col] = format_number(val)
                else:
                    friendly_mc_summary.loc[row, col] = format_number(val)

        friendly_mc_summary = (
            friendly_mc_summary
            .reset_index()
            .rename(columns={
                "index": "Metric",
                "p5": "P5",
                "median": "Median",
                "p95": "P95"
            })
        )

        friendly_risk_probs = pd.DataFrame({
            "Metric": [
                "Probability of Losing Money",
                "Probability DD Worse Than -5%",
                "Probability DD Worse Than -10%",
                "Probability Return Above 20%",
                "Probability Return Above 30%",
            ],
            "Value": [
                format_pct(risk_probs["prob_losing_money"]),
                format_pct(risk_probs["prob_drawdown_worse_than_5pct"]),
                format_pct(risk_probs["prob_drawdown_worse_than_10pct"]),
                format_pct(risk_probs["prob_return_above_20pct"]),
                format_pct(risk_probs["prob_return_above_30pct"]),
            ]
        })
    else:
        friendly_mc_summary = pd.DataFrame(columns=["Metric", "P5", "Median", "P95"])
        friendly_risk_probs = pd.DataFrame(columns=["Metric", "Value"])

    return (
        daily,
        summary,
        mc,
        mc_summary,
        risk_probs,
        friendly_summary,
        friendly_mc_summary,
        friendly_risk_probs
    )


# =========================
# Load data
# =========================

live_df = read_csv_from_s3(
    "dashboard/live_signals_portfolio.csv"
)

closed_df = read_csv_from_s3(
    "dashboard/close_signals_historycally.csv"
)

daily_df = read_csv_from_s3(
    "dashboard/daily_pnl.csv"
)


# live_df = pd.read_csv("live_signals_portfolio.csv")
# closed_df = pd.read_csv("close_signals_historycally.csv")
# daily_df = pd.read_csv("daily_pnl.csv")

for df in [live_df, closed_df, daily_df]:
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        format="mixed"
    ).dt.normalize()

# =========================
# Prepare daily summary
# =========================

daily_df = daily_df.copy()

daily_df = daily_df.rename(columns={
    "n_signal_rows_x": "open_triplets",
    "n_signal_rows_y": "closed_triplets",
    "total_bp_y": "realized_bp",
    "cum_pnl": "total_pnl"
})

needed_cols = [
    "date",
    "unrealized_daily_pnl",
    "open_triplets",
    "total_bp",
    "realised_daily_pnl",
    "closed_triplets",
    "realized_bp",
    "daily_pnl",
    "total_pnl",
    "strategy_equity",
    "spy_equity"
]

existing_needed_cols = [c for c in needed_cols if c in daily_df.columns]
summary_df = daily_df[existing_needed_cols].copy()

summary_df = summary_df.sort_values("date").reset_index(drop=True)

numeric_cols = [
    "unrealized_daily_pnl",
    "open_triplets",
    "total_bp",
    "realised_daily_pnl",
    "closed_triplets",
    "realized_bp",
    "daily_pnl",
    "total_pnl",
    "strategy_equity",
    "spy_equity"
]

for col in numeric_cols:
    if col in summary_df.columns:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0)

if "open_triplets" in summary_df.columns:
    summary_df["open_triplets"] = summary_df["open_triplets"].astype(int)

if "total_pnl" not in summary_df.columns:
    summary_df["total_pnl"] = summary_df["daily_pnl"].cumsum()

# =========================
# Strategy statistics
# =========================

starting_capital = 3_000_000

(
    stats_daily,
    stats_summary,
    mc,
    mc_summary,
    risk_probs,
    friendly_summary,
    friendly_mc_summary,
    friendly_risk_probs
) = strategy_statistics_from_df(
    summary_df[["date", "daily_pnl", "total_bp"]],
    starting_capital=starting_capital,
    n_mc_paths=5_000
)

# =========================
# Latest data
# =========================

latest_date = summary_df["date"].max()
latest_summary = summary_df[summary_df["date"] == latest_date].iloc[0]

latest_portfolio = live_df[live_df["date"] == latest_date].copy()

open_triplets = (
    int(latest_summary["open_triplets"])
    if "open_triplets" in latest_summary.index
    else latest_portfolio["uuid"].nunique()
)

# =========================
# Header
# =========================

st.title("Daily Portfolio Update")
st.subheader(f"Date: {latest_date.date()}")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total PnL", f"${latest_summary['total_pnl']:,.2f}")
col2.metric("Daily PnL", f"${latest_summary['daily_pnl']:,.2f}")
col3.metric("Unrealized Daily PnL", f"${latest_summary['unrealized_daily_pnl']:,.2f}")
col4.metric("Total BP", f"${latest_summary['total_bp']:,.0f}")
col5.metric("Open Triplets", f"{open_triplets:,}")

st.divider()

# =========================
# Total PnL graph
# =========================


pnl_chart = summary_df[["date", "total_pnl"]].copy()
pnl_chart["date"] = pnl_chart["date"].dt.date

st.subheader("Total PnL")
st.line_chart(pnl_chart.set_index("date"))

# =========================
# Strategy vs SPY graph
# =========================

if {"strategy_equity", "spy_equity"}.issubset(summary_df.columns):

    comparison_df = summary_df[
        ["date", "strategy_equity", "spy_equity"]
    ].copy()

    comparison_df["date"] = comparison_df["date"].dt.date

    comparison_df = comparison_df.set_index("date").rename(
        columns={
            "strategy_equity": "Strategy",
            "spy_equity": "SPY"
        }
    )

    st.subheader("Strategy vs SPY")
    st.line_chart(comparison_df)

else:
    st.info("strategy_equity and/or spy_equity columns not found in daily_pnl.csv")

# =========================
# Strategy statistics
# =========================

st.subheader("Strategy Statistics")

st.dataframe(
    friendly_summary,
    width="stretch",
    hide_index=True
)

st.subheader("Monte Carlo Summary")

st.dataframe(
    friendly_mc_summary,
    width="stretch",
    hide_index=True
)

st.subheader("Risk Probabilities")

st.dataframe(
    friendly_risk_probs,
    width="stretch",
    hide_index=True
)

# =========================
# Daily summary
# =========================

st.subheader("Daily Summary History")

st.dataframe(
    summary_df.sort_values("date", ascending=False),
    width="stretch"
)

# =========================
# Current portfolio
# =========================

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

# =========================
# Closed trades
# =========================

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
