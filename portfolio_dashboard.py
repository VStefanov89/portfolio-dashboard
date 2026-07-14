import streamlit as st
import boto3
import pandas as pd
import numpy as np

from io import BytesIO


# ============================================================
# Page configuration
# Must appear before other Streamlit UI commands
# ============================================================

st.set_page_config(
    page_title="Portfolio Daily Update",
    layout="wide",
)


# ============================================================
# AWS / S3 configuration
# ============================================================

AWS_SECRETS = st.secrets["aws"]

BUCKET = AWS_SECRETS["bucket_name"]

LIVE_PORTFOLIO_KEY = "dashboard/live_signals_portfolio.csv"
CLOSED_TRADES_KEY = "dashboard/close_signals_historycally.csv"
DAILY_PNL_KEY = "dashboard/daily_pnl.csv"


def create_s3_client():
    """
    Create an authenticated S3 client using Streamlit secrets.
    """
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_SECRETS["access_key_id"],
        aws_secret_access_key=AWS_SECRETS["secret_access_key"],
        region_name=AWS_SECRETS["region"],
    )


@st.cache_data(
    ttl=55,
    show_spinner=False,
)
def read_csv_from_s3(key):
    """
    Download and parse a CSV file from S3.

    The cache expires after 55 seconds. The dashboard fragment
    reruns every 60 seconds, so each fragment run should see the
    newest S3 version.
    """
    s3 = create_s3_client()

    response = s3.get_object(
        Bucket=BUCKET,
        Key=key,
    )

    raw_bytes = response["Body"].read()

    return pd.read_csv(
        BytesIO(raw_bytes)
    )


@st.cache_data(
    ttl=55,
    show_spinner=False,
)
def get_s3_last_modified(key):
    """
    Return the LastModified timestamp for an S3 object.
    """
    s3 = create_s3_client()

    response = s3.head_object(
        Bucket=BUCKET,
        Key=key,
    )

    return response["LastModified"]


# ============================================================
# Helper functions
# ============================================================

def max_drawdown(equity):
    equity = pd.Series(equity).astype(float)

    running_max = equity.cummax()

    dd = equity / running_max - 1
    dd_dollar = equity - running_max

    return (
        dd.min(),
        dd,
        dd_dollar.min(),
        dd_dollar,
    )


def format_money(x):
    if pd.isna(x):
        return ""

    return f"${x:,.0f}"


def format_pct(x):
    if pd.isna(x):
        return ""

    return f"{x:.2%}"


def format_number(x):
    if pd.isna(x):
        return ""

    return f"{x:,.2f}"


def format_int(x):
    if pd.isna(x):
        return ""

    return f"{int(x):,}"


@st.cache_data(
    show_spinner=False,
)
def strategy_statistics_from_df(
    df,
    starting_capital=3_000_000,
    date_col="date",
    pnl_col="daily_pnl",
    bp_col="total_bp",
    n_mc_paths=5_000,
    random_state=42,
):
    """
    Calculate strategy statistics and Monte Carlo summaries.

    This function is cached. It will only rerun when its input
    dataframe or arguments change.
    """
    df = df.copy()

    if df.empty:
        empty_series = pd.Series(dtype=float)
        empty_df = pd.DataFrame()

        return (
            empty_df,
            empty_series,
            empty_df,
            empty_df,
            empty_series,
            empty_df,
            empty_df,
            empty_df,
        )

    required_cols = {
        date_col,
        pnl_col,
        bp_col,
    }

    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise KeyError(
            f"Missing columns for strategy statistics: "
            f"{sorted(missing_cols)}"
        )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        format="mixed",
    ).dt.normalize()

    df[pnl_col] = pd.to_numeric(
        df[pnl_col],
        errors="coerce",
    ).fillna(0)

    df[bp_col] = pd.to_numeric(
        df[bp_col],
        errors="coerce",
    ).fillna(0)

    df = df.dropna(
        subset=[date_col]
    )

    daily = (
        df.groupby(
            date_col,
            as_index=False,
        )
        .agg(
            daily_pnl=(pnl_col, "sum"),
            total_bp=(bp_col, "sum"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if daily.empty:
        empty_series = pd.Series(dtype=float)
        empty_df = pd.DataFrame()

        return (
            empty_df,
            empty_series,
            empty_df,
            empty_df,
            empty_series,
            empty_df,
            empty_df,
            empty_df,
        )

    daily["return_on_capital"] = (
        daily["daily_pnl"] / starting_capital
    )

    daily["return_on_deployed_bp"] = (
        daily["daily_pnl"]
        / daily["total_bp"].replace(0, np.nan)
    )

    daily["bp_utilization"] = (
        daily["total_bp"] / starting_capital
    )

    daily["equity"] = (
        starting_capital
        + daily["daily_pnl"].cumsum()
    )

    n_days = len(daily)

    years = (
        n_days / 252
        if n_days > 0
        else np.nan
    )

    total_pnl = daily["daily_pnl"].sum()

    ending_equity = daily["equity"].iloc[-1]

    total_return = (
        ending_equity / starting_capital - 1
    )

    cagr = (
        (ending_equity / starting_capital) ** (1 / years) - 1
        if years > 0
        and ending_equity > 0
        else np.nan
    )

    ret_std = daily["return_on_capital"].std(
        ddof=1
    )

    ann_vol = (
        ret_std * np.sqrt(252)
        if not pd.isna(ret_std)
        else np.nan
    )

    sharpe = (
        daily["return_on_capital"].mean()
        / ret_std
        * np.sqrt(252)
        if ret_std != 0
        and not pd.isna(ret_std)
        else np.nan
    )

    (
        max_dd,
        dd_series,
        max_dd_dollar,
        dd_dollar_series,
    ) = max_drawdown(
        daily["equity"]
    )

    daily["drawdown"] = dd_series
    daily["drawdown_dollar"] = dd_dollar_series

    gross_profit = daily.loc[
        daily["daily_pnl"] > 0,
        "daily_pnl",
    ].sum()

    gross_loss = daily.loc[
        daily["daily_pnl"] < 0,
        "daily_pnl",
    ].sum()

    profit_factor = (
        gross_profit / abs(gross_loss)
        if gross_loss != 0
        else np.inf
    )

    summary = pd.Series(
        {
            "n_days": n_days,
            "starting_capital": starting_capital,
            "ending_equity": ending_equity,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "max_drawdown_dollar": max_dd_dollar,
            "win_rate_days": (
                daily["daily_pnl"] > 0
            ).mean(),
            "profit_factor": profit_factor,
            "avg_daily_pnl": daily[
                "daily_pnl"
            ].mean(),
            "median_daily_pnl": daily[
                "daily_pnl"
            ].median(),
            "best_day": daily[
                "daily_pnl"
            ].max(),
            "worst_day": daily[
                "daily_pnl"
            ].min(),
            "avg_bp_utilization": daily[
                "bp_utilization"
            ].mean(),
            "median_bp_utilization": daily[
                "bp_utilization"
            ].median(),
            "max_bp_utilization": daily[
                "bp_utilization"
            ].max(),
            "avg_deployed_return": daily[
                "return_on_deployed_bp"
            ].mean(),
        }
    )

    rng = np.random.default_rng(
        random_state
    )

    daily_returns = daily[
        "return_on_capital"
    ].to_numpy()

    mc_rows = []

    if n_days > 1:
        for _ in range(n_mc_paths):
            sampled_returns = rng.choice(
                daily_returns,
                size=n_days,
                replace=True,
            )

            path_equity = (
                starting_capital
                * np.cumprod(
                    1 + sampled_returns
                )
            )

            path_total_return = (
                path_equity[-1]
                / starting_capital
                - 1
            )

            path_cagr = (
                (
                    path_equity[-1]
                    / starting_capital
                ) ** (1 / years)
                - 1
                if years > 0
                and path_equity[-1] > 0
                else np.nan
            )

            (
                path_max_dd,
                _,
                path_max_dd_dollar,
                _,
            ) = max_drawdown(
                path_equity
            )

            path_std = sampled_returns.std(
                ddof=1
            )

            path_sharpe = (
                sampled_returns.mean()
                / path_std
                * np.sqrt(252)
                if path_std != 0
                and not pd.isna(path_std)
                else np.nan
            )

            mc_rows.append(
                {
                    "final_equity": path_equity[-1],
                    "total_return": path_total_return,
                    "cagr": path_cagr,
                    "max_drawdown": path_max_dd,
                    "max_drawdown_dollar": (
                        path_max_dd_dollar
                    ),
                    "sharpe": path_sharpe,
                }
            )

    mc = pd.DataFrame(
        mc_rows
    )

    if not mc.empty:
        mc_summary = pd.DataFrame(
            {
                "p5": mc.quantile(0.05),
                "median": mc.quantile(0.50),
                "p95": mc.quantile(0.95),
            }
        )

        risk_probs = pd.Series(
            {
                "prob_losing_money": (
                    mc["total_return"] < 0
                ).mean(),
                "prob_drawdown_worse_than_5pct": (
                    mc["max_drawdown"] < -0.05
                ).mean(),
                "prob_drawdown_worse_than_10pct": (
                    mc["max_drawdown"] < -0.10
                ).mean(),
                "prob_return_above_20pct": (
                    mc["total_return"] > 0.20
                ).mean(),
                "prob_return_above_30pct": (
                    mc["total_return"] > 0.30
                ).mean(),
            }
        )

    else:
        mc_summary = pd.DataFrame()
        risk_probs = pd.Series(
            dtype=float
        )

    friendly_summary = pd.DataFrame(
        {
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
                "Average Deployed Return",
            ],
            "Value": [
                format_int(
                    summary["n_days"]
                ),
                format_money(
                    summary["starting_capital"]
                ),
                format_money(
                    summary["ending_equity"]
                ),
                format_money(
                    summary["total_pnl"]
                ),
                format_pct(
                    summary["total_return"]
                ),
                format_pct(
                    summary["cagr"]
                ),
                format_pct(
                    summary[
                        "annualized_volatility"
                    ]
                ),
                format_number(
                    summary["sharpe"]
                ),
                format_pct(
                    summary["max_drawdown"]
                ),
                format_money(
                    summary[
                        "max_drawdown_dollar"
                    ]
                ),
                format_pct(
                    summary["win_rate_days"]
                ),
                (
                    format_number(
                        summary["profit_factor"]
                    )
                    + "x"
                ),
                format_money(
                    summary["avg_daily_pnl"]
                ),
                format_money(
                    summary["median_daily_pnl"]
                ),
                format_money(
                    summary["best_day"]
                ),
                format_money(
                    summary["worst_day"]
                ),
                format_pct(
                    summary[
                        "avg_bp_utilization"
                    ]
                ),
                format_pct(
                    summary[
                        "median_bp_utilization"
                    ]
                ),
                format_pct(
                    summary[
                        "max_bp_utilization"
                    ]
                ),
                format_pct(
                    summary[
                        "avg_deployed_return"
                    ]
                ),
            ],
        }
    )

    if not mc_summary.empty:
        friendly_mc_summary = (
            mc_summary
            .copy()
            .astype(object)
        )

        for row in friendly_mc_summary.index:
            for col in friendly_mc_summary.columns:
                value = mc_summary.loc[
                    row,
                    col,
                ]

                if pd.isna(value):
                    formatted_value = ""

                elif row in [
                    "final_equity",
                    "max_drawdown_dollar",
                ]:
                    formatted_value = format_money(
                        value
                    )

                elif row in [
                    "total_return",
                    "cagr",
                    "max_drawdown",
                ]:
                    formatted_value = format_pct(
                        value
                    )

                elif row == "sharpe":
                    formatted_value = format_number(
                        value
                    )

                else:
                    formatted_value = format_number(
                        value
                    )

                friendly_mc_summary.loc[
                    row,
                    col,
                ] = formatted_value

        friendly_mc_summary = (
            friendly_mc_summary
            .reset_index()
            .rename(
                columns={
                    "index": "Metric",
                    "p5": "P5",
                    "median": "Median",
                    "p95": "P95",
                }
            )
        )

        friendly_risk_probs = pd.DataFrame(
            {
                "Metric": [
                    "Probability of Losing Money",
                    "Probability DD Worse Than -5%",
                    "Probability DD Worse Than -10%",
                    "Probability Return Above 20%",
                    "Probability Return Above 30%",
                ],
                "Value": [
                    format_pct(
                        risk_probs[
                            "prob_losing_money"
                        ]
                    ),
                    format_pct(
                        risk_probs[
                            "prob_drawdown_worse_than_5pct"
                        ]
                    ),
                    format_pct(
                        risk_probs[
                            "prob_drawdown_worse_than_10pct"
                        ]
                    ),
                    format_pct(
                        risk_probs[
                            "prob_return_above_20pct"
                        ]
                    ),
                    format_pct(
                        risk_probs[
                            "prob_return_above_30pct"
                        ]
                    ),
                ],
            }
        )

    else:
        friendly_mc_summary = pd.DataFrame(
            columns=[
                "Metric",
                "P5",
                "Median",
                "P95",
            ]
        )

        friendly_risk_probs = pd.DataFrame(
            columns=[
                "Metric",
                "Value",
            ]
        )

    return (
        daily,
        summary,
        mc,
        mc_summary,
        risk_probs,
        friendly_summary,
        friendly_mc_summary,
        friendly_risk_probs,
    )


# ============================================================
# Auto-refreshing dashboard
# ============================================================

@st.fragment(
    run_every="60s",
)
def render_live_dashboard():
    """
    Reload S3 data and redraw the dashboard every 60 seconds
    while a user has the page open.
    """

    # ========================================================
    # Load data
    # ========================================================

    try:
        live_df = read_csv_from_s3(
            LIVE_PORTFOLIO_KEY
        )

        closed_df = read_csv_from_s3(
            CLOSED_TRADES_KEY
        )

        daily_df = read_csv_from_s3(
            DAILY_PNL_KEY
        )

        live_last_modified = get_s3_last_modified(
            LIVE_PORTFOLIO_KEY
        )

        closed_last_modified = get_s3_last_modified(
            CLOSED_TRADES_KEY
        )

        daily_last_modified = get_s3_last_modified(
            DAILY_PNL_KEY
        )

    except Exception as exc:
        st.error(
            "Could not retrieve the latest portfolio "
            f"data from S3: {exc}"
        )
        return

    # ========================================================
    # Validate loaded data
    # ========================================================

    if daily_df.empty:
        st.warning(
            "daily_pnl.csv is empty."
        )
        return

    if live_df.empty:
        st.warning(
            "live_signals_portfolio.csv is empty."
        )

    if closed_df.empty:
        st.info(
            "close_signals_historycally.csv is empty."
        )

    # ========================================================
    # Convert dates
    # ========================================================

    for dataframe in [
        live_df,
        closed_df,
        daily_df,
    ]:
        if "date" in dataframe.columns:
            dataframe["date"] = pd.to_datetime(
                dataframe["date"],
                errors="coerce",
                format="mixed",
            ).dt.normalize()

    if "date" not in daily_df.columns:
        st.error(
            "The daily PnL file does not contain "
            "a 'date' column."
        )
        return

    daily_df = daily_df.dropna(
        subset=["date"]
    )

    if daily_df.empty:
        st.warning(
            "No valid dates were found in daily_pnl.csv."
        )
        return

    # ========================================================
    # Prepare daily summary
    # ========================================================

    daily_df = daily_df.copy()

    daily_df = daily_df.rename(
        columns={
            "n_signal_rows_x": "open_triplets",
            "n_signal_rows_y": "closed_triplets",
            "total_bp_y": "realized_bp",
            "cum_pnl": "total_pnl",
        }
    )

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
        "spy_equity",
    ]

    existing_needed_cols = [
        column
        for column in needed_cols
        if column in daily_df.columns
    ]

    summary_df = daily_df[
        existing_needed_cols
    ].copy()

    summary_df = (
        summary_df
        .sort_values("date")
        .reset_index(drop=True)
    )

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
        "spy_equity",
    ]

    for column in numeric_cols:
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(
                summary_df[column],
                errors="coerce",
            ).fillna(0)

    required_summary_defaults = {
        "unrealized_daily_pnl": 0.0,
        "open_triplets": 0,
        "total_bp": 0.0,
        "realised_daily_pnl": 0.0,
        "closed_triplets": 0,
        "realized_bp": 0.0,
        "daily_pnl": 0.0,
    }

    for column, default_value in (
        required_summary_defaults.items()
    ):
        if column not in summary_df.columns:
            summary_df[column] = default_value

    summary_df["open_triplets"] = (
        summary_df["open_triplets"]
        .fillna(0)
        .astype(int)
    )

    summary_df["closed_triplets"] = (
        summary_df["closed_triplets"]
        .fillna(0)
        .astype(int)
    )

    if "total_pnl" not in summary_df.columns:
        summary_df["total_pnl"] = (
            summary_df["daily_pnl"].cumsum()
        )

    if summary_df.empty:
        st.warning(
            "No rows are available in the daily summary."
        )
        return

    # ========================================================
    # Strategy statistics
    # ========================================================

    starting_capital = 3_000_000

    (
        stats_daily,
        stats_summary,
        mc,
        mc_summary,
        risk_probs,
        friendly_summary,
        friendly_mc_summary,
        friendly_risk_probs,
    ) = strategy_statistics_from_df(
        summary_df[
            [
                "date",
                "daily_pnl",
                "total_bp",
            ]
        ],
        starting_capital=starting_capital,
        n_mc_paths=5_000,
    )

    # ========================================================
    # Latest data
    # ========================================================

    latest_date = summary_df[
        "date"
    ].max()

    latest_rows = summary_df[
        summary_df["date"] == latest_date
    ]

    if latest_rows.empty:
        st.warning(
            "No summary row was found for the latest date."
        )
        return

    latest_summary = latest_rows.iloc[0]

    if (
        not live_df.empty
        and "date" in live_df.columns
    ):
        latest_portfolio = live_df[
            live_df["date"] == latest_date
        ].copy()

    else:
        latest_portfolio = pd.DataFrame()

    open_triplets = int(
        latest_summary.get(
            "open_triplets",
            (
                latest_portfolio["uuid"].nunique()
                if (
                    not latest_portfolio.empty
                    and "uuid"
                    in latest_portfolio.columns
                )
                else 0
            ),
        )
    )

    # ========================================================
    # Header
    # ========================================================

    st.title(
        "Daily Portfolio Update"
    )

    st.subheader(
        f"Date: {latest_date.date()}"
    )

    local_daily_timestamp = (
        daily_last_modified.astimezone()
    )

    st.caption(
        "Latest dashboard data uploaded to S3: "
        f"{local_daily_timestamp:%Y-%m-%d %H:%M:%S %Z}"
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
        st.columns(5)
    )

    metric_col1.metric(
        "Total PnL",
        f"${latest_summary.get('total_pnl', 0):,.2f}",
    )

    metric_col2.metric(
        "Daily PnL",
        f"${latest_summary.get('daily_pnl', 0):,.2f}",
    )

    metric_col3.metric(
        "Unrealized Daily PnL",
        (
            f"${latest_summary.get(
                'unrealized_daily_pnl',
                0
            ):,.2f}"
        ),
    )

    metric_col4.metric(
        "Total BP",
        f"${latest_summary.get('total_bp', 0):,.0f}",
    )

    metric_col5.metric(
        "Open Triplets",
        f"{open_triplets:,}",
    )

    st.divider()

    # ========================================================
    # Data freshness panel
    # ========================================================

    with st.expander(
        "Data freshness",
        expanded=False,
    ):
        freshness_df = pd.DataFrame(
            {
                "Dataset": [
                    "Live Portfolio",
                    "Closed Trades",
                    "Daily PnL",
                ],
                "S3 Key": [
                    LIVE_PORTFOLIO_KEY,
                    CLOSED_TRADES_KEY,
                    DAILY_PNL_KEY,
                ],
                "Last Modified": [
                    live_last_modified.astimezone(),
                    closed_last_modified.astimezone(),
                    daily_last_modified.astimezone(),
                ],
                "Rows Loaded": [
                    len(live_df),
                    len(closed_df),
                    len(daily_df),
                ],
            }
        )

        st.dataframe(
            freshness_df,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "The open page checks S3 approximately "
            "every 60 seconds."
        )

    # ========================================================
    # Total PnL graph
    # ========================================================

    pnl_chart = summary_df[
        [
            "date",
            "total_pnl",
        ]
    ].copy()

    pnl_chart["date"] = (
        pnl_chart["date"].dt.date
    )

    st.subheader(
        "Total PnL"
    )

    st.line_chart(
        pnl_chart.set_index(
            "date"
        )
    )

    # ========================================================
    # Strategy vs SPY graph
    # ========================================================

    if {
        "strategy_equity",
        "spy_equity",
    }.issubset(summary_df.columns):

        comparison_df = summary_df[
            [
                "date",
                "strategy_equity",
                "spy_equity",
            ]
        ].copy()

        comparison_df["date"] = (
            comparison_df["date"].dt.date
        )

        comparison_df = (
            comparison_df
            .set_index("date")
            .rename(
                columns={
                    "strategy_equity": "Strategy",
                    "spy_equity": "SPY",
                }
            )
        )

        st.subheader(
            "Strategy vs SPY"
        )

        st.line_chart(
            comparison_df
        )

    else:
        st.info(
            "strategy_equity and/or spy_equity "
            "columns were not found in daily_pnl.csv."
        )

    # ========================================================
    # Strategy statistics
    # ========================================================

    st.subheader(
        "Strategy Statistics"
    )

    st.dataframe(
        friendly_summary,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "Monte Carlo Summary"
    )

    st.dataframe(
        friendly_mc_summary,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "Risk Probabilities"
    )

    st.dataframe(
        friendly_risk_probs,
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # Daily summary
    # ========================================================

    st.subheader(
        "Daily Summary History"
    )

    st.dataframe(
        summary_df.sort_values(
            "date",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # Current portfolio
    # ========================================================

    st.subheader(
        "Current Portfolio"
    )

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
        "n_earnings_next_5d",
    ]

    available_portfolio_cols = [
        column
        for column in show_cols
        if column in latest_portfolio.columns
    ]

    if (
        not latest_portfolio.empty
        and available_portfolio_cols
    ):
        latest_display = latest_portfolio[
            available_portfolio_cols
        ].copy()

        if (
            "calibrated_tail_p"
            in latest_display.columns
        ):
            latest_display = (
                latest_display
                .sort_values(
                    "calibrated_tail_p",
                    ascending=False,
                )
            )

        st.dataframe(
            latest_display,
            width="stretch",
            hide_index=True,
        )

    else:
        st.info(
            "No current portfolio positions were "
            "found for the latest date."
        )

    # ========================================================
    # Closed trades
    # ========================================================

    st.subheader(
        "Closed Trades"
    )

    if not closed_df.empty:
        closed_display = (
            closed_df.copy()
        )

        closed_cols = [
            column
            for column in show_cols
            if column in closed_display.columns
        ]

        if closed_cols:
            closed_display = closed_display[
                closed_cols
            ]

            if (
                "date"
                in closed_display.columns
            ):
                closed_display = (
                    closed_display
                    .sort_values(
                        "date",
                        ascending=False,
                    )
                )

            st.dataframe(
                closed_display,
                width="stretch",
                hide_index=True,
            )

        else:
            st.info(
                "No matching closed trade "
                "columns were found."
            )

    else:
        st.info(
            "There are no closed trades to display."
        )

    # ========================================================
    # Risk alerts
    # ========================================================

    st.subheader(
        "Risk Alerts"
    )

    if (
        not latest_portfolio.empty
        and "calibrated_tail_p"
        in latest_portfolio.columns
    ):
        tail_probability = pd.to_numeric(
            latest_portfolio[
                "calibrated_tail_p"
            ],
            errors="coerce",
        )

        risk_df = latest_portfolio[
            tail_probability >= 0.96
        ].copy()

        if not risk_df.empty:
            st.warning(
                f"{len(risk_df):,} positions have "
                "tail probability above 0.96."
            )

            risk_cols = [
                column
                for column in show_cols
                if column in risk_df.columns
            ]

            risk_display = risk_df[
                risk_cols
            ].copy()

            if (
                "calibrated_tail_p"
                in risk_display.columns
            ):
                risk_display = (
                    risk_display
                    .sort_values(
                        "calibrated_tail_p",
                        ascending=False,
                    )
                )

            st.dataframe(
                risk_display,
                width="stretch",
                hide_index=True,
            )

        else:
            st.success(
                "No high tail-risk positions today."
            )

    else:
        st.info(
            "Column calibrated_tail_p was not found "
            "in the latest portfolio."
        )


# ============================================================
# Start dashboard
# ============================================================

render_live_dashboard()
