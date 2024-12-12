import os

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=["date"])


def simulate_strategy(data, strategy, num_reits, investment):
    # Identify REITs
    reits = [col.split("_")[0] for col in data.columns if "_pred_returns" in col]

    # Sort REITs based on predicted returns for the strategy
    weights = []
    for _, row in data.iterrows():
        returns = {reit: row[f"{reit}_pred_returns"] for reit in reits}
        sorted_reits = sorted(returns.items(), key=lambda x: x[1], reverse=(strategy == "Long Best"))

        if strategy in ["Long Best", "Short Worst"]:
            selected_reits = sorted_reits[:num_reits] if strategy == "Long Best" else sorted_reits[-num_reits:]
            weight = {reit: 1 / num_reits for reit, _ in selected_reits}
        else:
            longs = sorted_reits[:num_reits]
            shorts = sorted_reits[-num_reits:]
            weight = {reit: (1 / num_reits) if reit in dict(longs) else -(1 / num_reits) for reit in reits}

        weights.append(weight)

    # Calculate portfolio returns
    portfolio_returns = []
    for i, row in data.iterrows():
        daily_return = sum(
            row[f"{reit}_returns"] * weights[i].get(reit, 0) for reit in reits
        )
        portfolio_returns.append(daily_return)

    # Add portfolio returns and cumulative value to the DataFrame
    data["Portfolio_Returns"] = portfolio_returns
    data["Portfolio_Cumulative"] = (1 + pd.Series(portfolio_returns)).cumprod() * investment

    # S&P 500 cumulative value
    data["SPY_Cumulative"] = (1 + data["SPY_return_1week"]).cumprod() * investment

    return data

def calculate_performance_metrics(data, portfolio_col, benchmark_col, risk_free_rate=0.02):
    # Cumulative returns
    portfolio_cumulative_return = (data[portfolio_col].iloc[-1] / data[portfolio_col].iloc[0]) - 1
    benchmark_cumulative_return = (data[benchmark_col].iloc[-1] / data[benchmark_col].iloc[0]) - 1

    # Annualized returns (approximation using weekly data)
    num_weeks = len(data)
    portfolio_annualized_return = (1 + portfolio_cumulative_return) ** (52 / num_weeks) - 1
    benchmark_annualized_return = (1 + benchmark_cumulative_return) ** (52 / num_weeks) - 1

    # Volatility (standard deviation of weekly returns)
    portfolio_volatility = data["Portfolio_Returns"].std() * np.sqrt(52)
    benchmark_volatility = data["SPY_return_1week"].std() * np.sqrt(52)

    # Sharpe Ratio
    portfolio_sharpe = (portfolio_annualized_return - risk_free_rate) / portfolio_volatility
    benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility

    # Maximum Drawdown
    portfolio_drawdown = data[portfolio_col].div(data[portfolio_col].cummax()).min() - 1
    benchmark_drawdown = data[benchmark_col].div(data[benchmark_col].cummax()).min() - 1

    return {
        "Cumulative Return": [portfolio_cumulative_return, benchmark_cumulative_return],
        "Annualized Return": [portfolio_annualized_return, benchmark_annualized_return],
        "Volatility": [portfolio_volatility, benchmark_volatility],
        "Sharpe Ratio": [portfolio_sharpe, benchmark_sharpe],
        "Max Drawdown": [portfolio_drawdown, benchmark_drawdown],
    }

def main():
    st.title("📈 Backtesting & Trading Strategies")

    # Load data
    data = load_data("data/REIT_predictions.csv")

    # Sidebar options
    strategies = ["Long Best", "Short Worst", "Long/Short"]
    strategy = st.sidebar.selectbox("Select Trading Strategy", strategies)
    num_reits = st.sidebar.slider("Number of REITs", min_value=1, max_value=4, value=2)
    investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, step=500, value=10000)
    # Date Range Selector
    st.sidebar.subheader("Select Date Range")
    min_date = pd.to_datetime(data["date"].min())
    max_date = pd.to_datetime(data["date"].max())
    default_start = min_date
    default_end = max_date

    start_date = st.sidebar.date_input("Start Date", default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", default_end, min_value=min_date, max_value=max_date)

    # Ensure valid date range
    if start_date > end_date:
        st.error("Start Date must be before End Date.")
    else:
        # Filter data based on date range
        filtered_data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))]

        # Simulate strategy
        simulated_data = simulate_strategy(filtered_data, strategy, num_reits, investment)

        # Plot cumulative returns
        fig = px.line(
            simulated_data,
            x="date",
            y=["Portfolio_Cumulative", "SPY_Cumulative"],
            labels={"value": "Cumulative Value ($)", "variable": "Investment"},
            title=f"Cumulative Returns ({start_date} to {end_date})",
        )
        st.plotly_chart(fig)

        # Add performance metrics
        metrics = calculate_performance_metrics(simulated_data, "Portfolio_Cumulative", "SPY_Cumulative")
        metrics_df = pd.DataFrame(metrics, index=["Portfolio", "S&P 500"])

        st.subheader(f"Performance Metrics ({start_date} to {end_date})")
        st.table(metrics_df.style.format("{:.2%}" if metrics_df.columns.name != "Sharpe Ratio" else "{:.2f}"))

    # Download results
    st.subheader("Download Simulated Data")
    st.download_button(
        label="Download as CSV",
        data=simulated_data.to_csv(index=False),
        file_name="simulated_strategy.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
