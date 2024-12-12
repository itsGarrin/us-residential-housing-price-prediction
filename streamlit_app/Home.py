import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# Load data for metrics
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["date"])
    return data


def home_page():
    st.title("📊 REIT Price Prediction Dashboard")
    st.markdown("""
    Welcome to the **REIT Price Prediction Dashboard**, where you can explore predictions, analyze key metrics, and gain insights into real estate investment trust (REIT) performance over time.  
    Use the navigation on the left to explore different sections of the dashboard.
    """)

    # Load the data
    file_path = "data/REIT_predictions.csv"
    data = load_data(file_path)

    # Key metrics section (vertical layout)
    st.subheader("Key Metrics")
    latest_date = data["date"].max()
    earliest_date = data["date"].min()

    # Calculate metrics
    reit_choices = ["AVB", "EQR", "ESS", "INVH"]
    metrics = []
    for reit in reit_choices:
        actual_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"
        residuals = data[actual_col] - data[pred_col]
        mae = np.mean(np.abs(residuals))
        metrics.append((reit, mae, np.std(residuals), residuals.idxmax()))

    # Find the most accurate REIT and largest error
    most_accurate_reit = min(metrics, key=lambda x: x[1])  # REIT with lowest MAE
    largest_error_index = max(metrics,
                              key=lambda x: data.loc[x[3], f"{x[0]}_adj_close"] - data.loc[x[3], f"{x[0]}_pred"])

    # Display the metrics in a vertical layout
    st.write(f"**📅 Date Range:** {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
    st.write(f"**✅ Most Accurate REIT:** {most_accurate_reit[0]} (MAE: {most_accurate_reit[1]:.2f})")
    largest_error_date = data.loc[largest_error_index[3], "date"].strftime('%Y-%m-%d')
    largest_error_value = data.loc[largest_error_index[3], f"{largest_error_index[0]}_adj_close"] - data.loc[
        largest_error_index[3], f"{largest_error_index[0]}_pred"]
    st.write(
        f"**❗ Largest Error:** {largest_error_index[0]} on {largest_error_date} (Error: {largest_error_value:.2f})")

    # Highlight chart: Predicted vs Actual Prices
    st.subheader("📉 REIT Predicted vs Actual Prices")
    melted_data = data.melt(
        id_vars="date",
        value_vars=[f"{reit}_adj_close" for reit in reit_choices] + [f"{reit}_pred" for reit in reit_choices],
        var_name="Type",
        value_name="Price"
    )
    melted_data["Type"] = melted_data["Type"].str.replace("_adj_close", " (Actual)").str.replace("_pred",
                                                                                                 " (Predicted)")

    fig = px.line(
        melted_data,
        x="date",
        y="Price",
        color="Type",
        title="Predicted vs Actual Prices Over Time",
        labels={"date": "Date", "Price": "Price ($)"},
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    home_page()
