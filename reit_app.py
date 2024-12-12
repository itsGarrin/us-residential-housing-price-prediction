import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'])
    return data


# Main app
def main():
    st.title("REIT Price Prediction Dashboard")
    st.sidebar.header("Options")
    st.sidebar.info("Select the REIT(s) to analyze and compare their actual vs. predicted prices.")

    # File upload or default file
    data = load_data("data/REIT_predictions.csv")

    # Dropdown for selecting REIT(s) to display
    reit_choices = ["AVB", "EQR", "ESS", "INVH"]
    selected_reit = st.sidebar.multiselect("Select REIT(s)", reit_choices, default=reit_choices)

    # Filter data for selected REITs
    if not selected_reit:
        st.warning("Please select at least one REIT.")
        return

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(data["date"].min(), data["date"].max()),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )
    filtered_data = data[
        (data["date"] >= pd.to_datetime(date_range[0])) & (data["date"] <= pd.to_datetime(date_range[1]))]

    # Display data summary
    display_data_summary(filtered_data, selected_reit)

    # Display REIT plots for selected REITs
    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        adj_close_1week_col = f"{reit}_adj_close_1week"
        pred_col = f"{reit}_pred"

        st.subheader(f"{reit} Actual vs Predicted Prices")
        plot_actual_vs_predicted(filtered_data, adj_close_col, adj_close_1week_col, pred_col)
        plot_residual_distribution(filtered_data, adj_close_col, pred_col)

        # Model Evaluation Metrics
        mae, mse, rmse, r2 = calculate_metrics(filtered_data, adj_close_col, pred_col)
        st.write(f"**Evaluation Metrics for {reit}:**")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R²: {r2:.2f}")

    avg_mae, avg_mse, avg_rmse, avg_r2 = calculate_aggregate_metrics(filtered_data, selected_reit)
    st.write("### Aggregate Metrics Across Selected REITs")
    st.write(f"MAE: {avg_mae:.2f}, MSE: {avg_mse:.2f}, RMSE: {avg_rmse:.2f}, R²: {avg_r2:.2f}")

    with st.expander("Show Filtered Data"):
        st.write(filtered_data.head(10))

    # Download button for predictions
    st.sidebar.subheader("Download Predictions")
    if st.sidebar.button("Download Predictions as CSV"):
        predictions_data = filtered_data[["date"] + [f"{reit}_pred" for reit in selected_reit]]
        predictions_data.to_csv("predictions.csv", index=False)
        st.sidebar.success("Predictions CSV ready to download!")

# Data Summary Section
def display_data_summary(data, selected_reit):
    st.write("### Dataset Summary")
    st.write(f"**Total Rows:** {data.shape[0]}")
    st.write(f"**Total Columns:** {data.shape[1]}")
    st.write(f"**Selected REITs:** {', '.join(selected_reit)}")
    st.write(f"**Date Range:** {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    st.write(" ")


def plot_actual_vs_predicted(data, adj_close_col, adj_close_1week_col, pred_col):
    melted_data = data.melt(
        id_vars="date",
        value_vars=[adj_close_1week_col, pred_col],
        var_name="Price Type",
        value_name="Price"
    )
    fig = px.line(
        melted_data,
        x="date",
        y="Price",
        color="Price Type",
        labels={"date": "Date", "Price": "Price ($)"},
        title=f"Actual vs Predicted Prices ({adj_close_col})",
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig)

def plot_residual_distribution(data, actual_col, pred_col):
    residuals = data[actual_col] - data[pred_col]
    fig = px.histogram(
        residuals,
        nbins=50,
        title="Residual Distribution",
        labels={"value": "Residuals"},
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)


# Function to calculate model evaluation metrics
def calculate_metrics(data, actual_col, pred_col):
    actual = data[actual_col]
    predicted = data[pred_col]

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(actual - predicted))

    # Mean Squared Error (MSE)
    mse = np.mean((actual - predicted) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R-squared (R²)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return mae, mse, rmse, r2

def calculate_aggregate_metrics(data, selected_reit):
    aggregate_metrics = {"MAE": [], "MSE": [], "RMSE": [], "R²": []}
    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"
        mae, mse, rmse, r2 = calculate_metrics(data, adj_close_col, pred_col)
        aggregate_metrics["MAE"].append(mae)
        aggregate_metrics["MSE"].append(mse)
        aggregate_metrics["RMSE"].append(rmse)
        aggregate_metrics["R²"].append(r2)

    # Calculate mean metrics
    avg_mae = np.mean(aggregate_metrics["MAE"])
    avg_mse = np.mean(aggregate_metrics["MSE"])
    avg_rmse = np.mean(aggregate_metrics["RMSE"])
    avg_r2 = np.mean(aggregate_metrics["R²"])
    return avg_mae, avg_mse, avg_rmse, avg_r2


if __name__ == "__main__":
    main()
