import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'])
    return data


# Main app
def main():
    st.title("REIT Price Prediction Dashboard")
    st.sidebar.header("Options")

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

    # Show filtered data
    st.subheader("Filtered Data")
    st.write(filtered_data.head(10))

    # Display REIT plots for selected REITs
    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        adj_close_1week_col = f"{reit}_adj_close_1week"
        pred_col = f"{reit}_pred"

        st.subheader(f"{reit} Actual vs Predicted Prices")
        plot_actual_vs_predicted(filtered_data, adj_close_col, adj_close_1week_col, pred_col)

        # Model Evaluation Metrics
        mae, mse, rmse, r2 = calculate_metrics(filtered_data, adj_close_col, pred_col)
        st.write(f"**Evaluation Metrics for {reit}:**")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R²: {r2:.2f}")

    # Download button for predictions
    st.sidebar.subheader("Download Predictions")
    if st.sidebar.button("Download Predictions as CSV"):
        predictions_data = filtered_data[["date"] + [f"{reit}_pred" for reit in selected_reit]]
        predictions_data.to_csv("predictions.csv", index=False)
        st.sidebar.success("Predictions CSV ready to download!")


# Function to plot actual vs predicted prices
def plot_actual_vs_predicted(data, adj_close_col, adj_close_3month_col, pred_col):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["date"], data[adj_close_col], label="Actual (Current)", color="blue")
    ax.plot(data["date"], data[adj_close_3month_col], label="Actual (3-Month Ahead)", color="green")
    ax.plot(data["date"], data[pred_col], label="Predicted (3-Month)", color="orange", linestyle="--")
    ax.set_title(f"{adj_close_col} Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


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


if __name__ == "__main__":
    main()
