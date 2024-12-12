import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['date'])

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
    aggregate_metrics = {"MAE": [], "RMSE": [], "R²": []}
    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"
        mae, mse, rmse, r2 = calculate_metrics(data, adj_close_col, pred_col)
        aggregate_metrics["MAE"].append(mae)
        aggregate_metrics["RMSE"].append(rmse)
        aggregate_metrics["R²"].append(r2)

    # Calculate mean metrics
    avg_mae = np.mean(aggregate_metrics["MAE"])
    avg_rmse = np.mean(aggregate_metrics["RMSE"])
    avg_r2 = np.mean(aggregate_metrics["R²"])
    return avg_mae, avg_rmse, avg_r2, aggregate_metrics

def main():
    st.title("Predictions & Metrics")

    # Load data
    data = load_data("data/REIT_predictions.csv")

    # Sidebar options
    reit_choices = ["AVB", "EQR", "ESS", "INVH"]
    selected_reit = st.sidebar.multiselect("Select REIT(s)", reit_choices, default=reit_choices)
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(data["date"].min(), data["date"].max()),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )

    if not selected_reit:
        st.warning("Please select at least one REIT.")
        return

    # Filter data
    filtered_data = data[
        (data["date"] >= pd.to_datetime(date_range[0])) & (data["date"] <= pd.to_datetime(date_range[1]))
    ]

    st.subheader("Model Evaluation Metrics")

    # Initialize data for the metrics table and plots
    metrics_table = []
    scatter_plots = []

    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"
        mae, mse, rmse, r2 = calculate_metrics(filtered_data, adj_close_col, pred_col)

        # Append to metrics table
        metrics_table.append({"REIT": reit, "MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2})

        # Create scatter plot for actual vs. predicted
        scatter_fig = px.scatter(
            filtered_data,
            x=adj_close_col,
            y=pred_col,
            labels={adj_close_col: "Actual Price ($)", pred_col: "Predicted Price ($)"},
            title=f"{reit} Actual vs. Predicted Prices",
            trendline="ols",
        )
        scatter_fig.update_traces(marker=dict(size=7, opacity=0.7))
        scatter_plots.append((reit, scatter_fig))

    # Display metrics in a table
    metrics_df = pd.DataFrame(metrics_table)
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.2f}"}))

    # Show scatter plots for each REIT
    st.subheader("Scatter Plots: Actual vs. Predicted Prices")
    for reit, scatter_fig in scatter_plots:
        st.plotly_chart(scatter_fig)

    # Aggregate metrics
    avg_mae, avg_rmse, avg_r2, aggregate_metrics = calculate_aggregate_metrics(filtered_data, selected_reit)

    # Aggregate metrics bar chart
    st.subheader("Aggregate Metrics Across Selected REITs")
    agg_fig = go.Figure(data=[
        go.Bar(name="MAE", x=selected_reit, y=aggregate_metrics["MAE"]),
        go.Bar(name="RMSE", x=selected_reit, y=aggregate_metrics["RMSE"]),
        go.Bar(name="R²", x=selected_reit, y=aggregate_metrics["R²"])
    ])
    agg_fig.update_layout(
        barmode="group",
        title="Aggregate Metrics Comparison",
        xaxis_title="REIT",
        yaxis_title="Metric Value",
    )
    st.plotly_chart(agg_fig)

    st.write(f"**Aggregate Metrics:**")
    st.write(f"MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}, R²: {avg_r2:.2f}")

    # Download predictions
    st.subheader("Download Predictions")
    predictions_data = filtered_data[["date"] + [f"{reit}_pred" for reit in selected_reit]]
    st.download_button(
        label="Download Predictions as CSV",
        data=predictions_data.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()