import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['date'])


@st.cache_resource
def load_model(file_path):
    import pickle
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_feature_importances(model, feature_names):
    """
    Extract feature importances and return a sorted DataFrame.
    Ensure feature_names length matches model.feature_importances_.
    """
    if len(feature_names) != len(model.feature_importances_):
        raise ValueError("Mismatch between number of features and feature importance array length.")

    importances = model.feature_importances_
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)


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
    aggregate_metrics = {"MAE": [], "RMSE": [], "R²": [], "SD": []}
    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"
        mae, mse, rmse, r2 = calculate_metrics(data, adj_close_col, pred_col)
        std_dev = calculate_standard_deviation(data, adj_close_col)
        aggregate_metrics["MAE"].append(mae)
        aggregate_metrics["RMSE"].append(rmse)
        aggregate_metrics["R²"].append(r2)
        aggregate_metrics["SD"].append(std_dev)

    # Calculate mean metrics
    avg_mae = np.mean(aggregate_metrics["MAE"])
    avg_rmse = np.mean(aggregate_metrics["RMSE"])
    avg_r2 = np.mean(aggregate_metrics["R²"])
    avg_sd = np.mean(aggregate_metrics["SD"])
    return avg_mae, avg_rmse, avg_r2, avg_sd, aggregate_metrics


def calculate_standard_deviation(data, actual_col):
    """Calculate the standard deviation of actual prices."""
    return np.std(data[actual_col])


def main():
    st.title("Predictions & Metrics")

    # Load data
    data = load_data("data/REIT_predictions.csv")

    # Sidebar options
    reit_choices = ["AVB", "EQR", "ESS", "INVH"]
    selected_reit = st.multiselect("Select REIT(s)", reit_choices, default=[])
    date_range = st.date_input(
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
    std_devs = []  # List to store standard deviations for visualization

    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"

        # Calculate metrics
        mae, mse, rmse, r2 = calculate_metrics(filtered_data, adj_close_col, pred_col)
        std_dev = calculate_standard_deviation(filtered_data, adj_close_col)

        # Append to metrics table
        metrics_table.append({
            "REIT": reit, "MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2, "SD (Actual Price)": std_dev
        })

        # Store SD for visualization
        std_devs.append((reit, std_dev))

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
    st.dataframe(metrics_df.style.format({
        "MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.2f}", "SD (Actual Price)": "{:.2f}"
    }))

    # Show scatter plots for each REIT
    st.subheader("Scatter Plots: Actual vs. Predicted Prices")
    for reit, scatter_fig in scatter_plots:
        st.plotly_chart(scatter_fig)

    # Feature Importances Section
    st.subheader("Feature Importances")

    # Define the columns to drop
    drop_columns = [
        'date', 'State', 'Unnamed: 0', 'EQR_adj_close_1day', 'ESS_adj_close_1day', 'AVB_adj_close_1day',
        'INVH_adj_close_1day',
        'EQR_adj_close_1week', 'ESS_adj_close_1week', 'AVB_adj_close_1week', 'INVH_adj_close_1week',
        'EQR_adj_close', 'ESS_adj_close', 'AVB_adj_close', 'INVH_adj_close', 'EQR_adj_close_1month',
        'ESS_adj_close_1month', 'AVB_adj_close_1month', 'INVH_adj_close_1month', 'EQR_adj_close_3month',
        'ESS_adj_close_3month', 'AVB_adj_close_3month', 'INVH_adj_close_3month'
    ]

    # Dictionary to map REIT to model type
    reit_to_model = {
        'AVB': 'ab',
        'ESS': 'ab',
        'EQR': 'ada',
        'INVH': 'xgb'
    }

    # List to store feature importances
    feature_importances = []

    # Load column names
    columns = pd.read_csv("data/final_data.csv", nrows=0).columns
    av_columns = pd.read_csv("data/av_data.csv", nrows=0).columns

    # Combine columns
    combined_columns = columns.union(av_columns)

    # Remove unwanted columns
    feature_names = combined_columns.difference(drop_columns)

    # Iterate through selected REITs
    for reit in selected_reit:
        # Determine the model name
        model_name = reit_to_model.get(reit)
        if model_name is None:
            print(f"Model for {reit} not found.")
            continue

        # Load the model
        model_path = f"models/{reit}_{model_name}.pkl"
        model = load_model(model_path)

        # Get feature importances
        reit_importances = get_feature_importances(model, feature_names)
        reit_importances["REIT"] = reit
        feature_importances.append(reit_importances)

    # Concatenate results for multiple REITs
    feature_importances = pd.concat(feature_importances)

    # Visualization: Feature Importances
    fig = px.bar(
        feature_importances,
        x="Importance",
        y="Feature",
        color="REIT",
        orientation="h",
        title="Feature Importances Across Selected REITs",
        labels={"Importance": "Importance Score", "Feature": "Feature Name"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Aggregate metrics
    avg_mae, avg_rmse, avg_r2, avg_sd, aggregate_metrics = calculate_aggregate_metrics(filtered_data, selected_reit)

    # Aggregate metrics bar chart
    st.subheader("Aggregate Metrics Across Selected REITs")
    agg_fig = go.Figure(data=[
        go.Bar(name="MAE", x=selected_reit, y=aggregate_metrics["MAE"], marker_color='blue'),
        go.Bar(name="RMSE", x=selected_reit, y=aggregate_metrics["RMSE"], marker_color='orange'),
        go.Bar(name="R²", x=selected_reit, y=aggregate_metrics["R²"], marker_color='green'),
        go.Bar(name="SD (Actual Price)", x=selected_reit, y=aggregate_metrics["SD"], marker_color='purple')
    ])
    agg_fig.update_layout(
        barmode="group",
        title="Aggregate Metrics Comparison",
        xaxis_title="REIT",
        yaxis_title="Metric Value",
        legend_title="Metrics",
    )
    st.plotly_chart(agg_fig)

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
