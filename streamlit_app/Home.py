import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

    # Key metrics section
    st.subheader("📋 Key Metrics")
    latest_date = data["date"].max()
    earliest_date = data["date"].min()

    # Calculate metrics for each REIT
    reit_choices = ["AVB", "EQR", "ESS", "INVH"]
    metrics = []
    for reit in reit_choices:
        actual_col = f"{reit}_adj_close"
        pred_col = f"{reit}_pred"
        residuals = data[actual_col] - data[pred_col]
        mae = np.mean(np.abs(residuals))
        metrics.append({
            "REIT": reit,
            "MAE": mae,
            "Residual StdDev": np.std(residuals),

            "Largest Error Value": residuals.max()
        })

    # Create a dataframe for metrics
    metrics_df = pd.DataFrame(metrics)

    # Display metrics in a table
    st.write(f"**📅 Date Range:** {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
    st.dataframe(
        metrics_df.style.format({
            "MAE": "{:.2f}",
            "Residual StdDev": "{:.2f}",
            "Largest Error Value": "{:.2f}"
        }),
        use_container_width=True
    )

    # Highlight chart: Predicted vs Actual Prices
    st.subheader("📉 REIT Predicted vs Actual Prices")

    # Define color mapping
    reit_colors = {
        "AVB": ("#1f77b4", "#aec7e8"),  # Dark blue, light blue
        "EQR": ("#ff7f0e", "#ffbb78"),  # Dark orange, light orange
        "ESS": ("#d62728", "#ff9896"),  # Dark red, light red
        "INVH": ("#2ca02c", "#98df8a")  # Dark green, light green
    }

    # Create the plot
    fig = go.Figure()

    for reit, (actual_color, predicted_color) in reit_colors.items():
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data[f"{reit}_adj_close"],
                mode="lines",
                name=f"{reit} Actual",
                line=dict(color=actual_color, width=2)
            )
        )
        # Add predicted values
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data[f"{reit}_pred"],
                mode="lines",
                name=f"{reit} Predicted",
                line=dict(color=predicted_color, width=2, dash="dash")
            )
        )

    # Update layout
    fig.update_layout(
        title="Predicted vs Actual Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend_title="Legend",
        template="plotly_white",
        xaxis=dict(tickformat="%Y-%m-%d", showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified"
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    home_page()
