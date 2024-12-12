import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['date'])


def plot_actual_vs_predicted(data, adj_close_col, adj_close_1week_col, pred_col):
    melted_data = data.melt(
        id_vars="date",
        value_vars=[adj_close_col, adj_close_1week_col, pred_col],
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


def main():
    st.title("Visualizations")

    # Load data
    data = load_data("data/REIT_predictions.csv")

    # Sidebar options
    reit_choices = ["AVB", "EQR", "ESS", "INVH"]
    selected_reit = st.sidebar.multiselect("Select REIT(s)", reit_choices, default=[])
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(data["date"].min(), data["date"].max()),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )

    if not selected_reit:
        st.warning("Please select at least one REIT using the navigation on the left.")
        return

    # Filter data
    filtered_data = data[
        (data["date"] >= pd.to_datetime(date_range[0])) & (data["date"] <= pd.to_datetime(date_range[1]))
        ]

    for reit in selected_reit:
        adj_close_col = f"{reit}_adj_close"
        adj_close_1week_col = f"{reit}_adj_close_1week"
        pred_col = f"{reit}_pred"

        st.subheader(f"{reit} Actual vs Predicted Prices")
        plot_actual_vs_predicted(filtered_data, adj_close_col, adj_close_1week_col, pred_col)

        st.subheader(f"{reit} Residual Distribution")
        plot_residual_distribution(filtered_data, adj_close_col, pred_col)


if __name__ == "__main__":
    main()
