import altair as alt
import pandas as pd
import streamlit as st

# Title and Introduction
st.title("Residential REIT Price Prediction Backtest")
st.subheader("Model Performance (2023–2024)")
st.markdown("This dashboard visualizes predictions vs. actual prices for four REITs using tree-based models.")

# Sidebar for filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Select Date Range", [])
selected_reit = st.sidebar.selectbox("Select REIT", ["AVB", "EQR", "ESS", "INVH"])
show_actual = st.sidebar.checkbox("Show Actual Prices", value=True)
highlight_deviation = st.sidebar.checkbox("Highlight Significant Deviations", value=False)

# Data loading
# Assuming 'output' is your DataFrame with predictions
data = pd.read_csv("data/REIT_predictions.csv")
filtered_data = data  # Apply date filters here

# Line chart for Actual vs. Predicted
st.header(f"Actual vs. Predicted Prices for {selected_reit}")
chart = alt.Chart(filtered_data).mark_line().encode(
    x="date:T",
    y="predicted_price:Q",
    color=alt.value("orange")
).interactive()

if show_actual:
    actual_chart = alt.Chart(filtered_data).mark_line().encode(
        x="date:T",
        y="actual_price:Q",
        color=alt.value("blue")
    )
    chart = chart + actual_chart

st.altair_chart(chart, use_container_width=True)

# Performance metrics
st.subheader("Performance Metrics")
st.table(filtered_data[["date", "error", "r_squared"]])  # Example metrics
