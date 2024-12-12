import streamlit as st
import pandas as pd

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['date'])

def main():
    st.title("Data Summary")
    data = load_data("data/REIT_predictions.csv")

    # Data overview
    st.write(f"**Total Rows:** {data.shape[0]}")
    st.write(f"**Total Columns:** {data.shape[1]}")
    st.write(f"**Date Range:** {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    st.write(data.describe())  # Summary statistics

if __name__ == "__main__":
    main()
