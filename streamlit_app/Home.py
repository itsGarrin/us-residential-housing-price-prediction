import streamlit as st

def main():
    st.title("REIT Price Prediction Dashboard")
    st.write("""
        Welcome to the REIT Price Prediction Dashboard!  
        Use the sidebar to navigate between different sections:
        - **Data Summary**: Overview of the dataset.
        - **Visualizations**: Plots of actual vs. predicted prices.
        - **Predictions**: Evaluation metrics and download options.
    """)

if __name__ == "__main__":
    main()