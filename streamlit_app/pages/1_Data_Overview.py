import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st


# Load data
@st.cache_data
def load_full_data(file_path):
    """Load the full dataset."""
    data = pd.read_csv(file_path, parse_dates=["date"])
    return data


def data_overview_page():
    st.title("Data Overview")

    # Load the data
    file_path = "data/final_data.csv"  # Update the path if needed
    data = load_full_data(file_path)

    # Sidebar filters
    st.sidebar.header("Filters")
    # State selector
    selected_states = st.multiselect(
        "Select State(s):",
        options=data["State"].unique(),
        default=[],  # Start with no states selected
        help="Select one or more states to filter the data. Leave empty to view all states."
    )

    # Filter data based on selected states
    if selected_states:
        filtered_data = data[data["State"].isin(selected_states)]
    else:
        st.info("No states selected. Showing data for all states.")
        filtered_data = data
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data["date"].min(), data["date"].max()),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )

    # Filter the data
    filtered_data = filtered_data[
        (data["date"] >= pd.to_datetime(date_range[0])) &
        (data["date"] <= pd.to_datetime(date_range[1]))
        ]

    # Display filtered dataframe
    st.subheader("Filtered Dataframe")
    st.dataframe(filtered_data)

    # Download button for the filtered data
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_data.to_csv(index=False),
        file_name="filtered_real_estate_data.csv",
        mime="text/csv"
    )

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    numeric_columns = filtered_data.select_dtypes(include=["float64", "int64"]).columns
    st.dataframe(filtered_data[numeric_columns].describe().transpose())

    # Visualizations Section
    st.subheader("Visualizations")

    # 1. Time-Series Line Chart
    st.markdown("### Time-Series Trends")
    time_series_column = st.selectbox("Select Column to Plot Over Time", numeric_columns)
    if time_series_column:
        fig = px.line(filtered_data, x="date", y=time_series_column, title=f"Trend of {time_series_column} Over Time")
        st.plotly_chart(fig)

    # 2. Interactive Correlation Heatmap with Feature Selection
    st.markdown("### Correlation Heatmap")
    if len(numeric_columns) > 1:
        selected_features = st.multiselect(
            "Select Features for Correlation Matrix:",
            options=numeric_columns,
            default=[],  # Default to all numeric columns
            help="Select the numeric features to include in the correlation heatmap."
        )

        if len(selected_features) > 1:
            corr_matrix = filtered_data[selected_features].corr()

            # Create Plotly heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="Viridis",
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Correlation")
                )
            )
            fig.update_layout(
                title="Interactive Correlation Heatmap",
                xaxis_title="Features",
                yaxis_title="Features",
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig)
        else:
            st.warning("Please select at least two features to display the correlation heatmap.")

    # 3. Bar Chart by State
    st.markdown("### Aggregated Metrics by State")
    bar_chart_metric = st.selectbox("Select Metric for Bar Chart", numeric_columns)
    if bar_chart_metric:
        state_agg = filtered_data.groupby("State")[bar_chart_metric].mean().reset_index()
        fig = px.bar(state_agg, x="State", y=bar_chart_metric, title=f"{bar_chart_metric} by State")
        st.plotly_chart(fig)

    # 4. Scatter Plot
    st.markdown("### Scatter Plot")
    x_col = st.selectbox("X-axis Column", numeric_columns)
    y_col = st.selectbox("Y-axis Column", numeric_columns, index=1)
    if x_col and y_col:
        fig = px.scatter(
            filtered_data,
            x=x_col,
            y=y_col,
            color="State",
            title=f"{y_col} vs {x_col}",
            labels={x_col: x_col, y_col: y_col},
        )
        st.plotly_chart(fig)


# Call this function when navigating to the "Data Overview" page
if __name__ == "__main__":
    data_overview_page()
