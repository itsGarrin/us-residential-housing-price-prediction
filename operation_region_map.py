#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.express as px
import pandas as pd

# Define the groups of states to highlight
data = pd.DataFrame({
    'State': [
        'CA', 'CO', 'DC', 'MA', 'NY', 'WA',  # EQR
        'CA', 'WA',                          # ESS
        'CA', 'CO', 'CT', 'DC', 'FL', 'MD', 'MA', 'NJ', 'NY', 'NC', 'TX', 'VA', 'WA',  # AVB
        'CA', 'CO', 'TX', 'NC', 'SC', 'GA', 'IL', 'FL', 'NV', 'MN', 'TN', 'AZ', 'WA'   # INVH
    ],
    'Group': [
        'EQR', 'EQR', 'EQR', 'EQR', 'EQR', 'EQR',  # Group 1
        'ESS', 'ESS',                                              # Group 2
        'AVB', 'AVB', 'AVB', 'AVB', 'AVB', 'AVB',
        'AVB', 'AVB', 'AVB', 'AVB', 'AVB', 'AVB', 'AVB',  # Group 3
        'INVH', 'INVH', 'INVH', 'INVH', 'INVH', 'INVH',
        'INVH', 'INVH', 'INVH', 'INVH', 'INVH', 'INVH', 'INVH'   # Group 4
    ]
})

# Combine states and groups to ensure unique group combinations are visualized
data['Group_Combination'] = data.groupby('State')['Group'].transform(lambda x: ', '.join(sorted(set(x))))

# Create a choropleth map with unique group combinations for states
fig = px.choropleth(
    data,
    locations="State",
    locationmode="USA-states",  # Use state abbreviations
    color="Group_Combination", # Assign unique colors for group combinations
    scope="usa",
    color_discrete_sequence=px.colors.qualitative.Pastel  # Set a pastel color scheme
)


fig.update_layout(
    title_text="Region of Operation for each REIT",
    geo=dict(
        lakecolor='rgb(255, 255, 255)',  # Set lake color
        showlakes=True                  # Display lakes
    )
)

fig.show()