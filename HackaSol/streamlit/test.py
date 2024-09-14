import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Load the dataset
deliveres = pd.read_csv('../dataset/deliveries.csv')

# Filter the dataset
filtered_df = deliveres[(deliveres['batter'] == 'V Kohli') & (deliveres['bowling_team'] == 'Chennai Super Kings')]

# Group by dismissal kind and count occurrences
dismissal_counts = filtered_df['dismissal_kind'].value_counts()

# Labels (dismissal kinds) and values (counts of dismissals)
labels = dismissal_counts.index.tolist()
values = dismissal_counts.values.tolist()

# Define a monochromatic color scheme
color_scheme = ["#ffa600", "#e8eef1","#43b0f1","#057dcd","#1e3d58"]

# Create the donut chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, 
                             marker=dict(colors=color_scheme))])

# Customize layout and title
fig.update_layout(
    annotations=[dict(text='Dismissals', x=0.5, y=0.5, font_size=20, showarrow=False)]
)

# Display the chart in Streamlit
st.plotly_chart(fig)
