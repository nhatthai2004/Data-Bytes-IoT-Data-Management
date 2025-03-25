import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page configuration
st.set_page_config(page_title="BME280 Sensor Dashboard", layout="wide")

# DataBuffer class for managing continuous data
class DataBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def append(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_data(self):
        return pd.DataFrame(self.buffer)

# Function to load initial data from CSV
@st.cache_data
def load_initial_data():
    df = pd.read_csv('bme280_data.csv')
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    return df

# Initialize session state
if 'data_buffer' not in st.session_state:
    initial_data = load_initial_data()
    st.session_state.data_buffer = DataBuffer(max_size=len(initial_data) * 2)
    for _, row in initial_data.iterrows():
        st.session_state.data_buffer.append(row.to_dict())
    st.session_state.sample_start = 0

# Get actual column names
data = st.session_state.data_buffer.get_data()
timestamp_col = data.columns[0]
data_cols = data.columns[1:]

# Sidebar for controls
st.sidebar.title("Controls")
graph_type = st.sidebar.selectbox("Graph Type", ['Line', 'Scatter', 'Distribution'])
samples = st.sidebar.number_input("Number of Samples", min_value=10, max_value=len(data), value=100)
update_interval = 10  # Fixed update interval for BME280 sensor

# Main content
st.title("BME280 Sensor Data Visualization")

# Create placeholders for plots
plot_placeholders = {col: st.empty() for col in data_cols}

# Define colors for each parameter
colors = {data_cols[0]: 'red', data_cols[1]: 'green', data_cols[2]: 'blue'}

# Function to update plots
def update_plots():
    data = st.session_state.data_buffer.get_data()
    df_subset = data.iloc[-samples:]  # Always show the most recent samples
    
    for col in data_cols:
        fig = make_subplots(rows=1, cols=1)

        if graph_type == 'Distribution':
            fig.add_trace(go.Histogram(x=df_subset[col], name=col, marker=dict(color=colors.get(col, '#FFFFFF'))))
        else:
            if graph_type == 'Line':
                fig.add_trace(go.Scatter(x=df_subset[timestamp_col], y=df_subset[col], mode='lines', name=col, line=dict(color=colors.get(col, '#FFFFFF'))))
            else:  # Scatter
                fig.add_trace(go.Scatter(x=df_subset[timestamp_col], y=df_subset[col], mode='markers', name=col, marker=dict(color=colors.get(col, '#FFFFFF'))))

        fig.update_layout(
            title=f"{col} Data",
            xaxis_title="Timestamp" if graph_type != 'Distribution' else "Value",
            yaxis_title=col if graph_type != 'Distribution' else "Count",
            legend_title="Parameter",
            height=400,
            template="plotly_dark"
        )
        plot_placeholders[col].plotly_chart(fig, use_container_width=True)

    # Display statistics
    st.subheader("Statistics")
    stats = {col: [f"{df_subset[col].mean():.2f}", f"{df_subset[col].median():.2f}", f"{df_subset[col].std():.2f}"] for col in data_cols}
    stats_df = pd.DataFrame(stats, index=['Mean', 'Median', 'Std Dev'])
    st.table(stats_df)

    # Display current range
    st.write(f"Displaying the most recent {samples} samples out of {len(data)} total samples")

# Function to simulate new data
def simulate_new_data():
    last_timestamp = st.session_state.data_buffer.get_data().iloc[-1][timestamp_col]
    new_data = {
        timestamp_col: last_timestamp + pd.Timedelta(seconds=10),
        data_cols[0]: np.random.uniform(20, 30),  # Simulated Temperature
        data_cols[1]: np.random.uniform(950, 1050),  # Simulated Pressure
        data_cols[2]: np.random.uniform(30, 90)  # Simulated Humidity
    }
    st.session_state.data_buffer.append(new_data)

# Update plots initially
update_plots()

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    data = st.session_state.data_buffer.get_data()
    st.dataframe(data.iloc[-samples:])  # Show only the most recent samples

# Auto-update feature
if st.sidebar.checkbox("Enable Auto-update"):
    st.write(f"Auto-updating every {update_interval} seconds...")
    while True:
        simulate_new_data()
        time.sleep(update_interval)
        update_plots()
        st.rerun()