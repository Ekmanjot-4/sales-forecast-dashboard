import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(layout="wide")

st.title("📊 Sales Analytics Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    return df

@st.cache_data
def load_forecast():
    df = pd.read_csv("forecast.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_anomalies():
    df = pd.read_csv("anomalies.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_clusters():
    return pd.read_csv("clusters.csv")

df = load_data()
forecast_df = load_forecast()
anomaly_df = load_anomalies()
cluster_df = load_clusters()

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio("Navigate", [
    "Sales Overview",
    "Forecast Explorer",
    "Anomaly Report",
    "Product Segments"
])

# =========================
# PAGE 1 — SALES OVERVIEW
# =========================
if page == "Sales Overview":
    st.header("📈 Sales Overview")

    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.to_period("M").astype(str)

    # Total sales by year
    yearly = df.groupby('Year')['Sales'].sum().reset_index()
    fig1 = px.bar(yearly, x='Year', y='Sales', title="Total Sales by Year")
    st.plotly_chart(fig1, use_container_width=True)

    # Monthly trend
    monthly = df.groupby('Month')['Sales'].sum().reset_index()
    fig2 = px.line(monthly, x='Month', y='Sales', title="Monthly Sales Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # Filters
    region = st.selectbox("Select Region", df['Region'].unique())
    category = st.selectbox("Select Category", df['Category'].unique())

    filtered = df[(df['Region'] == region) & (df['Category'] == category)]

    fig3 = px.bar(filtered, x='Sub-Category', y='Sales',
                  title="Sales by Sub-Category")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Forecast Explorer":
    st.header("🔮 Forecast Explorer")

    import plotly.graph_objects as go
    import numpy as np

    # 1. Load your grouped forecast data
    try:
        forecast_df = pd.read_csv("forecast1.csv")
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    except FileNotFoundError:
        st.error("⚠️ The file 'forecast1.csv' was not found. Please ensure it is in the repository folder.")
        st.stop()

    # 2. Side-by-Side Dropdown Controls
    col1, col2 = st.columns(2)
    with col1:
        category_option = st.selectbox("Select Category", df['Category'].unique())
    with col2:
        region_option = st.selectbox("Select Region", df['Region'].unique())
        
    horizon = st.slider("Forecast Months Ahead", 1, 3)

    # 3. Filter & aggregate historical data by BOTH inputs
    filtered = df[(df['Category'] == category_option) & (df['Region'] == region_option)]
    
    if filtered.empty:
        st.warning(f"No historical data found for '{category_option}' in the '{region_option}' region.")
        st.stop()

    monthly = filtered.groupby(
        pd.Grouper(key="Order Date", freq="ME")
    )["Sales"].sum().reset_index()

    last_date = monthly["Order Date"].max()

    # 4. Filter forecast1.csv rows matching BOTH category and region selections
    cat_region_forecast = forecast_df[
        (forecast_df["Category"] == category_option) & 
        (forecast_df["Region"] == region_option)
    ].copy()
    
    if cat_region_forecast.empty:
        st.error(f"⚠️ No Prophet forecast found in forecast1.csv for '{category_option}' in the '{region_option}' region.")
        st.stop()

    # Enforce user horizon slider limit
    forecast_filtered = cat_region_forecast.head(horizon).copy()

    # Align forecast timeline dates sequentially from the last historical point
    forecast_filtered["Order Date"] = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1),
        periods=len(forecast_filtered),
        freq="ME"
    )

    # 5. Build and Plot Unified Interactive Chart
    fig = go.Figure()

    # Plot actual historical curve line
    fig.add_trace(go.Scatter(
        x=monthly["Order Date"],
        y=monthly["Sales"],
        mode='lines',
        name="Actual History",
        line=dict(color='#1f77b4', width=2)
    ))

    # Connect historical endpoint to Prophet future horizon dynamically
    fig.add_trace(go.Scatter(
        x=pd.concat([monthly["Order Date"].tail(1), forecast_filtered["Order Date"]]),
        y=pd.concat([monthly["Sales"].tail(1), forecast_filtered["Sales"]]),
        mode='lines+markers',
        name="Prophet Forecast",
        line=dict(dash='dash', color='#ff7f0e', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=f"Sales Forecast — {category_option} ({region_option} Region)",
        xaxis_title="Timeline",
        yaxis_title="Total Sales ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)
# =========================
# PAGE 3 — ANOMALIES
# =========================
elif page == "Anomaly Report":
    st.header("⚠️ Anomaly Report")

    # Ensure datetime format
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    anomaly_df['Date'] = pd.to_datetime(anomaly_df['Date'])

    # ✅ Create WEEKLY data (same as model)
    weekly = df.groupby(
        pd.Grouper(key="Order Date", freq="W")
    )["Sales"].sum().reset_index()

    # ✅ Plot weekly sales
    import plotly.express as px
    fig = px.line(
        weekly,
        x="Order Date",
        y="Sales",
        title="Weekly Sales with Anomalies"
    )

    # ✅ Plot anomalies (perfect alignment)
    fig.add_scatter(
        x=anomaly_df["Date"],
        y=anomaly_df["Sales"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("📋 Detected Anomalies")
    st.dataframe(anomaly_df, use_container_width=True)
# =========================
# PAGE 4 — CLUSTERS
# =========================
elif page == "Product Segments":
    st.header("🧩 Product Demand Segments")

    import plotly.express as px
    import pandas as pd

    # Scatter Plot
    fig = px.scatter(
        cluster_df,
        x="Feature1",
        y="Feature2",
        color="Cluster",
        text="Sub-Category"
    )

    fig.update_traces(textposition='top center')

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # FIXED CLUSTER MAPPING
    # =========================

    cluster_mapping = {
        "Cluster 1 (High Growth)": ["Copiers"],

        "Cluster 2 (High Volume, Stable)": [
            "Machines", "Phones", "Chairs", "Tables",
            "Binders", "Storage"
        ],

        "Cluster 3 (Low Volume, Stable)": [
            "Fasteners", "Labels", "Art", "Envelopes",
            "Paper", "Furnishings", "Supplies",
            "Bookcases", "Appliances", "Accessories"
        ]
    }

    # Convert to table format
    rows = []
    for cluster, subcats in cluster_mapping.items():
        for sub in subcats:
            rows.append({
                "Cluster": cluster,
                "Sub-Category": sub
            })

    cluster_table = pd.DataFrame(rows)

    st.subheader("📊 Sub-Category Cluster Mapping")
    st.dataframe(cluster_table, use_container_width=True)