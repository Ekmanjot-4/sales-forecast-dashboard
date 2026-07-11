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
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # Load data
    forecast_df = pd.read_csv("forecast.csv")
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

    test_results = pd.read_csv("test_results.csv")
    metrics=pd.read_csv("metrics.csv")

    # Inputs
    option = st.selectbox("Select Category", df['Category'].unique())
    horizon = st.slider("Forecast Months Ahead", 1, 3)

    # Filter & aggregate
    filtered = df[df['Category'] == option]

    monthly = filtered.groupby(
        pd.Grouper(key="Order Date", freq="ME")
    )["Sales"].sum().reset_index()

    last_date = monthly["Order Date"].max()

    # Forecast prep
    forecast_filtered = forecast_df.head(horizon).copy()

    forecast_filtered["Order Date"] = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1),
        periods=horizon,
        freq="ME"
    )

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly["Order Date"],
        y=monthly["Sales"],
        mode='lines',
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([monthly["Order Date"].tail(1), forecast_filtered["Order Date"]]),
        y=pd.concat([monthly["Sales"].tail(1), forecast_filtered["Sales"]]),
        mode='lines+markers',
        name="Forecast",
        line=dict(dash='dash')
    ))

    fig.update_layout(title=f"Sales Forecast ({option})")

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.subheader("📊 Model Performance")

    try:
        mae = metrics.loc[metrics["Metric"] == "MAE", "Value"].values[0]
        rmse = metrics.loc[metrics["Metric"] == "RMSE", "Value"].values[0]

        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")

    except Exception:
        st.warning("⚠️ metrics.csv not found or invalid.")
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