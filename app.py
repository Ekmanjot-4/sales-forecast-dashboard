import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 
st.set_page_config(layout="wide", page_title="Sales Analytics Dashboard")
 
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
    df = pd.read_csv("forecast1.csv")
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
 
@st.cache_data
def load_metrics():
    return pd.read_csv("metrics.csv")
 
# Load all data
df = load_data()
forecast_df = load_forecast()
anomaly_df = load_anomalies()
cluster_df = load_clusters()
metrics_df = load_metrics()
 
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
 
    yearly = df.groupby('Year')['Sales'].sum().reset_index()
    fig1 = px.bar(yearly, x='Year', y='Sales', title="Total Sales by Year")
    st.plotly_chart(fig1, width="stretch")
 
    monthly = df.groupby('Month')['Sales'].sum().reset_index()
    fig2 = px.line(monthly, x='Month', y='Sales', title="Monthly Sales Trend")
    st.plotly_chart(fig2, width="stretch")
 
    st.subheader("Sales by Region and Category")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        region = st.selectbox("Select Region", sorted(df['Region'].unique()))
    with filter_col2:
        category = st.selectbox("Select Category", sorted(df['Category'].unique()))
 
    filtered = df[(df['Region'] == region) & (df['Category'] == category)]
 
    fig3 = px.bar(
        filtered.groupby('Sub-Category')['Sales'].sum().reset_index(),
        x='Sub-Category', y='Sales',
        title=f"Sales by Sub-Category — {category} ({region})"
    )
    st.plotly_chart(fig3, width="stretch")
 
# =========================
# PAGE 2 — FORECAST
# =========================
elif page == "Forecast Explorer":
    st.header("🔮 Forecast Explorer")
 
    col1, col2 = st.columns(2)
 
    with col1:
        category_option = st.selectbox("Select Category", sorted(df['Category'].unique()))
 
    with col2:
        region_option = st.selectbox("Select Region", sorted(df['Region'].unique()))
 
    horizon = st.slider("Forecast Horizon (months ahead)", 1, 3, value=3)
 
    # HISTORICAL DATA
    filtered = df[
        (df['Category'] == category_option) &
        (df['Region'] == region_option)
    ]
 
    if filtered.empty:
        st.warning("No historical data available.")
    else:
        monthly = filtered.groupby(
            pd.Grouper(key="Order Date", freq="ME")
        )["Sales"].sum().reset_index()
 
        # FORECAST DATA — sort chronologically, then keep only the
        # first `horizon` months (the nearest N months ahead of the
        # last training date), not the last `horizon` rows.
        forecast_filtered = forecast_df[
            (forecast_df["Category"] == category_option) &
            (forecast_df["Region"] == region_option)
        ].sort_values("Date").head(horizon)
 
        if forecast_filtered.empty:
            st.error("No forecast data available for this Category/Region combination.")
        else:
            fig = go.Figure()
 
            fig.add_trace(go.Scatter(
                x=monthly["Order Date"],
                y=monthly["Sales"],
                mode='lines',
                name="Actual",
                line=dict(width=3)
            ))
 
            fig.add_trace(go.Scatter(
                x=forecast_filtered["Date"],
                y=forecast_filtered["Sales"],
                mode='lines+markers',
                name="Forecast",
                line=dict(dash="dash")
            ))
 
            fig.update_layout(
                title=f"Sales Forecast — {category_option} ({region_option}), "
                      f"{horizon} month{'s' if horizon > 1 else ''} ahead",
                xaxis_title="Date",
                yaxis_title="Sales",
                hovermode="x unified"
            )
 
            st.plotly_chart(fig, width="stretch")
 
            # METRICS
            st.subheader("📊 Model Performance")
 
            metrics_dict = dict(zip(metrics_df["Metric"], metrics_df["Value"]))
 
            mae = metrics_dict.get("MAE", 0)
            rmse = metrics_dict.get("RMSE", 0)
            mape = metrics_dict.get("MAPE", None)
 
            mcol1, mcol2, mcol3 = st.columns(3)
 
            mcol1.metric("MAE", f"{mae:,.2f}")
            mcol2.metric("RMSE", f"{rmse:,.2f}")
 
            if mape is not None:
                mcol3.metric("MAPE (%)", f"{mape:.2f}%")
 
            st.caption(
                "MAE / RMSE / MAPE reflect overall test-set performance of the "
                "best forecasting model (see Task 4)."
            )
 
# =========================
# PAGE 3 — ANOMALIES
# =========================
elif page == "Anomaly Report":
    st.header("⚠️ Anomaly Report")
 
    weekly = df.groupby(
        pd.Grouper(key="Order Date", freq="W")
    )["Sales"].sum().reset_index()
 
    fig = px.line(
        weekly,
        x="Order Date",
        y="Sales",
        title="Weekly Sales with Anomalies"
    )
 
    fig.add_scatter(
        x=anomaly_df["Date"],
        y=anomaly_df["Sales"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=10)
    )
 
    st.plotly_chart(fig, width="stretch")
 
    st.subheader("📋 Detected Anomalies")
    st.dataframe(
        anomaly_df.sort_values("Date").reset_index(drop=True),
        use_container_width=True
    )
 
# =========================
# PAGE 4 — CLUSTERS
# =========================
elif page == "Product Segments":
    st.header("🧩 Product Demand Segments")
 
    fig = px.scatter(
        cluster_df,
        x="Feature1",
        y="Feature2",
        color=cluster_df["Cluster"].astype(str),
        text="Sub-Category",
        labels={"color": "Cluster"},
        title="Sub-Category Clusters (PCA-reduced feature space)"
    )
 
    fig.update_traces(textposition='top center')
 
    st.plotly_chart(fig, width="stretch")
 
    st.subheader("📊 Sub-Category Cluster Mapping")
 
    # Build the mapping table directly from clusters.csv so it always
    # matches the chart above (no hardcoded/duplicated cluster lists).
    cluster_table = (
        cluster_df[["Cluster", "Sub-Category"]]
        .sort_values(["Cluster", "Sub-Category"])
        .reset_index(drop=True)
    )
    cluster_table["Cluster"] = cluster_table["Cluster"].apply(lambda c: f"Cluster {c}")
 
    st.dataframe(cluster_table, use_container_width=True)
 
    with st.expander("Cluster sizes"):
        st.dataframe(
            cluster_df.groupby("Cluster").size()
            .reset_index(name="Sub-Category Count"),
            use_container_width=True
        )
 