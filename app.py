import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    st.plotly_chart(fig1, use_container_width=True)

    monthly = df.groupby('Month')['Sales'].sum().reset_index()
    fig2 = px.line(monthly, x='Month', y='Sales', title="Monthly Sales Trend")
    st.plotly_chart(fig2, use_container_width=True)

    region = st.selectbox("Select Region", df['Region'].unique())
    category = st.selectbox("Select Category", df['Category'].unique())

    filtered = df[(df['Region'] == region) & (df['Category'] == category)]

    fig3 = px.bar(filtered, x='Sub-Category', y='Sales',
                  title="Sales by Sub-Category")
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# PAGE 2 — FORECAST
# =========================
elif page == "Forecast Explorer":
    st.header("🔮 Forecast Explorer")

    col1, col2 = st.columns(2)

    with col1:
        category_option = st.selectbox("Select Category", df['Category'].unique())

    with col2:
        region_option = st.selectbox("Select Region", df['Region'].unique())

    horizon = st.slider("Forecast Months Ahead", 1, 3)

    # Filter historical data
    filtered = df[
        (df['Category'] == category_option) &
        (df['Region'] == region_option)
    ]

    if filtered.empty:
        st.warning("No historical data available for selection.")
    else:
        # Monthly aggregation
        monthly = filtered.groupby(
            pd.Grouper(key="Order Date", freq="ME")
        )["Sales"].sum().reset_index()

        last_date = monthly["Order Date"].max()

        # Filter forecast data
        forecast_filtered = forecast_df[
            (forecast_df["Category"] == category_option) &
            (forecast_df["Region"] == region_option)
        ].head(horizon).copy()

        if forecast_filtered.empty:
            st.error("No forecast data available for selection.")
        else:
            # Create future dates
            forecast_filtered["Order Date"] = pd.date_range(
                start=last_date + pd.offsets.MonthEnd(1),
                periods=len(forecast_filtered),
                freq="ME"
            )

            # Plot
            fig = go.Figure()

            # Actual
            fig.add_trace(go.Scatter(
                x=monthly["Order Date"],
                y=monthly["Sales"],
                mode='lines',
                name="Actual History"
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=pd.concat([monthly["Order Date"].tail(1), forecast_filtered["Order Date"]]),
                y=pd.concat([monthly["Sales"].tail(1), forecast_filtered["Sales"]]),
                mode='lines+markers',
                name="Forecast"
            ))

            fig.update_layout(
                title=f"Sales Forecast — {category_option} ({region_option})",
                xaxis_title="Date",
                yaxis_title="Sales",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # =========================
            # METRICS (FIXED)
            # =========================
            st.subheader("📊 Model Performance")

            # Convert row-based CSV → dictionary
            metrics_dict = dict(zip(metrics_df["Metric"], metrics_df["Value"]))

            mae = metrics_dict.get("MAE", 0)
            rmse = metrics_dict.get("RMSE", 0)
            mape = metrics_dict.get("MAPE", None)

            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", f"{mae:,.2f}")
            col2.metric("RMSE", f"{rmse:,.2f}")

            if mape is not None:
                col3.metric("MAPE (%)", f"{mape:.2f}%")

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

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detected Anomalies")
    st.dataframe(anomaly_df, use_container_width=True)

# =========================
# PAGE 4 — CLUSTERS
# =========================
elif page == "Product Segments":
    st.header("🧩 Product Demand Segments")

    fig = px.scatter(
        cluster_df,
        x="Feature1",
        y="Feature2",
        color="Cluster",
        text="Sub-Category"
    )

    fig.update_traces(textposition='top center')

    st.plotly_chart(fig, use_container_width=True)

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

    rows = []
    for cluster, subs in cluster_mapping.items():
        for sub in subs:
            rows.append({"Cluster": cluster, "Sub-Category": sub})

    cluster_table = pd.DataFrame(rows)

    st.subheader("📊 Sub-Category Cluster Mapping")
    st.dataframe(cluster_table, use_container_width=True)