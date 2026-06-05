import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="HydroClimatic Intelligence Dashboard",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for StreamUI aesthetics
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    div[data-testid="stExpander"] { border: 1px solid #e9ecef; border-radius: 8px; }
    </style>
    """, unsafe_cache=True)

# ==========================================
# 2. DATA INGESTION & CACHING
# ==========================================
@st.cache_data
def load_data():
    # Looks for file in local directory or data folder
    try:
        df = pd.read_csv("Rainfall_Data_LL.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/Rainfall_Data_LL.csv")
    
    # Standardize column naming strips if any whitespace exists
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load dataset. Please verify 'Rainfall_Data_LL.csv' path. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION & FILTERS
# ==========================================
st.sidebar.title("⛈️ HydroClimatic Engine")
st.sidebar.markdown("---")

subdivisions = sorted(df['SUBDIVISION'].unique())
selected_subdivision = st.sidebar.selectbox("🎯 Target Sub-Division", subdivisions, index=0)

# Year Filter Slider
min_year, max_year = int(df['YEAR'].min()), int(df['YEAR'].max())
selected_years = st.sidebar.slider("📅 Temporal Bound", min_year, max_year, (min_year, max_year))

# Global Filtered DataFrames
sub_df = df[df['SUBDIVISION'] == selected_subdivision]
sub_df_filtered = sub_df[(sub_df['YEAR'] >= selected_years[0]) & (sub_df['YEAR'] <= selected_years[1])]

# ==========================================
# 4. STREAMUI NAVIGATION TABS
# ==========================================
st.title("🌦️ India Rainfall Analytics & Deep Insights")
st.markdown("An advanced environmental data platform evaluating macro-climatic historical shifts.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Dashboard", 
    "🔍 Regional Deep-Dive", 
    "📈 Trend & Anomaly Analytics", 
    "💬 Rainfall Chat Insights"
])

# ------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# ------------------------------------------
with tab1:
    st.subheader("🌐 National Macro-Overview")
    
    # Calculate Key Metrics Across Data
    national_mean = df['ANNUAL'].mean()
    max_record = df.loc[df['ANNUAL'].idxmax()]
    min_record = df.loc[df['ANNUAL'].idxmin()]
    
    # Coefficient of variation per subdivision to define volatility
    sub_cv = df.groupby('SUBDIVISION')['ANNUAL'].std() / df.groupby('SUBDIVISION')['ANNUAL'].mean()
    most_volatile_sub = sub_cv.idxmax()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical National Mean", f"{national_mean:.1f} mm")
    col2.metric("Absolute Max Record", f"{max_record['ANNUAL']:.1f} mm", f"{max_record['SUBDIVISION']} ({max_record['YEAR']})", delta_color="inverse")
    col3.metric("Absolute Min Record", f"{min_record['ANNUAL']:.1f} mm", f"{min_record['SUBDIVISION']} ({min_record['YEAR']})", delta_color="normal")
    col4.metric("Highest Volatility Sub-div", most_volatile_sub)

    st.markdown("---")
    
    m_col1, m_col2 = st.columns([3, 2])
    
    with m_col1:
        st.markdown("##### 📍 Geospatial Precipitation Intensity Map")
        # Aggregating coordinates for mapping layout
        geo_df = df.groupby(['SUBDIVISION', 'Latitude', 'Longitude'])['ANNUAL'].mean().reset_index()
        fig_map = px.scatter_mapbox(
            geo_df, 
            lat="Latitude", 
            lon="Longitude", 
            size="ANNUAL", 
            color="ANNUAL",
            color_continuous_scale=px.colors.sequential.Viridis, 
            zoom=3.5,
            hover_name="SUBDIVISION",
            mapbox_style="open-street-map",
            height=500,
            title="Long-Period Average Annual Rainfall Intensity"
        )
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    with m_col2:
        st.markdown("##### 🏆 Ranking Benchmarks")
        rank_metric = st.selectbox("Sort Ranking Criterion", ["Wettest Regions", "Driest Regions"])
        
        agg_sub = df.groupby('SUBDIVISION')['ANNUAL'].mean().reset_index()
        if rank_metric == "Wettest Regions":
            sorted_agg = agg_sub.sort_values(by='ANNUAL', ascending=True).tail(10)
        else:
            sorted_agg = agg_sub.sort_values(by='ANNUAL', ascending=False).tail(10)
            
        fig_rank = px.bar(
            sorted_agg, 
            x='ANNUAL', 
            y='SUBDIVISION', 
            orientation='h',
            color='ANNUAL',
            color_continuous_scale=px.colors.sequential.Cividis if rank_metric=="Driest Regions" else px.colors.sequential.Plasma,
            labels={'ANNUAL': 'Mean Annual Rainfall (mm)', 'SUBDIVISION': 'Subdivision'}
        )
        fig_rank.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=430)
        st.plotly_chart(fig_rank, use_container_width=True)

# ------------------------------------------
# TAB 2: REGIONAL DEEP-DIVE
# ------------------------------------------
with tab2:
    st.subheader(f"🔍 Regional Profiling: {selected_subdivision}")
    
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.markdown("##### 📈 Timeline of Annual Precipitation")
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=sub_df_filtered['YEAR'], y=sub_df_filtered['ANNUAL'],
            mode='lines+markers', name='Annual Total', line=dict(color='#1f77b4')
        ))
        # Adding Rolling 10-year Average
        sub_df_filtered = sub_df_filtered.sort_values('YEAR')
        sub_df_filtered['Rolling_10Y'] = sub_df_filtered['ANNUAL'].rolling(window=10, min_periods=1).mean()
        fig_timeline.add_trace(go.Scatter(
            x=sub_df_filtered['YEAR'], y=sub_df_filtered['Rolling_10Y'],
            mode='lines', name='10-Year Trend Line', line=dict(color='#ff7f0e', dash='dash')
        ))
        fig_timeline.update_layout(xaxis_title="Year", yaxis_title="Rainfall (mm)", legend_orientation="h")
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    with r_col2:
        st.markdown("##### 🍰 Macro-Seasonal Distribution Share")
        seasonal_cols = ['Jan-Feb', 'Mar-May', 'June-September', 'Oct-Dec']
        seasonal_sums = sub_df_filtered[seasonal_cols].mean()
        
        fig_pie = px.pie(
            names=seasonal_sums.index, 
            values=seasonal_sums.values,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(legend_orientation="v")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.markdown("##### 📊 Monthly In-depth Distribution Spread Across Selected Bounds")
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    melted_months = sub_df_filtered.melt(id_vars=['YEAR'], value_vars=months, var_name='Month', value_name='Rainfall')
    fig_box = px.box(
        melted_months, 
        x='Month', 
        y='Rainfall',
        color='Month',
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title="Distribution Volatility Ranges (Monthly Metrics Boxplot)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------
# TAB 3: TREND & ANOMALY ANALYTICS
# ------------------------------------------
with tab3:
    st.subheader("📊 Statistical Climate Anomaly Analysis")
    st.markdown(
        r"Anomalies are computed relative to long-period operational variables. "
        r"A standard Z-Score normalization framework is applied: $Z = \frac{X - \mu}{\sigma}$"
    )
    
    # Standard calculations
    mu = sub_df['ANNUAL'].mean()
    sigma = sub_df['ANNUAL'].std()
    
    sub_df['Z_Score'] = (sub_df['ANNUAL'] - mu) / sigma
    sub_df['Anomaly_Type'] = np.where(sub_df['Z_Score'] > 1.2, 'Surplus Year (Flood Threat)',
                               np.where(sub_df['Z_Score'] < -1.2, 'Deficit Year (Drought Threat)', 'Normal Variability'))
    
    filtered_anomalies = sub_df[(sub_df['YEAR'] >= selected_years[0]) & (sub_df['YEAR'] <= selected_years[1])]
    
    fig_anomaly = px.bar(
        filtered_anomalies, 
        x='YEAR', 
        y='Z_Score', 
        color='Anomaly_Type',
        color_discrete_map={
            'Surplus Year (Flood Threat)': '#2ca02c',
            'Deficit Year (Drought Threat)': '#d62728',
            'Normal Variability': '#bcbd22'
        },
        labels={'Z_Score': 'Z-Score Anomaly Vector', 'YEAR': 'Year'}
    )
    fig_anomaly.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="+1.2 Std Dev threshold")
    fig_anomaly.add_hline(y=-1.2, line_dash="dash", line_color="red", annotation_text="-1.2 Std Dev threshold")
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Advanced Statistical Linear Regression Projection
    st.markdown("##### 📉 Long-Term Trend Trajectory Fitting")
    X_reg = filtered_anomalies[['YEAR']].values
    y_reg = filtered_anomalies['ANNUAL'].values
    
    if len(X_reg) > 1:
        model = LinearRegression().fit(X_reg, y_reg)
        slope = model.coef_[0]
        intercept = model.intercept_
        r_sq = model.score(X_reg, y_reg)
        
        st.info(
            f"**Linear Trend Analysis Expression:** $\\text{{Rainfall}} = {slope:.3f} \\times \\text{{Year}} + {intercept:.1f}$ \n\n"
            f"**Decadal Variance Factor:** According to this localized regression profile, rainfall is changing at a rate of "
            f"**{slope * 10:.2f} mm per decade** (with an $R^2$ score of {r_sq:.4f})."
        )
    else:
        st.warning("Insufficient timeframe chosen to safely construct statistical regression lines.")

# ------------------------------------------
# TAB 4: RAINFALL CHAT INSIGHTS
# ------------------------------------------
with tab4:
    st.subheader("💬 HydroClimatic AI Chat Assistant")
    st.markdown("Query historical records naturally! Enter plain-text expressions to retrieve direct data insights.")
    
    # Initialize Streamlit chat logs session-state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your HydroClimatic Assistant engine. Ask me specific analytical questions like:\n- *What is the highest rainfall in Kerala?*\n- *Show me average rainfall for Lakshadweep.*\n- *Which year had the lowest rainfall overall?*\n- *Identify drought vectors in Punjab.*"}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat engine analytical lookup function
    def evaluate_hydro_query(query, core_df):
        q = query.lower().strip()
        all_subs = core_df['SUBDIVISION'].unique()
        matched_subs = [s for s in all_subs if s.lower() in q]
        
        # Check logic strings
        if "highest" in q or "maximum" in q or "peak" in q:
            if matched_subs:
                target = matched_subs[0]
                t_df = core_df[core_df['SUBDIVISION'] == target]
                row = t_df.loc[t_df['ANNUAL'].idxmax()]
                return f"📊 The historical highest recorded annual rainfall for **{target}** was **{row['ANNUAL']:.1f} mm** in **{int(row['YEAR'])}**."
            else:
                row = core_df.loc[core_df['ANNUAL'].idxmax()]
                return f"🌐 Across all data records in India, the absolute highest annual rainfall was **{row['ANNUAL']:.1f} mm** recorded in **{row['SUBDIVISION']}** during the year **{int(row['YEAR'])}**."
                
        elif "lowest" in q or "minimum" in q or "driest" in q:
            if matched_subs:
                target = matched_subs[0]
                t_df = core_df[core_df['SUBDIVISION'] == target]
                row = t_df.loc[t_df['ANNUAL'].idxmin()]
                return f"📉 The historical lowest recorded annual rainfall for **{target}** was **{row['ANNUAL']:.1f} mm** in **{int(row['YEAR'])}**."
            else:
                row = core_df.loc[core_df['ANNUAL'].idxmin()]
                return f"⚠️ The lowest annual rainfall value logged across all records was **{row['ANNUAL']:.1f} mm** in **{row['SUBDIVISION']}** in the year **{int(row['YEAR'])}**."
                
        elif "average" in q or "mean" in q or "typical" in q:
            if matched_subs:
                target = matched_subs[0]
                val = core_df[core_df['SUBDIVISION'] == target]['ANNUAL'].mean()
                return f"ℹ️ The long-term calculated mean annual rainfall for **{target}** stands at **{val:.1f} mm**."
            else:
                val = core_df['ANNUAL'].mean()
                return f"ℹ️ The collective multi-regional historical average annual rainfall across India is **{val:.1f} mm**."
                
        elif "drought" in q or "deficit" in q:
            if matched_subs:
                target = matched_subs[0]
                t_df = core_df[core_df['SUBDIVISION'] == target].copy()
                m = t_df['ANNUAL'].mean()
                s = t_df['ANNUAL'].std()
                dr_years = t_df[t_df['ANNUAL'] < (m - 1.2 * s)]['YEAR'].astype(int).tolist()
                if dr_years:
                    return f"🚨 **Drought/Severe Deficit Years** identified for **{target}** (rainfall $< \\mu - 1.2\\sigma$):\n\n {', '.join(map(str, dr_years))}"
                else:
                    return f"✅ Based on our historical threshold profile, no anomalous severe drought events were isolated for **{target}**."
            else:
                return "💡 Please specify a valid subdivision region name along with your request (e.g., *'drought in Punjab'*)."
                
        else:
            return "🤖 I parsed your input but could not extract a matching query combination. Try targeting specific commands using terms like *'highest'*, *'lowest'*, *'average'*, or *'drought'* paired with an explicit subdivision name."

    # Listen for new query entries
    if user_prompt := st.chat_input("Ask about historical precipitation trends..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Process output
        engine_response = evaluate_hydro_query(user_prompt, df)
        
        with st.chat_message("assistant"):
            st.markdown(engine_response)
        st.session_state.messages.append({"role": "assistant", "content": engine_response})
