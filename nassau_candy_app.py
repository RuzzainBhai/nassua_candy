import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy Optimizer",
    layout="wide",
    page_icon="🍬"
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
FILE_NAME = "nassau_candy_distributor.csv"

FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":  (32.076176,  -81.088371),
    "Sugar Shack":      (48.11914,   -96.18115),
    "Secret Factory":   (41.446333,  -90.565487),
    "The Other Factory":(35.1175,    -89.971107),
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

STATE_COORDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337), "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137), "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526), "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067), "Louisiana": (31.169960, -91.867805),
    "Maine": (44.693947, -69.381927), "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106), "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192), "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368), "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082), "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896), "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482), "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419), "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915), "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938), "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780), "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828), "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461), "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686), "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494), "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508), "Wyoming": (42.755966, -107.302490),
    "District of Columbia": (38.897438, -77.026817),
}

# ─────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(FILE_NAME)

    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)

    # Lead time (target variable)
    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Map factory
    df['Factory'] = df['Product Name'].map(PRODUCT_FACTORY_MAP)

    # Shipping distance using state centroids
    def haversine(lat1, lon1, lat2, lon2):
        R = 3958.8
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def get_distance(row):
        factory = row['Factory']
        state   = row['State/Province']
        if factory in FACTORY_COORDS and state in STATE_COORDS:
            f_lat, f_lon = FACTORY_COORDS[factory]
            d_lat, d_lon = STATE_COORDS[state]
            return haversine(f_lat, f_lon, d_lat, d_lon)
        return np.nan

    df['Shipping Distance'] = df.apply(get_distance, axis=1)
    df['Profit Margin %']   = (df['Gross Profit'] / df['Sales']) * 100

    return df

# ─────────────────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    model_df = df[['Ship Mode','Region','Factory','Product Name',
                   'Shipping Distance','Sales','Units','Cost',
                   'Profit Margin %','Lead Time']].dropna()

    le_dict = {}
    for col in ['Ship Mode','Region','Factory','Product Name']:
        le = LabelEncoder()
        model_df = model_df.copy()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        le_dict[col] = le

    X = model_df.drop('Lead Time', axis=1)
    y = model_df['Lead Time']
    feat_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "MAE":  round(mean_absolute_error(y_test, preds), 2),
            "R2":   round(r2_score(y_test, preds), 3),
        }

    best_name = min(results, key=lambda k: results[k]['RMSE'])
    best_model = results[best_name]['model']

    return best_model, le_dict, feat_cols, results, best_name

# ─────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────
def simulate_reassignment(best_model, le_dict, feat_cols,
                           product_name, region, ship_mode,
                           avg_sales, avg_units, avg_cost, avg_margin):
    results = []
    for factory in FACTORY_COORDS.keys():
        row = {
            'Ship Mode':         ship_mode,
            'Region':            region,
            'Factory':           factory,
            'Product Name':      product_name,
            'Shipping Distance': 1000.0,
            'Sales':             avg_sales,
            'Units':             avg_units,
            'Cost':              avg_cost,
            'Profit Margin %':   avg_margin,
        }
        for col in ['Ship Mode','Region','Factory','Product Name']:
            try:
                row[col] = le_dict[col].transform([str(row[col])])[0]
            except ValueError:
                row[col] = 0
        X_input = pd.DataFrame([row])[feat_cols]
        pred_lt = best_model.predict(X_input)[0]
        results.append({'Factory': factory, 'Predicted Lead Time': round(pred_lt, 1)})

    return pd.DataFrame(results).sort_values('Predicted Lead Time').reset_index(drop=True)

# ─────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def generate_recommendations(_best_model, _le_dict, feat_cols, df):
    recommendations = []
    for product, current_factory in PRODUCT_FACTORY_MAP.items():
        subset = df[df['Product Name'] == product]
        if subset.empty:
            continue
        for region in subset['Region'].unique():
            reg_data = subset[subset['Region'] == region]
            ship_mode  = reg_data['Ship Mode'].mode()[0]
            avg_sales  = reg_data['Sales'].mean()
            avg_units  = reg_data['Units'].mean()
            avg_cost   = reg_data['Cost'].mean()
            avg_margin = reg_data['Profit Margin %'].mean()
            current_lt = reg_data['Lead Time'].mean()

            sim = simulate_reassignment(
                _best_model, _le_dict, feat_cols,
                product, region, ship_mode,
                avg_sales, avg_units, avg_cost, avg_margin)

            best_row = sim.iloc[0]
            improvement = current_lt - best_row['Predicted Lead Time']

            if best_row['Factory'] != current_factory and improvement > 0:
                recommendations.append({
                    'Product':             product,
                    'Region':              region,
                    'Current Factory':     current_factory,
                    'Recommended Factory': best_row['Factory'],
                    'Current Avg LT (days)':    round(current_lt, 1),
                    'Predicted New LT (days)':  round(best_row['Predicted Lead Time'], 1),
                    'LT Improvement (days)':    round(improvement, 1),
                    'Improvement %':            round((improvement / current_lt) * 100, 1),
                })

    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df = rec_df.sort_values('Improvement %', ascending=False).reset_index(drop=True)
    return rec_df

# ─────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────
st.title("🍬 Nassau Candy Distributor — Factory Reallocation Optimizer")
st.caption(f"📂 Data Source: `{FILE_NAME}` &nbsp;|&nbsp; 10,194 orders across 15 products & 5 factories")

# Load data
with st.spinner("Loading Nassau_Candy_Distributor.csv and engineering features..."):
    df = load_and_prepare_data()

# Train models
with st.spinner("Training ML models (Linear Regression, Random Forest, Gradient Boosting)..."):
    best_model, le_dict, feat_cols, model_results, best_name = train_models(df)

# Generate recommendations
with st.spinner("Running factory reassignment simulation engine..."):
    rec_df = generate_recommendations(best_model, le_dict, feat_cols, df)

# ── TOP KPI BANNER ─────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Total Orders",    f"{len(df):,}")
k2.metric("🏭 Factories",       len(FACTORY_COORDS))
k3.metric("🍫 Products",        df['Product Name'].nunique())
k4.metric("⏱ Avg Lead Time",   f"{df['Lead Time'].mean():.1f} days")
k5.metric("💰 Avg Profit Margin", f"{df['Profit Margin %'].mean():.1f}%")

st.divider()

# ── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA Overview",
    "🤖 Model Performance",
    "🔮 Factory Simulator",
    "📋 Recommendations",
    "⚠️ Risk & Impact Panel"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — EDA OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        # Lead time by factory
        lt_factory = df.groupby('Factory')['Lead Time'].mean().reset_index()
        lt_factory.columns = ['Factory', 'Avg Lead Time']
        lt_factory = lt_factory.sort_values('Avg Lead Time', ascending=True)
        fig1 = px.bar(lt_factory, x='Avg Lead Time', y='Factory',
                      orientation='h', color='Avg Lead Time',
                      color_continuous_scale='Reds',
                      title="📍 Avg Lead Time by Factory",
                      labels={'Avg Lead Time': 'Days'})
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Lead time by ship mode
        lt_ship = df.groupby('Ship Mode')['Lead Time'].mean().reset_index()
        fig2 = px.bar(lt_ship, x='Ship Mode', y='Lead Time',
                      color='Ship Mode',
                      title="🚚 Avg Lead Time by Ship Mode",
                      labels={'Lead Time': 'Days'})
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Profit margin by division
        div_profit = df.groupby('Division')['Profit Margin %'].mean().reset_index()
        fig3 = px.pie(div_profit, values='Profit Margin %', names='Division',
                      title="💰 Avg Profit Margin by Division",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Distance vs Lead Time scatter
        sample = df.dropna(subset=['Shipping Distance','Lead Time']).sample(
            min(2000, len(df)), random_state=42)
        fig4 = px.scatter(sample, x='Shipping Distance', y='Lead Time',
                          color='Ship Mode', opacity=0.5,
                          title="📏 Shipping Distance vs Lead Time",
                          labels={'Shipping Distance': 'Miles', 'Lead Time': 'Days'})
        st.plotly_chart(fig4, use_container_width=True)

    # Heatmap: Region × Factory
    st.subheader("🔥 Lead Time Heatmap: Region × Factory")
    pivot = df.pivot_table('Lead Time', 'Region', 'Factory', aggfunc='mean').round(1)
    fig5 = px.imshow(pivot, text_auto=True, color_continuous_scale='YlOrRd',
                     title="Avg Lead Time (days) by Region and Factory")
    st.plotly_chart(fig5, use_container_width=True)

    # Top products by sales
    st.subheader("🏆 Top Products by Total Sales")
    top_prod = df.groupby('Product Name')['Sales'].sum().sort_values(
        ascending=False).reset_index()
    fig6 = px.bar(top_prod, x='Sales', y='Product Name', orientation='h',
                  color='Sales', color_continuous_scale='Blues',
                  labels={'Sales': 'Total Sales ($)'})
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 ML Model Comparison — Predicting Lead Time")
    st.info(f"**Best Model Selected:** `{best_name}` — used for all simulations and recommendations")

    metrics_data = []
    for name, res in model_results.items():
        metrics_data.append({
            'Model': name,
            'RMSE (↓ better)': res['RMSE'],
            'MAE (↓ better)':  res['MAE'],
            'R² (↑ better)':   res['R2'],
            'Status': '✅ Best' if name == best_name else ''
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    fig_rmse = px.bar(metrics_df, x='Model', y='RMSE (↓ better)',
                      color='Model', title="RMSE Comparison")
    fig_mae  = px.bar(metrics_df, x='Model', y='MAE (↓ better)',
                      color='Model', title="MAE Comparison")
    fig_r2   = px.bar(metrics_df, x='Model', y='R² (↑ better)',
                      color='Model', title="R² Comparison")
    c1.plotly_chart(fig_rmse, use_container_width=True)
    c2.plotly_chart(fig_mae,  use_container_width=True)
    c3.plotly_chart(fig_r2,   use_container_width=True)

    st.subheader("📌 KPI Summary")
    best_r2   = model_results[best_name]['R2']
    best_mae  = model_results[best_name]['MAE']
    best_rmse = model_results[best_name]['RMSE']
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy (R²)",  f"{best_r2:.3f}")
    m2.metric("Avg Prediction Error", f"{best_mae:.2f} days")
    m3.metric("RMSE",                 f"{best_rmse:.2f} days")

# ══════════════════════════════════════════════════════════
# TAB 3 — FACTORY SIMULATOR
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔮 What-If Factory Simulator")
    st.write("Select a product, region, and ship mode to simulate predicted lead time across all 5 factories.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_product = st.selectbox("🍫 Product",
                                   sorted(PRODUCT_FACTORY_MAP.keys()), key="sim_prod")
    with col2:
        sel_region = st.selectbox("🌎 Region",
                                  sorted(df['Region'].dropna().unique()), key="sim_reg")
    with col3:
        sel_ship = st.selectbox("🚚 Ship Mode",
                                sorted(df['Ship Mode'].dropna().unique()), key="sim_ship")

    # Get actuals for selected product
    subset = df[df['Product Name'] == sel_product]
    avg_sales  = subset['Sales'].mean()
    avg_units  = subset['Units'].mean()
    avg_cost   = subset['Cost'].mean()
    avg_margin = subset['Profit Margin %'].mean()
    current_factory = PRODUCT_FACTORY_MAP[sel_product]
    current_lt = subset[subset['Region'] == sel_region]['Lead Time'].mean() \
                 if sel_region in subset['Region'].values else subset['Lead Time'].mean()

    sim_results = simulate_reassignment(
        best_model, le_dict, feat_cols,
        sel_product, sel_region, sel_ship,
        avg_sales, avg_units, avg_cost, avg_margin)

    # Highlight current factory
    sim_results['Status'] = sim_results['Factory'].apply(
        lambda f: '⭐ Current' if f == current_factory else '🔄 Alternative')

    fig_sim = px.bar(sim_results, x='Factory', y='Predicted Lead Time',
                     color='Status',
                     color_discrete_map={'⭐ Current': '#f4a261', '🔄 Alternative': '#2a9d8f'},
                     title=f"Predicted Lead Time: {sel_product} across All Factories",
                     labels={'Predicted Lead Time': 'Days'})
    fig_sim.add_hline(y=current_lt, line_dash='dot', line_color='red',
                      annotation_text=f"Actual Avg: {current_lt:.1f} days")
    st.plotly_chart(fig_sim, use_container_width=True)

    best_alt = sim_results[sim_results['Factory'] != current_factory].iloc[0]
    saving = current_lt - best_alt['Predicted Lead Time']

    i1, i2, i3 = st.columns(3)
    i1.metric("⭐ Current Factory",    current_factory)
    i2.metric("🏆 Best Alternative",   best_alt['Factory'])
    i3.metric("⏱ Potential Lead Time Saving",
              f"{saving:.1f} days" if saving > 0 else "Already Optimal",
              delta=f"{-saving:.1f}" if saving > 0 else None)

    st.dataframe(sim_results[['Factory','Predicted Lead Time','Status']],
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Factory Reassignment Recommendations")

    if rec_df.empty:
        st.success("✅ All products are currently assigned to their optimal factories!")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("🔁 Total Reassignments Suggested", len(rec_df))
        r2.metric("📈 Avg Lead Time Improvement",
                  f"{rec_df['Improvement %'].mean():.1f}%")
        r3.metric("⏱ Max LT Saving",
                  f"{rec_df['LT Improvement (days)'].max():.1f} days")

        # Filter controls
        st.write("**Filter Recommendations:**")
        fc1, fc2 = st.columns(2)
        with fc1:
            filter_product = st.multiselect("Product Filter",
                                            rec_df['Product'].unique(),
                                            default=list(rec_df['Product'].unique()))
        with fc2:
            filter_region = st.multiselect("Region Filter",
                                           rec_df['Region'].unique(),
                                           default=list(rec_df['Region'].unique()))

        filtered = rec_df[
            rec_df['Product'].isin(filter_product) &
            rec_df['Region'].isin(filter_region)
        ]

        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Top 10 chart
        top10 = filtered.head(10)
        fig_rec = px.bar(top10, x='Improvement %',
                         y=top10['Product'] + " → " + top10['Region'],
                         orientation='h', color='Improvement %',
                         color_continuous_scale='Greens',
                         title="🏆 Top 10 Reassignment Opportunities by Lead Time Improvement %")
        st.plotly_chart(fig_rec, use_container_width=True)

        # Factory flow
        st.subheader("🔄 Reassignment Flow: Current → Recommended Factory")
        flow_df = filtered.groupby(
            ['Current Factory','Recommended Factory']).size().reset_index(name='Count')
        fig_flow = px.bar(flow_df, x='Current Factory', y='Count',
                          color='Recommended Factory',
                          barmode='group',
                          title="Volume of Suggested Reassignments by Factory Pair")
        st.plotly_chart(fig_flow, use_container_width=True)

        # Download button
        csv_out = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Recommendations CSV",
                           data=csv_out,
                           file_name="nassau_candy_recommendations.csv",
                           mime='text/csv')

# ══════════════════════════════════════════════════════════
# TAB 5 — RISK & IMPACT PANEL
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("⚠️ Risk & Profit Impact Panel")

    # Profit margin by factory
    profit_factory = df.groupby('Factory')['Profit Margin %'].mean().reset_index()
    fig_p1 = px.bar(profit_factory, x='Factory', y='Profit Margin %',
                    color='Profit Margin %', color_continuous_scale='RdYlGn',
                    title="💰 Avg Profit Margin % by Factory")
    st.plotly_chart(fig_p1, use_container_width=True)

    if not rec_df.empty:
        # High risk: low improvement %
        high_risk = rec_df[rec_df['Improvement %'] < 5]
        low_risk  = rec_df[rec_df['Improvement %'] >= 10]

        rk1, rk2, rk3 = st.columns(3)
        rk1.metric("🟢 High-Confidence Reassignments (≥10%)", len(low_risk))
        rk2.metric("🟡 Marginal Reassignments (<5%)",         len(high_risk))
        rk3.metric("🔴 Model Confidence (R²)",
                   f"{model_results[best_name]['R2']:.3f}")

        if not high_risk.empty:
            st.warning(f"⚠️ {len(high_risk)} reassignments have <5% improvement — review carefully before actioning.")
            st.dataframe(high_risk, use_container_width=True, hide_index=True)

        # Improvement distribution
        fig_dist = px.histogram(rec_df, x='Improvement %', nbins=20,
                                color_discrete_sequence=['#2a9d8f'],
                                title="Distribution of Lead Time Improvement % Across All Recommendations")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Profit by product
        st.subheader("💵 Sales & Profit by Product")
        prod_summary = df.groupby('Product Name').agg(
            Total_Sales=('Sales','sum'),
            Total_Profit=('Gross Profit','sum'),
            Avg_Margin=('Profit Margin %','mean')
        ).reset_index().sort_values('Total_Sales', ascending=False)

        fig_prod = px.scatter(prod_summary, x='Total_Sales', y='Avg_Margin',
                              size='Total_Profit', color='Product Name',
                              hover_name='Product Name',
                              title="Product: Sales vs Margin (bubble = Gross Profit)")
        st.plotly_chart(fig_prod, use_container_width=True)

    # Route Clustering
    st.subheader("🗺️ Route Cluster Analysis")
    cluster_df = df.groupby(['Factory','Region']).agg(
        Avg_Lead_Time=('Lead Time','mean'),
        Avg_Profit=('Profit Margin %','mean'),
        Order_Count=('Row ID','count')
    ).reset_index().dropna()

    if len(cluster_df) >= 4:
        scaler = StandardScaler()
        X_clust = scaler.fit_transform(
            cluster_df[['Avg_Lead_Time','Avg_Profit']])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_df['Cluster'] = kmeans.fit_predict(X_clust).astype(str)

        fig_clust = px.scatter(cluster_df,
                               x='Avg_Lead_Time', y='Avg_Profit',
                               color='Cluster', size='Order_Count',
                               hover_data=['Factory','Region'],
                               title="Route Clusters: Lead Time vs Profit Margin",
                               labels={'Avg_Lead_Time':'Avg Lead Time (days)',
                                       'Avg_Profit':'Avg Profit Margin %'})
        st.plotly_chart(fig_clust, use_container_width=True)
        st.caption("Each bubble = a Factory–Region route. Size = order volume. "
                   "High lead time + low profit clusters = priority for reassignment.")

st.divider()
st.caption(f"🍬 Nassau Candy Distributor | File: `{FILE_NAME}` | "
           f"Best Model: `{best_name}` | R²: {model_results[best_name]['R2']:.3f}")
