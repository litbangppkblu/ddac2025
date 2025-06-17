import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import json
import folium
from streamlit_folium import st_folium
from difflib import get_close_matches

# Initialize session state
if "tab_selected" not in st.session_state:
    st.session_state["tab_selected"] = "Visualisasi"

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("DATASET 211024.xlsx", sheet_name="BENER")
    # Clean data
    df = df.dropna(subset=[
        "Provinsi", "Tahun", "Kesehatan + Pendidikan",
        "Output Layanan Kesehatan", "AHH", "Unmeet Need"
    ])
    # Apply log transformations
    df["ln_blj_kesehatan_pendidikan"] = np.log(df["Kesehatan + Pendidikan"])
    df["ln_output_layanan_kesehatan"] = np.log(df["Output Layanan Kesehatan"])
    return df

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("indonesia.geojson", "r", encoding="utf-8") as file:
        return json.load(file)

# Match province names
@st.cache_data
def match_province_names(df, geojson_data):
    # Extract province names from GeoJSON
    geojson_provinces = {feature["properties"]["state"] for feature in geojson_data["features"]}
    
    # Function to match province names using similarity
    def match_province(name, choices, threshold=0.6):
        name = name.upper().strip()
        match = get_close_matches(name, [p.upper() for p in choices], n=1, cutoff=threshold)
        if match:
            # Find the original case version
            idx = [p.upper() for p in choices].index(match[0])
            return list(choices)[idx]
        return name
    
    # Create a mapping dictionary
    province_mapping = {}
    for province in df["Provinsi"].unique():
        matched = match_province(province, geojson_provinces)
        province_mapping[province] = matched
    
    return province_mapping

# Regression model (dynamic) - cached to avoid recalculation
@st.cache_data
def get_dynamic_coefficients(df):
    X = df[["ln_blj_kesehatan_pendidikan", "ln_output_layanan_kesehatan"]]
    X = sm.add_constant(X)
    
    # AHH model
    model_ahh = sm.OLS(df["AHH"], X).fit()
    
    # Unmet Need model
    model_unmet = sm.OLS(df["Unmeet Need"], X).fit()
    
    # Return coefficients instead of model objects
    ahh_coeffs = model_ahh.params.tolist()
    unmet_coeffs = model_unmet.params.tolist()
    ahh_stats = {"rsquared": model_ahh.rsquared, "pvalue": model_ahh.f_pvalue}
    unmet_stats = {"rsquared": model_unmet.rsquared, "pvalue": model_unmet.f_pvalue}
    
    return ahh_coeffs, unmet_coeffs, ahh_stats, unmet_stats

# Function to get filtered dataframe - cached to avoid recalculation
@st.cache_data
def get_filtered_data(df, selected_provinces, selected_years):
    return df[df["Provinsi"].isin(selected_provinces) & df["Tahun"].isin(selected_years)]

# Create visualizations - moved to a function
def create_visualizations(filtered_df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Angka Harapan Hidup (AHH)**")
        fig_ahh = px.line(
            filtered_df, x="Tahun", y="AHH", color="Provinsi", 
            markers=True, line_shape="spline",
            title="Tren AHH per Provinsi"
        )
        fig_ahh.update_layout(height=400)
        st.plotly_chart(fig_ahh, use_container_width=True)
    
    with col2:
        st.markdown("**Unmet Need**")
        fig_unmet = px.line(
            filtered_df, x="Tahun", y="Unmeet Need", color="Provinsi", 
            markers=True, line_shape="spline",
            title="Tren Unmet Need per Provinsi"
        )
        fig_unmet.update_layout(height=400)
        st.plotly_chart(fig_unmet, use_container_width=True)
    
    # Correlation scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Korelasi AHH vs Belanja**")
        scatter1 = px.scatter(
            filtered_df, x="ln_blj_kesehatan_pendidikan", y="AHH",
            color="Provinsi", size="Output Layanan Kesehatan", 
            trendline="ols", hover_data=["Tahun"],
            labels={"ln_blj_kesehatan_pendidikan": "Log Belanja (Kes+Pend)"}
        )
        scatter1.update_layout(height=400)
        st.plotly_chart(scatter1, use_container_width=True)
    
    with col2:
        st.markdown("**Korelasi Unmet Need vs Belanja**")
        scatter2 = px.scatter(
            filtered_df, x="ln_blj_kesehatan_pendidikan", y="Unmeet Need",
            color="Provinsi", size="Output Layanan Kesehatan", 
            trendline="ols", hover_data=["Tahun"],
            labels={"ln_blj_kesehatan_pendidikan": "Log Belanja (Kes+Pend)"}
        )
        scatter2.update_layout(height=400)
        st.plotly_chart(scatter2, use_container_width=True)

# Create map - prep data for map (cacheable)
@st.cache_data
def prepare_map_data(df, choropleth_year, province_mapping):
    # Filter data for selected year
    df_map = df[df["Tahun"] == choropleth_year].copy()
    # Map the province names to match GeoJSON
    df_map["Mapped_Province"] = df_map["Provinsi"].map(province_mapping)
    return df_map

# Generate the folium map - not cached because folium objects aren't cacheable
def generate_folium_map(df_map, geojson_data, choropleth_year):
    # Create a folium map
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")
    
    # Create choropleth
    folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=df_map,
        columns=["Mapped_Province", "AHH"],
        key_on="feature.properties.state",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"AHH - Tahun {choropleth_year}"
    ).add_to(m)
    
    # Add tooltips
    for feature in geojson_data["features"]:
        province_name = feature["properties"]["state"]
        province_data = df_map[df_map["Mapped_Province"] == province_name]
        
        if not province_data.empty:
            ahh_value = province_data["AHH"].values[0]
            unmet_value = province_data["Unmeet Need"].values[0]
            spending = province_data["Kesehatan + Pendidikan"].values[0] / 1e9  # Convert to billions
            
            tooltip_text = f"""
            <b>{province_name}</b><br>
            AHH: {ahh_value:.2f}<br>
            Unmet Need: {unmet_value:.2f}%<br>
            Belanja: {spending:.2f} Miliar Rp
            """
            
            folium.GeoJson(
                feature,
                tooltip=folium.Tooltip(tooltip_text, sticky=True),
                style_function=lambda x: {"weight": 0.5, "fillOpacity": 0}
            ).add_to(m)
    
    return m

def create_map(df, choropleth_year, geojson_data, province_mapping):
    # Use cacheable prep function
    df_map = prepare_map_data(df, choropleth_year, province_mapping)
    
    # Generate map (not cached)
    m = generate_folium_map(df_map, geojson_data, choropleth_year)
    
    # Display the map in Streamlit
    st_folium(m, width=800, height=500, key=f"map_{choropleth_year}")
    
    # Display data table below map
    st.markdown("### Data Provinsi")
    st.dataframe(
        df_map[["Provinsi", "AHH", "Unmeet Need", "Kesehatan + Pendidikan", "Output Layanan Kesehatan"]]
        .sort_values("AHH", ascending=False)
    )

# Create simulation - moved to a function
def create_simulation(df, use_dynamic):
    # Get regression models
    if use_dynamic:
        ahh_coeffs, unmet_coeffs, ahh_stats, unmet_stats = get_dynamic_coefficients(df)
        
        # Display model statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model AHH")
            st.write(f"R²: {ahh_stats['rsquared']:.3f}")
            st.write(f"p-value: {ahh_stats['pvalue']:.5f}")
        
        with col2:
            st.markdown("#### Model Unmet Need")
            st.write(f"R²: {unmet_stats['rsquared']:.3f}")
            st.write(f"p-value: {unmet_stats['pvalue']:.5f}")
    else:
        # Static coefficients
        ahh_coeffs = [4.105, 0.00183, 0.00884]
        unmet_coeffs = [1.870, 0.0222, -0.0721]
    
    # Input sliders
    col1, col2 = st.columns(2)
    
    with col1:
        ln_spending = st.slider(
            "Log Belanja Kesehatan + Pendidikan", 
            float(df["ln_blj_kesehatan_pendidikan"].min()), 
            float(df["ln_blj_kesehatan_pendidikan"].max()), 
            float(df["ln_blj_kesehatan_pendidikan"].mean()),
            0.1,
            key="ln_spending"
        )
        
        spending_rp = np.exp(ln_spending) / 1e9  # Convert to billions
        st.write(f"Belanja: {spending_rp:.2f} Miliar Rupiah")
    
    with col2:
        ln_output = st.slider(
            "Log Output Layanan Kesehatan", 
            float(df["ln_output_layanan_kesehatan"].min()), 
            float(df["ln_output_layanan_kesehatan"].max()), 
            float(df["ln_output_layanan_kesehatan"].mean()),
            0.1,
            key="ln_output"
        )
        
        output_actual = np.exp(ln_output)
        st.write(f"Output: {output_actual:.0f} layanan")
    
    # Predictions
    # Using coefficients directly instead of model objects
    pred_ahh = ahh_coeffs[0] + ahh_coeffs[1] * ln_spending + ahh_coeffs[2] * ln_output
    pred_unmet = unmet_coeffs[0] + unmet_coeffs[1] * ln_spending + unmet_coeffs[2] * ln_output
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📈 Prediksi AHH", f"{pred_ahh:.2f} tahun")
        
        # Compare with national average
        nat_avg_ahh = df[df["Tahun"] == max(df["Tahun"])]["AHH"].mean()
        delta_ahh = pred_ahh - nat_avg_ahh
        st.metric("Dibanding rata-rata nasional", f"{delta_ahh:+.2f} tahun")
    
    with col2:
        st.metric("📉 Prediksi Unmet Need", f"{pred_unmet:.2f}%")
        
        # Compare with national average
        nat_avg_unmet = df[df["Tahun"] == max(df["Tahun"])]["Unmeet Need"].mean()
        delta_unmet = pred_unmet - nat_avg_unmet
        st.metric("Dibanding rata-rata nasional", f"{delta_unmet:+.2f}%")
    
    # Sensitivity analysis
    st.subheader("Analisis Sensitivitas")
    st.markdown("""
    Grafik di bawah ini menunjukkan bagaimana perubahan belanja kesehatan+pendidikan 
    dan output layanan kesehatan mempengaruhi AHH dan Unmet Need.
    """)
    
    # Create sensitivity data
    create_sensitivity_plots(df, ahh_coeffs, unmet_coeffs)

# Function to create sensitivity plots - cached to avoid recalculation
@st.cache_data
def create_sensitivity_data(df, ahh_coeffs, unmet_coeffs):
    # Create ranges for sensitivity analysis
    spending_range = np.linspace(
        df["ln_blj_kesehatan_pendidikan"].min(),
        df["ln_blj_kesehatan_pendidikan"].max(),
        10
    )
    
    output_range = np.linspace(
        df["ln_output_layanan_kesehatan"].min(),
        df["ln_output_layanan_kesehatan"].max(),
        10
    )
    
    # Create sensitivity data
    sensitivity_df = pd.DataFrame()
    
    for sp in spending_range:
        for out in output_range:
            # Calculate predictions using coefficients
            ahh_pred = ahh_coeffs[0] + ahh_coeffs[1] * sp + ahh_coeffs[2] * out
            unmet_pred = unmet_coeffs[0] + unmet_coeffs[1] * sp + unmet_coeffs[2] * out
            
            temp_df = pd.DataFrame({
                'Log Belanja': [sp],
                'Log Output': [out],
                'Belanja (Miliar)': [np.exp(sp) / 1e9],
                'Output': [np.exp(out)],
                'AHH': [ahh_pred],
                'Unmet Need': [unmet_pred]
            })
            sensitivity_df = pd.concat([sensitivity_df, temp_df], ignore_index=True)
    
    return sensitivity_df

def create_sensitivity_plots(df, ahh_coeffs, unmet_coeffs):
    # Get sensitivity data
    sensitivity_df = create_sensitivity_data(df, ahh_coeffs, unmet_coeffs)
    
    # Plot sensitivity analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sens1 = px.scatter_3d(
            sensitivity_df, x='Log Belanja', y='Log Output', z='AHH',
            color='AHH', title="Sensitivitas AHH",
            labels={'Log Belanja': 'Log Belanja (Kes+Pend)', 'Log Output': 'Log Output Layanan'}
        )
        st.plotly_chart(fig_sens1, use_container_width=True)
    
    with col2:
        fig_sens2 = px.scatter_3d(
            sensitivity_df, x='Log Belanja', y='Log Output', z='Unmet Need',
            color='Unmet Need', title="Sensitivitas Unmet Need",
            labels={'Log Belanja': 'Log Belanja (Kes+Pend)', 'Log Output': 'Log Output Layanan'}
        )
        st.plotly_chart(fig_sens2, use_container_width=True)

# MAIN APP EXECUTION

# App title and description
st.title("📊 Dashboard BLU: Dampak Belanja terhadap Kesehatan")
st.markdown("""
    Dashboard ini menampilkan analisis dampak belanja kesehatan dan pendidikan 
    terhadap angka harapan hidup (AHH) dan kebutuhan yang tidak terpenuhi (Unmeet Need) 
    di Indonesia.
""")

# Load data first (cached)
df = load_data()
geojson_data = load_geojson()
province_mapping = match_province_names(df, geojson_data)
provinces = df["Provinsi"].unique()
years = sorted(df["Tahun"].unique())

# Sidebar
st.sidebar.header("🧭 Filter")
selected_provinces = st.sidebar.multiselect("Pilih Provinsi", provinces, default=list(provinces[:3]))
selected_years = st.sidebar.multiselect("Pilih Tahun", years, default=years)
use_dynamic = st.sidebar.checkbox("Gunakan Koefisien Regresi Dinamis", value=True)

# Use session state to track active tab to prevent unnecessary reruns
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📈 Visualisasi"

# Tabs using radio buttons with key to prevent rerunning
tab_selected = st.radio(
    "Pilih Tab:", 
    ["📈 Visualisasi", "🌍 Peta", "🔢 Simulasi"], 
    horizontal=True,
    key="tab_selector",
    index=["📈 Visualisasi", "🌍 Peta", "🔢 Simulasi"].index(st.session_state.active_tab)
)

# Update the active tab in session state
st.session_state.active_tab = tab_selected

# Get filtered data (cached)
filtered_df = get_filtered_data(df, selected_provinces, selected_years)

# Display content based on selected tab
if tab_selected == "📈 Visualisasi":
    st.subheader("Visualisasi Tren dan Korelasi")
    create_visualizations(filtered_df)
elif tab_selected == "🌍 Peta":
    st.subheader("🌍 Peta Persebaran AHH Nasional")
    # Use a key for selectbox to maintain state
    # Store choropleth_year in session state to prevent unnecessary recomputation
    if "choropleth_year" not in st.session_state:
        st.session_state.choropleth_year = years[-1]  # Default to the latest year
        
    choropleth_year = st.selectbox(
        "Pilih Tahun untuk Peta", 
        years, 
        index=years.index(st.session_state.choropleth_year),
        key="map_year_selector",
        on_change=lambda: setattr(st.session_state, "choropleth_year", 
                                  st.session_state.map_year_selector)
    )
    create_map(df, choropleth_year, geojson_data, province_mapping)
elif tab_selected == "🔢 Simulasi":
    st.subheader("🎛️ Simulasi Dampak Belanja dan Output")
    st.markdown("""
    Gunakan slider di bawah untuk mensimulasikan nilai AHH dan Unmet Need
    berdasarkan model regresi yang dibangun dari data historis.
    """)
    create_simulation(df, use_dynamic)

st.divider()
st.caption("Dikembangkan untuk Direktorat PPKBLU DJPb – Pemantauan Dampak Belanja BLU Berbasis Data")
