import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import ee
import json
import os
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium 

# =========================================================
# 1. GOOGLE EARTH ENGINE AUTH & INIT (SERVICE ACCOUNT SAFE)
# =========================================================
def initialize_gee():
    # Cek apakah sudah terinisialisasi dengan mencoba mengambil info proyek
    try:
        ee.data.getAssetRoots()
        return # Sudah inisialisasi, lanjut ke proses berikutnya
    except:
        pass # Belum inisialisasi, lanjut ke blok try di bawah

    try:
        # Coba ambil kredensial dari Secrets (Cloud Deployment)
        if "GOOGLE_APPLICATION_CREDENTIALS" in st.secrets:
            creds = dict(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
            with open("service_account.json", "w") as f:
                json.dump(creds, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
            ee.Initialize()
        else:
            # Fallback untuk lingkungan lokal
            ee.Initialize()
    except Exception as e:
        st.error(f"Gagal inisialisasi Earth Engine: {e}")
        st.stop()

initialize_gee()

# =========================================================
# FUNGSI KUSTOM: Jembatan Earth Engine ke Folium Murni
# =========================================================
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

# =========================================================
# 2. STREAMLIT CONFIG & DATA LOAD
# =========================================================
st.set_page_config(layout="wide", page_title="Kali Lamong Flood Early Warning System")
st.title("🌊 Sistem Peringatan Dini Banjir DAS Kali Lamong")

# Load AOI
with open("DAS.geojson") as f:
    geojson_data = json.load(f)

aoi = ee.Geometry(geojson_data['features'][0]['geometry'])
centroid = aoi.centroid().coordinates().getInfo()
lon, lat = centroid

# Model & Data
@st.cache_data
def generate_synthetic_data():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    return pd.DataFrame({'Rain_mm': np.random.gamma(1.5, 10, 100), 'Discharge_m3s': 20 + np.random.rand(100)*30}).set_index(dates)

df = generate_synthetic_data()

# =========================================================
# 3. LAYOUT DASHBOARD
# =========================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Analisis Data")
    pred_discharge = df['Discharge_m3s'].iloc[-1]
    st.metric(label="Prediksi Debit Besok", value=f"{pred_discharge:.2f} m³/s")
    st.line_chart(df[['Discharge_m3s']].tail(30))

with col2:
    st.subheader("🛰 Validasi Spasial")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

    # Proses Earth Engine
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(start_date, end_date)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).select('VV').min().clip(aoi)
    
    floodMask = s1.focal_median(30, 'circle', 'meters').lt(-16)
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    finalFloodMask = floodMask.updateMask(floodMask.gt(0).And(ee.Terrain.slope(dem).lt(5)))

    # Map Render
    Map = folium.Map(location=[lat, lon], tiles='OpenStreetMap')
    batas_das = folium.GeoJson(geojson_data, name="Boundary DAS", 
                               style_function=lambda x: {'fillColor': '#00000000', 'color': 'red', 'weight': 3})
    batas_das.add_to(Map)
    Map.fit_bounds(batas_das.get_bounds())
    
    Map.add_ee_layer(dem, {'min': 0, 'max': 50, 'palette': ['blue', 'green', 'yellow']}, 'DEM')
    Map.add_ee_layer(finalFloodMask, {'palette': ['00FFFF']}, 'Genangan')
    folium.LayerControl(collapsed=False).add_to(Map)
    st_folium(Map, width=800, height=500)