import streamlit as st
import numpy as np
import pandas as pd
import ee
import json
import os
import base64
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium 

# =========================================================
# 1. GOOGLE EARTH ENGINE AUTH & INIT (BASE64 DECODED)
# =========================================================
def initialize_gee():
    # Cek apakah sudah terinisialisasi
    try:
        ee.data.getAssetRoots()
        return
    except:
        pass

    try:
        # Baca dari Secrets
        if "GOOGLE_APPLICATION_CREDENTIALS" in st.secrets:
            # Ambil string base64
            b64_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]["base64_json"]
            
            # Decode kembali ke string JSON
            json_data = base64.b64decode(b64_json).decode('utf-8')
            creds = json.loads(json_data)
            
            # Simpan ke file sementara yang bersih
            json_path = "service_account.json"
            with open(json_path, "w") as f:
                f.write(json_data)
            
            # Gunakan ServiceAccountCredentials secara eksplisit
            credentials = ee.ServiceAccountCredentials(
                email=creds['client_email'], 
                key_file=json_path
            )
            ee.Initialize(credentials=credentials)
        else:
            # Mode dev lokal
            ee.Initialize()
    except Exception as e:
        st.error(f"Error Inisialisasi GEE: {e}")
        st.stop()

initialize_gee()

# =========================================================
# FUNGSI KUSTOM: Folium Renderer
# =========================================================
def add_ee_layer(self, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(self)
    except Exception as e:
        st.warning(f"Gagal memuat layer {name}: {e}")

folium.Map.add_ee_layer = add_ee_layer

# =========================================================
# 2. STREAMLIT CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="Kali Lamong Flood Early Warning System")
st.title("🌊 Sistem Peringatan Dini Banjir DAS Kali Lamong")

# Load Data dengan error handling
try:
    with open("DAS.geojson") as f:
        geojson_data = json.load(f)
    aoi = ee.Geometry(geojson_data['features'][0]['geometry'])
    centroid = aoi.centroid().coordinates().getInfo()
    lon, lat = centroid
except Exception as e:
    st.error("Gagal memuat file DAS.geojson. Pastikan file ada di repositori.")
    st.stop()

# =========================================================
# 3. DASHBOARD
# =========================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Analisis Data")
    val = 20 + np.random.rand()*30
    st.metric(label="Prediksi Debit Besok", value=f"{val:.2f} m³/s")
    st.line_chart(np.random.randn(30, 1))

with col2:
    st.subheader("🛰 Validasi Spasial")
    try:
        start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Sentinel-1 Processing
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV').min().clip(aoi)
        
        floodMask = s1.focal_median(30, 'circle', 'meters').lt(-16)
        
        # Rendering Map
        Map = folium.Map(location=[lat, lon], tiles='OpenStreetMap', zoom_start=11)
        batas_das = folium.GeoJson(geojson_data, name="Boundary DAS", 
                                   style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 3})
        batas_das.add_to(Map)
        
        Map.add_ee_layer(floodMask.updateMask(floodMask.gt(0)), {'palette': ['cyan']}, 'Genangan Air')
        folium.LayerControl().add_to(Map)
        
        st_folium(Map, width=800, height=500)
    except Exception as e:
        st.error(f"Error pada pemrosesan spasial: {e}")