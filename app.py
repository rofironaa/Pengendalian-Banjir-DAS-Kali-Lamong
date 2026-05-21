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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1. GOOGLE EARTH ENGINE AUTH
# =========================================================
def initialize_gee():
    try:
        ee.data.getAssetRoots()
        return
    except:
        pass
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS" in st.secrets:
            b64_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]["base64_json"]
            json_data = base64.b64decode(b64_json).decode('utf-8')
            creds = json.loads(json_data)
            json_path = "service_account.json"
            with open(json_path, "w") as f:
                f.write(json_data)
            credentials = ee.ServiceAccountCredentials(creds['client_email'], json_path)
            ee.Initialize(credentials=credentials)
        else:
            ee.Initialize()
    except Exception as e:
        st.error(f"Error Inisialisasi GEE: {e}")
        st.stop()

initialize_gee()

def add_ee_layer(self, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; Google Earth Engine',
            name=name, overlay=True, control=True
        ).add_to(self)
    except: pass
folium.Map.add_ee_layer = add_ee_layer

# =========================================================
# 2. DATA & MODEL LSTM
# =========================================================
@st.cache_data
def generate_synthetic_data():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    rain = np.random.gamma(shape=1.5, scale=10, size=1000)
    discharge = 20 + (rain * 1.5) + np.random.normal(0, 5, 1000)
    return pd.DataFrame({'Rain_mm': rain, 'Discharge_m3s': discharge}, index=dates)

@st.cache_resource
def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(3, len(scaled)):
        X.append(scaled[i-3:i, :]); Y.append(scaled[i, 1])
    model = Sequential([LSTM(32, input_shape=(3, 2)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(Y), epochs=5, verbose=0)
    return model, scaler

df = generate_synthetic_data()
model, scaler = train_lstm_model(df)

# =========================================================
# 3. DASHBOARD UI
# =========================================================
st.set_page_config(layout="wide", page_title="Kali Lamong Flood EWS")
st.title("🌊 Sistem Peringatan Dini Banjir DAS Kali Lamong")

# Prediksi Logika
last_3 = scaler.transform(df.tail(3))
pred_val = model.predict(np.array([last_3]))[0][0]
pred_discharge = scaler.inverse_transform([[0, pred_val]])[0, 1]
status = "AMAN" if pred_discharge < 35 else "WASPADA" if pred_discharge < 50 else "BAHAYA"
color = "green" if status == "AMAN" else "orange" if status == "WASPADA" else "red"

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Analisis Data")
    st.metric("Prediksi Debit Besok", f"{pred_discharge:.2f} m³/s")
    st.markdown(f"### Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    st.line_chart(df[['Discharge_m3s']].tail(30))

with col2:
    st.subheader("🛰 Validasi Spasial")
    
    # Memuat data sekali saja untuk efisiensi
    if 'geojson' not in st.session_state:
        with open("DAS.geojson") as f: st.session_state.geojson = json.load(f)
    
    aoi = ee.Geometry(st.session_state.geojson['features'][0]['geometry'])
    
    # Pemrosesan GEE
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(start_date, end_date)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).select('VV').min().clip(aoi)
    
    allWater = s1.focal_median(30, 'circle', 'meters').lt(-16)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi)
    permanentWater = jrc.select('occurrence').gt(40).unmask(0)
    floodWater = allWater.And(permanentWater.Not())
    
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    slope = ee.Terrain.slope(dem)
    finalFlood = floodWater.updateMask(floodWater.gt(0).And(slope.lt(5)))
    
    # Rendering Map dengan 'key' yang stabil
    centroid = aoi.centroid().coordinates().getInfo()
    Map = folium.Map(location=[centroid[1], centroid[0]], zoom_start=11)
    
    batas_das = folium.GeoJson(st.session_state.geojson, name="Boundary DAS", 
                               style_function=lambda x: {'color': 'red', 'fill': False, 'weight': 3})
    batas_das.add_to(Map)
    
    Map.add_ee_layer(dem, {'min': 0, 'max': 100, 'palette': ['blue', 'green', 'yellow', 'orange', 'red']}, 'Elevasi (DEM)')
    Map.add_ee_layer(permanentWater.updateMask(permanentWater), {'palette': ['blue']}, 'Air Permanen')
    Map.add_ee_layer(finalFlood, {'palette': ['cyan']}, 'Genangan Banjir Baru')
    
    folium.LayerControl(collapsed=False).add_to(Map)
    
    # Key 'map_kali_lamong' menjaga kestabilan komponen saat interaksi
    st_folium(Map, width=800, height=500, key="map_kali_lamong")

st.markdown("---")
st.markdown("### 🔧 Teknologi: Streamlit | TensorFlow LSTM | GEE | Sentinel-1 SAR | Folium")