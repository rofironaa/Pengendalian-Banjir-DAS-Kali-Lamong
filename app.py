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
# 1. GOOGLE EARTH ENGINE AUTH (TANGGUH DENGAN BASE64)
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

# Fungsi Jembatan EE ke Folium
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name, overlay=True, control=True
    ).add_to(self)
folium.Map.add_ee_layer = add_ee_layer

# =========================================================
# 2. LOGIKA MODEL LSTM (DARI KODE 2)
# =========================================================
@st.cache_data
def generate_synthetic_data():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    rain = np.random.gamma(shape=1.5, scale=10, size=1000)
    discharge = 20 + (rain * 1.5) + np.random.normal(0, 5, 1000)
    return pd.DataFrame({'Date': dates, 'Rain_mm': rain, 'Discharge_m3s': discharge}).set_index('Date')

@st.cache_resource
def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Rain_mm', 'Discharge_m3s']])
    X, Y = [], []
    for i in range(3, len(scaled_data)):
        X.append(scaled_data[i-3:i, :])
        Y.append(scaled_data[i, 1])
    model = Sequential([LSTM(32, input_shape=(3, 2)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(Y), epochs=5, verbose=0)
    return model, scaler

# Load Data
df = generate_synthetic_data()
model, scaler = train_lstm_model(df)

# =========================================================
# 3. DASHBOARD UI
# =========================================================
st.set_page_config(layout="wide", page_title="Kali Lamong Flood EWS")
st.title("🌊 Sistem Peringatan Dini Banjir DAS Kali Lamong")

# Load GeoJSON
with open("DAS.geojson") as f:
    geojson_data = json.load(f)
aoi = ee.Geometry(geojson_data['features'][0]['geometry'])
centroid = aoi.centroid().coordinates().getInfo()
lon, lat = centroid

# Prediksi & Status
last_3 = scaler.transform(df.tail(3)[['Rain_mm', 'Discharge_m3s']])
pred_discharge = scaler.inverse_transform([[0, model.predict(np.array([last_3]))[0][0]]])[0, 1]

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📊 Analisis Data")
    st.metric("Prediksi Debit Besok", f"{pred_discharge:.2f} m³/s")
    status = "AMAN" if pred_discharge < 35 else "WASPADA" if pred_discharge < 50 else "BAHAYA"
    st.markdown(f"### Status: {status}")
    st.line_chart(df[['Discharge_m3s']].tail(30))

with col2:
    st.subheader("🛰 Validasi Spasial")
    # (Proses Sentinel-1 & Map Rendering sesuai Kode 2 Anda)
    # [Tambahkan logika pemrosesan S1 dan Rendering Map di sini]
    st.info("Peta sedang dimuat...")