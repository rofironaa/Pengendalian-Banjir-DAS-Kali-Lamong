import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import ee
import json
from datetime import datetime, timedelta

# PERUBAHAN UTAMA: Gunakan Folium murni, hapus geemap
import folium
from streamlit_folium import st_folium 

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

# Tambahkan fungsi ini ke class folium.Map
folium.Map.add_ee_layer = add_ee_layer

# =========================================================
# 1. STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Kali Lamong Flood Early Warning System"
)

st.title("🌊 Sistem Peringatan Dini Banjir DAS Kali Lamong")
st.markdown("Prediksi debit sungai & validasi genangan berbasis Sentinel-1")

# =========================================================
# 2. GOOGLE EARTH ENGINE INIT
# =========================================================
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# =========================================================
# 3. LOAD AOI DARI GEOJSON DAS
# =========================================================
with open("DAS.geojson") as f:
    geojson_data = json.load(f)

# Convert GeoJSON menjadi Earth Engine Geometry
aoi = ee.Geometry(geojson_data['features'][0]['geometry'])

# Boundary DAS
das_boundary = ee.FeatureCollection(ee.Feature(aoi))

# Ambil centroid untuk center map otomatis
centroid = aoi.centroid().coordinates().getInfo()
lon, lat = centroid

# =========================================================
# 4. DATA DUMMY GENERATOR
# =========================================================
@st.cache_data
def generate_synthetic_data():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    rain = np.random.gamma(shape=1.5, scale=10, size=1000)
    discharge = 20 + (rain * 1.5) + np.random.normal(0, 5, 1000)
    return pd.DataFrame({'Date': dates, 'Rain_mm': rain, 'Discharge_m3s': discharge}).set_index('Date')

df = generate_synthetic_data()

# =========================================================
# 5. TRAINING LSTM MODEL
# =========================================================
@st.cache_resource
def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Rain_mm', 'Discharge_m3s']])
    X, Y = [], []
    for i in range(3, len(scaled_data)):
        X.append(scaled_data[i-3:i, :])
        Y.append(scaled_data[i, 1])
    X, Y = np.array(X), np.array(Y)

    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=5, batch_size=32, verbose=0)
    return model, scaler

model, scaler = train_lstm_model(df)

# =========================================================
# 6. PREDIKSI DEBIT BESOK
# =========================================================
last_3_days = df.tail(3)
scaled_last_3 = scaler.transform(last_3_days[['Rain_mm', 'Discharge_m3s']])
pred_input = np.array([scaled_last_3])
pred_scaled = model.predict(pred_input)

dummy_row = np.zeros((1, 2))
dummy_row[0, 1] = pred_scaled[0][0]
pred_discharge = scaler.inverse_transform(dummy_row)[0, 1]

# =========================================================
# 7. STATUS ALERT
# =========================================================
if pred_discharge < 35:
    status, color = "AMAN", "green"
elif pred_discharge < 50:
    status, color = "WASPADA", "orange"
else:
    status, color = "BAHAYA", "red"

# =========================================================
# 8. LAYOUT DASHBOARD
# =========================================================
col1, col2 = st.columns([1, 2])

# =========================================================
# 9. PANEL INFORMASI
# =========================================================
with col1:
    st.subheader("📊 Prediksi Debit")
    st.metric(label="Prediksi Debit Besok", value=f"{pred_discharge:.2f} m³/s")
    st.markdown(f"### Status:<br><span style='color:{color}; font-size:28px;'><b>{status}</b></span>", unsafe_allow_html=True)
    
    st.subheader("🌧 Data Historis")
    st.line_chart(df[['Rain_mm']].tail(30))
    st.subheader("🌊 Debit Historis")
    st.line_chart(df[['Discharge_m3s']].tail(30))

# =========================================================
# 10. VALIDASI SPASIAL FLOOD MAPPING (FOLIUM MURNI)
# =========================================================
with col2:
    st.subheader("🛰 Validasi Spasial: Deteksi Genangan")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

    # Proses Earth Engine (Tetap sama, kalkulasi dilakukan di server Google)
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(start_date, end_date)\
        .filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).select('VV')
    
    s1_latest = s1.min().clip(aoi)
    smoothed = s1_latest.focal_median(30, 'circle', 'meters')
    allWater = smoothed.lt(-16)
    
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi)
    permanentWater = jrc.select('occurrence').gt(40).unmask(0)
    floodWater = allWater.And(permanentWater.Not())
    
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    slope = ee.Terrain.slope(dem)
    finalFloodMask = floodWater.updateMask(floodWater.gt(0).And(slope.lt(5)))

    # =====================================================
    # MAP RENDER (DIREKAYASA ULANG TANPA GEEMAP)
    # =====================================================
    # Inisialisasi Peta
    Map = folium.Map(location=[lat, lon], tiles='OpenStreetMap')

    # 1. Simpan layer GeoJSON ke dalam variabel
    batas_das = folium.GeoJson(
        data=geojson_data,
        name="Boundary DAS",
        style_function=lambda x: {'fillColor': '#00000000', 'color': 'red', 'weight': 3}
    )
    
    # 2. Tambahkan garis batas DAS ke peta
    batas_das.add_to(Map)

    # 3. KUNCI PERBAIKAN: Paksa kamera peta untuk auto-zoom pas ke batas DAS
    Map.fit_bounds(batas_das.get_bounds())

    # Tambahkan Layer Earth Engine melalui jembatan kustom kita
    Map.add_ee_layer(dem, {'min': 0, 'max': 50, 'palette': ['blue', 'green', 'yellow', 'orange', 'red']}, 'DEM')
    Map.add_ee_layer(permanentWater.updateMask(permanentWater), {'palette': ['00008B']}, 'Badan Air Permanen')
    Map.add_ee_layer(finalFloodMask, {'palette': ['00FFFF']}, f'Genangan ({start_date} s/d {end_date})')

    # Tambahkan kontrol layer agar bisa di-ceklis/un-ceklis
    folium.LayerControl(collapsed=False).add_to(Map)

    # Render ke Streamlit
    st_folium(Map, width=800, height=700)

# =========================================================
# 11. FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
### 🔧 Teknologi
- Streamlit & Python
- TensorFlow LSTM
- Google Earth Engine (Direct API Injection)
- Folium Native Rendering
- Sentinel-1 SAR & JRC GSW
""")