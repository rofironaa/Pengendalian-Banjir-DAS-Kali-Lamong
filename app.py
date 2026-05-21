import streamlit as st
import numpy as np
import pandas as pd
import ee
import json
import base64
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1. INISIALISASI & AUTH
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
            json_path = "service_account.json"
            with open(json_path, "w") as f: f.write(json_data)
            ee.Initialize(credentials=ee.ServiceAccountCredentials(json.loads(json_data)['client_email'], json_path))
        else:
            ee.Initialize()
    except: ee.Authenticate(); ee.Initialize()

initialize_gee()

# Fungsi Jembatan EE ke Folium
def add_ee_layer(self, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine', name=name, overlay=True, control=True
        ).add_to(self)
    except: pass
folium.Map.add_ee_layer = add_ee_layer

# =========================================================
# 2. MODEL AI & DATA
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
    for i in range(3, len(scaled)): X.append(scaled[i-3:i, :]); Y.append(scaled[i, 1])
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

# Prediksi
last_3 = scaler.transform(df.tail(3))
pred_discharge = scaler.inverse_transform([[0, model.predict(np.array([last_3]))[0][0]]])[0, 1]
status, color = ("AMAN", "green") if pred_discharge < 35 else ("WASPADA", "orange") if pred_discharge < 50 else ("BAHAYA", "red")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Analisis Data")
    st.metric("Prediksi Debit Besok", f"{pred_discharge:.2f} m³/s")
    st.markdown(f"### Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Memastikan grafik tampil dengan pembungkus (expander) agar lebih rapi
    with st.expander("Lihat Data Historis (30 Hari)", expanded=True):
        st.write("🌧 **Curah Hujan**")
        if not df.empty:
            st.line_chart(df[['Rain_mm']].tail(30), height=250)
        else:
            st.warning("Data hujan tidak tersedia.")
            
        st.write("🌊 **Debit Sungai**")
        if not df.empty:
            st.line_chart(df[['Discharge_m3s']].tail(30), height=250)
        else:
            st.warning("Data debit tidak tersedia.")
            
    # Tambahan: Indikator data agar kita tahu sistem berjalan
    st.caption(f"Update terakhir: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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

st.markdown("---")
st.markdown("### 🔧 Teknologi: Streamlit | TensorFlow LSTM | GEE | Sentinel-1 SAR | Folium")