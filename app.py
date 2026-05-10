import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title=" TNT HOUSEPRICING",
    page_icon="🏠",
    layout="wide"
)

# ====================================
# CSS (GIỮ NGUYÊN)
# ====================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

h1,h2,h3,h4,h5,h6,p,label,div,span {
    color: #FFD700 !important;
    font-family: 'Segoe UI', sans-serif;
}

.stButton > button {
    background: linear-gradient(90deg,#FFD700,#FFA500);
    color: black;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    transform: scale(1.02);
}

div[data-baseweb="select"] * {
    color: black !important;
}

input {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# LOAD DATA (FIX DEPLOY BUG)
# ====================================
@st.cache_data
def load_data():
    df = pd.read_csv("dulieutro.csv")
    df.columns = df.columns.str.strip()
    return df

# ====================================
# LOAD MODEL
# ====================================
@st.cache_resource
def load_model():
    return joblib.load("model_real.pkl")

# ====================================
# INIT
# ====================================
df = load_data()
model = load_model()

# ====================================
# CHECK COLUMNS
# ====================================
required_cols = ["quan", "phuong", "latitude", "longitude"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"❌ Thiếu cột: {missing}")
    st.stop()

# ====================================
# SAFE LOCATION BUILD
# ====================================
df["quan"] = df["quan"].astype(str).fillna("Unknown")
df["phuong"] = df["phuong"].astype(str).fillna("Unknown")

df["dia_diem"] = (
    df["quan"].str.strip()
    + " - "
    + df["phuong"].str.strip()
)

dia_diem_list = sorted(df["dia_diem"].dropna().unique().tolist())

# ====================================
# CAMPUS
# ====================================
campus_options = {
    "UEH A - Nguyễn Đình Chiểu": (10.779785, 106.698372),
    "UEH B - Nguyễn Tri Phương": (10.762622, 106.682231),
    "UEH C - Trần Hữu Trang": (10.797800, 106.676200),
    "UEH D - Nguyễn Thị Minh Khai": (10.771900, 106.693800),
    "UEH E - Võ Thị Sáu": (10.779100, 106.693000)
}

# ====================================
# HAVERSINE
# ====================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ====================================
# TITLE
# ====================================
st.title("🏠 AI Dự Đoán Giá Thuê Trọ Sinh Viên UEH")

st.markdown("---")

# ====================================
# LAYOUT
# ====================================
col1, col2 = st.columns([1,1])

# ====================================
# INPUT
# ====================================
with col1:

    st.subheader("📝 Thông tin phòng trọ")

    selected_place = st.selectbox("📍 Chọn khu vực", dia_diem_list)

    selected_rows = df[df["dia_diem"] == selected_place]

    if selected_rows.empty:
        st.error("❌ Không tìm thấy dữ liệu khu vực này")
        st.stop()

    row = selected_rows.iloc[0]

    quan = row["quan"]
    phuong = row["phuong"]
    latitude = row["latitude"]
    longitude = row["longitude"]

    st.info(f"""
📌 Quận: {quan}
📍 Phường: {phuong}
""")

    dien_tich = st.slider("📐 Diện tích phòng (m²)", 8, 60, 20)

    st.markdown("### ⚙️ Tiện ích")

    may_lanh = st.checkbox("Máy lạnh")
    tu_lanh = st.checkbox("Tủ lạnh")
    wc_rieng = st.checkbox("WC riêng")
    ban_cong = st.checkbox("Ban công")
    co_bep = st.checkbox("Có bếp")
    cho_de_xe = st.checkbox("Chỗ để xe")
    gan_sieu_thi = st.checkbox("Gần siêu thị")
    gan_quan_an = st.checkbox("Gần quán ăn")
    gan_truong_hoc = st.checkbox("Gần trường học")
    gio_tu_do = st.checkbox("Giờ tự do")

    selected_campus = st.selectbox("🏫 Campus UEH", list(campus_options.keys()))

# ====================================
# DISTANCE
# ====================================
ueh_lat, ueh_lon = campus_options[selected_campus]

khoang_cach = haversine(latitude, longitude, ueh_lat, ueh_lon)

# ====================================
# UTILITIES
# ====================================
so_tien_ich = (
    int(may_lanh) + int(tu_lanh) + int(wc_rieng)
    + int(ban_cong) + int(co_bep) + int(cho_de_xe)
    + int(gan_sieu_thi) + int(gan_quan_an)
    + int(gan_truong_hoc) + int(gio_tu_do)
)

# ====================================
# INPUT MODEL
# ====================================
input_data = pd.DataFrame([{
    "area": dien_tich,
    "so_tien_ich": so_tien_ich,
    "khoang_cach": khoang_cach,
    "quan": quan
}])

# ====================================
# PREDICT
# ====================================
with col2:

    st.subheader("🔮 Kết quả dự đoán")

    if st.button("💰 Dự đoán giá thuê"):

        try:

            price = model.predict(input_data)[0]

            # AREA BOOST (giữ logic của bạn)
            if dien_tich <= 15:
                price += 0
            elif dien_tich <= 20:
                price += 300_000
            elif dien_tich <= 25:
                price += 700_000
            elif dien_tich <= 30:
                price += 1_200_000
            elif dien_tich <= 40:
                price += 2_000_000
            else:
                price += 3_000_000

            # UTILITIES BOOST
            price += so_tien_ich * 180_000

            # DISTANCE BOOST
            if khoang_cach < 1:
                price += 800_000
            elif khoang_cach < 3:
                price += 400_000

            # CLIP
            price = np.clip(price, 2_000_000, 12_000_000)

            st.success(f"# 💵 {price/1_000_000:.2f} triệu VNĐ/tháng")

            st.info(f"📍 Khoảng cách: {khoang_cach:.2f} km")

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

# ====================================
# MAP
# ====================================
st.subheader("🗺️ Bản đồ")

st.map(pd.DataFrame({
    "lat": [latitude, ueh_lat],
    "lon": [longitude, ueh_lon]
}))

# ====================================
# HEATMAP SAFE
# ====================================
st.markdown("---")

st.subheader("🔥 Heatmap giá thuê")

if {"latitude","longitude","price"}.issubset(df.columns):

    df_heat = df.dropna(subset=["latitude","longitude","price"]).copy()
    df_heat["price"] = pd.to_numeric(df_heat["price"], errors="coerce")
    df_heat = df_heat.dropna(subset=["price"])

    if len(df_heat) > 5:

        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=df_heat,
            get_position='[longitude, latitude]',
            get_weight="price",
            radiusPixels=60,
        )

        deck = pdk.Deck(
            layers=[heat_layer],
            initial_view_state=pdk.ViewState(
                latitude=10.77,
                longitude=106.68,
                zoom=11,
                pitch=40
            )
        )

        st.pydeck_chart(deck)

# ====================================
# DATA PREVIEW
# ====================================
with st.expander("📄 Xem dữ liệu"):
    st.dataframe(df.head())