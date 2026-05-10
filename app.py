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
# CSS
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
# LOAD DATA
# ====================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "dulieutro.csv"
    )

    return df

# ====================================
# LOAD MODEL
# ====================================
@st.cache_resource
def load_model():

    model = joblib.load(
        "model_real.pkl"
    )

    return model

# ====================================
# LOAD
# ====================================
df = load_data()

model = load_model()

# ====================================
# CHECK REQUIRED COLUMNS
# ====================================
required_cols = [
    "quan",
    "phuong",
    "latitude",
    "longitude"
]

if not all(
    col in df.columns
    for col in required_cols
):

    st.error(
        "❌ Dataset thiếu cột bắt buộc"
    )

    st.stop()

# ====================================
# CREATE LOCATION
# ====================================
df["dia_diem"] = (
    df["quan"].astype(str)
    + " - "
    + df["phuong"].astype(str)
)

dia_diem_list = sorted(
    df["dia_diem"].unique()
)

# ====================================
# CAMPUS UEH
# ====================================
campus_options = {

    "UEH A - Nguyễn Đình Chiểu": (
        10.779785,
        106.698372
    ),

    "UEH B - Nguyễn Tri Phương": (
        10.762622,
        106.682231
    ),

    "UEH C - Trần Hữu Trang": (
        10.797800,
        106.676200
    ),

    "UEH D - Nguyễn Thị Minh Khai": (
        10.771900,
        106.693800
    ),

    "UEH E - Võ Thị Sáu": (
        10.779100,
        106.693000
    )
}

# ====================================
# HAVERSINE
# ====================================
def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return R * c

# ====================================
# TITLE
# ====================================
st.title(
    "🏠 AI Dự Đoán Giá Thuê Trọ Sinh Viên UEH"
)

st.markdown("---")

# ====================================
# LAYOUT
# ====================================
col1, col2 = st.columns([1,1])

# ====================================
# INPUT
# ====================================
with col1:

    st.subheader(
        "📝 Thông tin phòng trọ"
    )

    selected_place = st.selectbox(
        "📍 Chọn khu vực",
        dia_diem_list
    )

    selected_rows = df[
        df["dia_diem"]
        == selected_place
    ]

    row = selected_rows.iloc[0]

    quan = row["quan"]

    phuong = row["phuong"]

    latitude = row["latitude"]

    longitude = row["longitude"]

    st.info(f"""
📌 Quận: {quan}

📍 Phường: {phuong}
""")

    dien_tich = st.slider(
        "📐 Diện tích phòng (m²)",
        8,
        60,
        20
    )

    st.markdown(
        "### ⚙️ Tiện ích"
    )

    may_lanh = st.checkbox(
        "Máy lạnh"
    )

    tu_lanh = st.checkbox(
        "Tủ lạnh"
    )

    wc_rieng = st.checkbox(
        "WC riêng"
    )

    ban_cong = st.checkbox(
        "Ban công"
    )

    co_bep = st.checkbox(
        "Có bếp"
    )

    cho_de_xe = st.checkbox(
        "Chỗ để xe"
    )

    gan_sieu_thi = st.checkbox(
        "Gần siêu thị"
    )

    gan_quan_an = st.checkbox(
        "Gần quán ăn"
    )

    gan_truong_hoc = st.checkbox(
        "Gần trường học"
    )

    gio_tu_do = st.checkbox(
        "Giờ tự do"
    )

    st.markdown(
        "### 🏫 Campus UEH"
    )

    selected_campus = st.selectbox(
        "Chọn campus",
        list(campus_options.keys())
    )

# ====================================
# DISTANCE
# ====================================
ueh_lat, ueh_lon = campus_options[
    selected_campus
]

khoang_cach = haversine(
    latitude,
    longitude,
    ueh_lat,
    ueh_lon
)

# ====================================
# COUNT UTILITIES
# ====================================
so_tien_ich = (

    int(may_lanh)

    + int(tu_lanh)

    + int(wc_rieng)

    + int(ban_cong)

    + int(co_bep)

    + int(cho_de_xe)

    + int(gan_sieu_thi)

    + int(gan_quan_an)

    + int(gan_truong_hoc)

    + int(gio_tu_do)

)

# ====================================
# MODEL INPUT
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

    st.subheader(
        "🔮 Kết quả dự đoán"
    )

    if st.button(
        "💰 Dự đoán giá thuê"
    ):

        try:

            # ====================================
            # AI PREDICT
            # ====================================
            price = model.predict(
                input_data
            )[0]

            # ====================================
            # AREA BOOST
            # ====================================
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

            # ====================================
            # UTILITIES BOOST
            # ====================================
            price += (
                so_tien_ich * 180_000
            )

            # ====================================
            # DISTANCE BOOST
            # ====================================
            if khoang_cach < 1:

                price += 800_000

            elif khoang_cach < 3:

                price += 400_000

            # ====================================
            # CLIP
            # ====================================
            price = np.clip(
                price,
                2_000_000,
                12_000_000
            )

            # ====================================
            # DISPLAY
            # ====================================
            st.success(f"""

# 💵 {price/1_000_000:.2f} triệu VNĐ/tháng

""")

            st.info(f"""
📍 Khoảng cách đến campus:

{khoang_cach:.2f} km
""")

            # ====================================
            # PRICE LEVEL
            # ====================================
            if price < 3_000_000:

                st.warning(
                    "💡 Mức giá rẻ"
                )

            elif price < 6_000_000:

                st.info(
                    "✨ Mức giá trung bình"
                )

            else:

                st.error(
                    "🔥 Mức giá cao"
                )

            # ====================================
            # MAP
            # ====================================
            st.subheader(
                "🗺️ Bản đồ"
            )

            map_df = pd.DataFrame({

                "lat": [
                    latitude,
                    ueh_lat
                ],

                "lon": [
                    longitude,
                    ueh_lon
                ]

            })

            st.map(map_df)

        except Exception as e:

            st.error(
                f"❌ Lỗi dự đoán: {e}"
            )

# ====================================
# HEATMAP
# ====================================
st.markdown("---")

st.subheader(
    "🔥 Heatmap giá thuê"
)

if {
    "latitude",
    "longitude",
    "price"
}.issubset(df.columns):

    df_heat = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "price"
        ]
    ).copy()

    df_heat["price"] = pd.to_numeric(
        df_heat["price"],
        errors="coerce"
    )

    df_heat = df_heat.dropna(
        subset=["price"]
    )

    if len(df_heat) >= 5:

        heat_layer = pdk.Layer(
            "HeatmapLayer",

            data=df_heat,

            get_position='[longitude, latitude]',

            get_weight="price",

            radiusPixels=60,
        )

        view_state = pdk.ViewState(
            latitude=10.77,
            longitude=106.68,
            zoom=11,
            pitch=40,
        )

        deck = pdk.Deck(
            layers=[heat_layer],
            initial_view_state=view_state,
        )

        st.pydeck_chart(deck)

# ====================================
# DATA PREVIEW
# ====================================
with st.expander(
    "📄 Xem dữ liệu mẫu"
):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# ====================================
# FOOTER
# ====================================
st.markdown("---")

st.markdown("""
<center>

<h4>
🎓 Ứng dụng AI dự đoán giá thuê trọ sinh viên UEH
</h4>

</center>
""", unsafe_allow_html=True)