# ====================================
# IMPORT
# ====================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ====================================
# LOAD DATA
# ====================================
df = pd.read_csv("dulieutro.csv")

print("📌 Rows gốc:", len(df))

# ====================================
# CHECK REQUIRED COLUMNS
# ====================================
required_cols = [
    "price",
    "area",
    "quan",
    "phuong",
    "latitude",
    "longitude"
]

for col in required_cols:

    if col not in df.columns:

        print(f"❌ Thiếu cột: {col}")
        exit()

# ====================================
# CLEAN PRICE
# ====================================
def clean_price(x):

    try:

        if pd.isna(x):

            return np.nan

        value = float(x)

        # nếu là 4.5 => 4.5 triệu
        if value < 1000:

            value *= 1_000_000

        return value

    except:

        return np.nan

# ====================================
# CLEAN AREA
# ====================================
def clean_area(x):

    try:

        if pd.isna(x):

            return np.nan

        value = float(x)

        return value

    except:

        return np.nan

# ====================================
# APPLY CLEAN
# ====================================
df["price"] = df["price"].apply(
    clean_price
)

df["area"] = df["area"].apply(
    clean_area
)

# ====================================
# FIX AREA QUÁ LỚN
# ====================================
df.loc[
    df["area"] > 80,
    "area"
] = df["area"] / 4

# ====================================
# DROP NA
# ====================================
df = df.dropna(subset=[
    "price",
    "area",
    "quan",
    "phuong",
    "latitude",
    "longitude"
])

print("✅ Sau dropna:", len(df))

# ====================================
# FILTER
# ====================================

# lọc giá
df = df[
    (df["price"] >= 1_500_000)
    &
    (df["price"] <= 15_000_000)
]

# lọc diện tích
df = df[
    (df["area"] >= 8)
    &
    (df["area"] <= 80)
]

print("✅ Sau filter:", len(df))

# ====================================
# CHECK EMPTY
# ====================================
if len(df) == 0:

    print("❌ DATA RỖNG")
    exit()

# ====================================
# UTILITIES
# ====================================
utility_cols = [
    "may_lanh",
    "tu_lanh",
    "gio_tu_do",
    "wc_rieng",
    "ban_cong",
    "co_bep",
    "cho_de_xe",
    "gan_sieu_thi",
    "gan_quan_an",
    "gan_truong_hoc"
]

for col in utility_cols:

    if col not in df.columns:

        df[col] = 0

# ====================================
# TỔNG TIỆN ÍCH
# ====================================
df["so_tien_ich"] = (

    df["may_lanh"].astype(int)

    + df["tu_lanh"].astype(int)

    + df["gio_tu_do"].astype(int)

    + df["wc_rieng"].astype(int)

    + df["ban_cong"].astype(int)

    + df["co_bep"].astype(int)

    + df["cho_de_xe"].astype(int)

    + df["gan_sieu_thi"].astype(int)

    + df["gan_quan_an"].astype(int)

    + df["gan_truong_hoc"].astype(int)

)

# ====================================
# HAVERSINE
# ====================================
UEH_LAT = 10.779785
UEH_LON = 106.698372

def haversine(lat1, lon1, lat2, lon2):

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

df["khoang_cach"] = haversine(
    df["latitude"],
    df["longitude"],
    UEH_LAT,
    UEH_LON
)

# ====================================
# FEATURES
# ====================================
features = [

    "area",

    "so_tien_ich",

    "khoang_cach",

    "quan"

]

target = "price"

X = df[features]

y = df[target]

# ====================================
# SPLIT
# ====================================
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

print("✅ Train:", len(X_train))
print("✅ Test :", len(X_test))

# ====================================
# PREPROCESS
# ====================================
numeric_features = [

    "area",

    "so_tien_ich",

    "khoang_cach"

]

categorical_features = [

    "quan"

]

preprocessor = ColumnTransformer([

    (
        "num",
        StandardScaler(),
        numeric_features
    ),

    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_features
    )

])

# ====================================
# MODEL
# ====================================
model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "regressor",

        RandomForestRegressor(

            n_estimators=300,

            max_depth=12,

            random_state=42,

            min_samples_split=5,

            min_samples_leaf=2
        )
    )

])

# ====================================
# TRAIN
# ====================================
model.fit(
    X_train,
    y_train
)

print("\n✅ Train model thành công")

# ====================================
# PREDICT
# ====================================
y_pred = model.predict(X_test)

# ====================================
# METRICS
# ====================================
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n========== METRICS ==========")

print(f"MAE  : {mae:,.0f} VNĐ")

print(f"RMSE : {rmse:,.0f} VNĐ")

print(f"R²   : {r2:.4f}")

# ====================================
# SAVE MODEL
# ====================================
joblib.dump(
    model,
    "model_real.pkl"
)

print("\n💾 Đã lưu:")
print("model_real.pkl")