# app.py — Final Streamlit app (full preprocessing, UI, graphs, XGBoost)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime
import math
import os

# -------------------------
# Config / Model path
# -------------------------
MODEL_PATH = "models/tuned_xgb_hour_model.joblib"   # your joblib model file
DATA_PATHS = ["hour.csv", "data/hour.csv", "data\\hour.csv"]  # try multiple locations

# -------------------------
# Load model (fail early)
# -------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model '{MODEL_PATH}': {e}")
    st.stop()

# -------------------------
# Load historical data (optional)
# -------------------------
df = None
for p in DATA_PATHS:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            if "dteday" in df.columns:
                df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
            break
        except Exception:
            df = None

# -------------------------
# Page config & consolidated CSS (fix top white bar)
# -------------------------
st.set_page_config(page_title="AI RideWise — Hourly Prediction", layout="wide")

st.markdown("""
<style>
/* keep existing dark theme */
.stApp { background-color: #0f1113 !important; color: #fff !important; }

/* pred card look (unchanged) */
.pred-card { background: linear-gradient(135deg,#e63946,#b81e2c); padding: 28px; border-radius: 16px; color: white; text-align:center; box-shadow: 0 10px 30px rgba(230,57,70,0.14); }

/* --- Strong rules to remove leftover white rounded "pills" directly below the pred-card --- */
/* Target any immediate sibling or following sibling elements after the pred-card */
.pred-card + *, .pred-card ~ * {
    background: transparent !important;
    box-shadow: none !important;
    border: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    height: auto !important;
    border-radius: 0 !important;
    overflow: hidden !important;
}

/* Target common streamlit wrapper patterns that render as white rounded pills */
.block-container > div[style*="border-radius"],
.block-container > section[style*="border-radius"],
.block-container > div[style*="background: rgb(255, 255, 255)"],
.block-container > div[style*="background: #fff"],
.block-container > div[style*="border-radius: 24px"],
.block-container > div[style*="border-radius: 25px"],
div[role="region"][style*="border-radius"],
div[role="group"][style*="border-radius"] {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    min-height: 0 !important;
    height: auto !important;
}

/* Hide tiny empty spacer elements that appear as pills */
.block-container > div:empty,
.block-container > section:empty,
.block-container > div:has(> .stButton):empty {
    display: none !important;
}

/* keep other card styles intact */
.graph-card { background: #0f1113; padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow:0 4px 12px rgba(0,0,0,0.04); }

/* enlarge title/caption and make button bigger */
.big-title { font-size:56px !important; font-weight:900 !important; color:#e63946 !important; margin:0; line-height:1; }
.big-caption { font-size:20px !important; color:#cfcfcf !important; margin:8px 0 18px 0; font-weight:500; }
.stButton>button, .stButton button { font-size:18px !important; padding:14px 28px !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("<div style='text-align:center;margin-bottom:6px'>", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🚴 AI RideWise</div>", unsafe_allow_html=True)
st.markdown("<div class='big-caption'>Predicting bike-sharing demand using weather & urban events</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Helper functions (preprocessing)
# -------------------------
def weekday_sunday0_from_date(d: datetime.date):
    # Python: Monday=0 .. Sunday=6. We map to Sunday=0..Saturday=6 used in training
    py = d.weekday()
    mapping = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:0}
    return mapping.get(py, py)

def season_code(name: str):
    # original dataset: 1=Spring,2=Summer,3=Fall,4=Winter
    return {"Spring":1, "Summer":2, "Fall":3, "Winter":4}[name]

def weathersit_code(name: str):
    # original dataset mapping (1..4)
    return {"Clear/Cloudy":1, "Mist":2, "Light Rain/Snow":3, "Heavy Rain/Snow":4}[name]

# exactly same cyclic transforms as notebook
def sin_cos_month(m): return math.sin(2*math.pi*m/12), math.cos(2*math.pi*m/12)
def sin_cos_weekday(w): return math.sin(2*math.pi*w/7), math.cos(2*math.pi*w/7)
def sin_cos_hour(h): return math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24)

# scaling to match dataset (UCI Bike Sharing)
def scale_temp_c(t_c): return float(t_c) / 41.0
def scale_atemp_c(a_c): return float(a_c) / 50.0
def scale_hum_pct(h): return float(h) / 100.0
def scale_wind_kmh(w): return float(w) / 67.0

# fe_hour reproduction (we won't call on entire df here, but follow same logic)
def build_features_from_inputs(selected_date, hr, season_name, weathersit_name,
                               holiday_flag, workingday_flag,
                               temp_c, atemp_c, hum_pct, wind_kmh,
                               df_for_trend=None):
    """
    Returns feature vector (1x25) in the same order the model expects:
    ['yr','mnth','holiday','weekday','workingday','temp','atemp','hum','windspeed',
     'year','month','hour','sin_month','cos_month','sin_weekday','cos_weekday',
     'sin_hour','cos_hour','trend','season_2','season_3','season_4',
     'weathersit_2','weathersit_3','weathersit_4']
    """
    # raw numeric pieces
    yr_flag = 0 if selected_date.year == 2011 else 1  # matches previous convention
    mnth = selected_date.month
    weekday_num = weekday_sunday0_from_date(selected_date)
    hour_val = hr

    # scaled sensors
    temp = scale_temp_c(temp_c)
    atemp = scale_atemp_c(atemp_c)
    hum = scale_hum_pct(hum_pct)
    windspeed = scale_wind_kmh(wind_kmh)

    # duplicate year/month fields as training did
    year_val = float(selected_date.year)
    month_val = float(selected_date.month)

    # cyclic features
    sin_month, cos_month = sin_cos_month(mnth)
    sin_weekday, cos_weekday = sin_cos_weekday(weekday_num)
    sin_hour, cos_hour = sin_cos_hour(hour_val)

    # trend: days since dataset min dteday (if available) else since 2011-01-01
    if df_for_trend is not None and "dteday" in df_for_trend.columns:
        try:
            start_date = pd.to_datetime(df_for_trend["dteday"]).min().date()
        except Exception:
            start_date = datetime.date(2011,1,1)
    else:
        start_date = datetime.date(2011,1,1)
    trend = (selected_date - start_date).days

    # one-hot dummies produced by pd.get_dummies(drop_first=True) -> season_2,3,4 and weathersit_2,3,4
    season_code_val = season_code(season_name)
    weathersit_code_val = weathersit_code(weathersit_name)

    season_2 = 1 if season_code_val == 2 else 0
    season_3 = 1 if season_code_val == 3 else 0
    season_4 = 1 if season_code_val == 4 else 0

    weathersit_2 = 1 if weathersit_code_val == 2 else 0
    weathersit_3 = 1 if weathersit_code_val == 3 else 0
    weathersit_4 = 1 if weathersit_code_val == 4 else 0

    features = [
        float(yr_flag),           # yr
        float(mnth),              # mnth
        float(holiday_flag),      # holiday
        float(weekday_num),       # weekday
        float(workingday_flag),   # workingday
        float(temp),              # temp (scaled)
        float(atemp),             # atemp (scaled)
        float(hum),               # hum (scaled)
        float(windspeed),         # windspeed (scaled)
        float(year_val),          # year (duplicate)
        float(month_val),         # month (duplicate)
        float(hour_val),          # hour
        float(sin_month),         # sin_month
        float(cos_month),         # cos_month
        float(sin_weekday),       # sin_weekday
        float(cos_weekday),       # cos_weekday
        float(sin_hour),          # sin_hour
        float(cos_hour),          # cos_hour
        float(trend),             # trend
        float(season_2),          # season_2
        float(season_3),          # season_3
        float(season_4),          # season_4
        float(weathersit_2),      # weathersit_2
        float(weathersit_3),      # weathersit_3
        float(weathersit_4)       # weathersit_4
    ]

    return np.array(features, dtype=float).reshape(1, -1)

# -------------------------
# UI — Inputs (3 columns) — with explicit labels and unique keys
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📅 Ride Details")

col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.date_input("Select Date", datetime.date(2012, 1, 1), key="date_input")
    hour = st.slider("Hour of Day (0–23)", 0, 23, 12, key="hour_input")
    season_choice = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], key="season_input")

with col2:
    holiday_choice = st.selectbox("Holiday", ["No", "Yes"], key="holiday_input")
    workingday_choice = st.selectbox("Working Day", ["No", "Yes"], key="working_input")
    weathersit_choice = st.selectbox("Weather Situation", ["Clear/Cloudy", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"], key="weathersit_input")

with col3:
    st.markdown("**Environmental Inputs**")
    temp_c = st.slider("Temperature (°C)", 0.0, 45.0, 22.0, 0.1, key="temp_input")
    atemp_c = st.slider("Feels-like Temp (°C)", 0.0, 50.0, 22.0, 0.1, key="atemp_input")
    hum_pct = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 1.0, key="hum_input")
    wind_kmh = st.slider("Windspeed (km/h)", 0.0, 67.0, 10.0, 0.1, key="wind_input")

st.markdown("</div>", unsafe_allow_html=True)

# Helper conversions from UI choices to numeric flags
holiday_flag = 1 if holiday_choice == "Yes" else 0
workingday_flag = 1 if workingday_choice == "Yes" else 0

# -------------------------
# Predict button (centered)
# -------------------------
col_l, col_c, col_r = st.columns([1,6,1])
with col_c:
    st.markdown("<div style='display:flex; justify-content:center; align-items:center; margin:8px 0 18px 0;'>", unsafe_allow_html=True)
    # center the button visually (applies to Streamlit buttons)
    st.markdown("""
    <style>
    /* make buttons block-level and centered inside their container */
    .stButton>button, .stButton button {
        display: block !important;
        margin: 0 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    predict_clicked = st.button("Predict Hourly Ride Count", key="predict_button")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Auto-scroll helper
# -------------------------
def scroll_to_result():
    st.markdown("""
    <script>
      const el = document.getElementById('result_anchor');
      if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    </script>
    """, unsafe_allow_html=True)

# -------------------------
# Run prediction
# -------------------------
if predict_clicked:
    # build features exactly as during training
    features = build_features_from_inputs(
        selected_date, hour, season_choice, weathersit_choice,
        holiday_flag, workingday_flag,
        temp_c, atemp_c, hum_pct, wind_kmh,
        df_for_trend=df
    )

    # sanity: validate model n_features_in_ if available
    model_n = getattr(model, "n_features_in_", None)
    if model_n is not None and features.shape[1] != model_n:
        st.error(f"Feature count mismatch: model expects {model_n} features but app prepared {features.shape[1]}.")
        st.write("Prepared feature names (expected 25 order):")
        st.write("['yr','mnth','holiday','weekday','workingday','temp','atemp','hum','windspeed','year','month','hour','sin_month','cos_month','sin_weekday','cos_weekday','sin_hour','cos_hour','trend','season_2','season_3','season_4','weathersit_2','weathersit_3','weathersit_4']")
        st.write("Prepared feature values:")
        st.write(features.tolist()[0])
    else:
        try:
            pred = model.predict(features)
            pred_val = int(float(pred[0]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred_val = None

        # scroll + result display
        st.markdown("<div id='result_anchor'></div>", unsafe_allow_html=True)
        scroll_to_result()

        # big prediction card
        st.markdown(f"""
        <div class='pred-card'>
            <div style='font-size:92px; font-weight:900; line-height:0.9'>{pred_val}</div>
            <div style='font-size:18px; opacity:0.95; margin-top:6px;'>Predicted bike rentals for selected hour</div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # Full-day predicted pattern (single wide graph)
        # Replaced the previous two-column layout: removed the single-hour bar & historical IQR graph.
        # -------------------------
        st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
        
        # -------------------------
        # Side-by-side: Full-day predicted pattern (left) + Feature importance (right)
        # -------------------------
        col_left, col_right = st.columns([3,1])

        with col_left:
            st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
            st.markdown("#### 🔄 Full-day Predicted Pattern (0–23 hours)")

            hourly_preds = []
            for h in range(24):
                fvec = build_features_from_inputs(
                    selected_date, h, season_choice, weathersit_choice,
                    holiday_flag, workingday_flag,
                    temp_c, atemp_c, hum_pct, wind_kmh,
                    df_for_trend=df
                )
                try:
                    hourly_preds.append(float(model.predict(fvec)[0]))
                except Exception:
                    hourly_preds.append(np.nan)

            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(range(24), hourly_preds, marker='o', linewidth=2, color='#e63946', label='Predicted (full day)')
            # highlight selected hour
            if pred_val is not None:
                ax.scatter([hour], [pred_val], color='red', s=100, zorder=10, label='Selected hour')
            ax.set_xticks(range(24))
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Rentals")
            ax.set_title("Predicted Hourly Demand")
            ax.grid(alpha=0.2)
            ax.legend()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Feature Importance")

            try:
                if hasattr(model, "feature_importances_"):
                    importances = np.array(model.feature_importances_, dtype=float)
                    feat_names = ["yr","mnth","holiday","weekday","workingday",
                                  "temp","atemp","hum","windspeed",
                                  "year","month","hour",
                                  "sin_month","cos_month","sin_weekday","cos_weekday","sin_hour","cos_hour",
                                  "trend",
                                  "season_2","season_3","season_4",
                                  "weathersit_2","weathersit_3","weathersit_4"]

                    # adapt names if lengths differ
                    if len(importances) != len(feat_names):
                        if features.shape[1] == len(importances):
                            feat_names = [f"f{i}" for i in range(len(importances))]
                        else:
                            # fallback: numeric labels
                            feat_names = [f"f{i}" for i in range(len(importances))]

                    sorted_idx = np.argsort(importances)[::-1]  # descending
                    top_idx = sorted_idx[:20]  # show top 20 if many
                    fig2, ax2 = plt.subplots(figsize=(4,6))
                    ax2.barh(np.array(feat_names)[top_idx][::-1], importances[top_idx][::-1], color="#0b6b3a")
                    ax2.set_xlabel("Importance")
                    ax2.set_title("Feature importances (top)")
                    plt.tight_layout()
                    st.pyplot(fig2)
                else:
                    st.info("Feature importance not available for this model.")
            except Exception as e:
                st.error(f"Failed to render feature importances: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

