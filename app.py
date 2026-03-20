import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AQI Intelligence System",
    page_icon="🌿",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background (Minimal Mint - Microsoft Style) */
.stApp {
    background: linear-gradient(120deg, #f8fffc, #ecfdf5);
    color: #1b4332;
}

/* Header */
.title {
    font-size: 42px;
    font-weight: 600;
    color: #2d6a4f;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #52796f;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.85);
    padding: 18px;
    border-radius: 12px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Button */
.stButton>button {
    background-color: #10b981;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 18px;
    width: 100%;
    border: none;
}
.stButton>button:hover {
    background-color: #059669;
}

/* Section spacing */
.section {
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("IndianAirQuality.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌿 AQI Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Air Quality Prediction & Insights</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown("### 🧾 Enter Environmental Parameters")

cols = st.columns(3)

labels = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
values = []

for i, label in enumerate(labels):
    col = cols[i % 3]
    val = col.number_input(label, value=0.0)
    values.append(val)

features = np.array([values])

# ---------------- AQI CATEGORY ----------------
def get_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

# ---------------- PREDICTION ----------------
if st.button("🚀 Run AI Analysis"):

    prediction = model.predict(features)[0]
    category = get_category(prediction)

    st.markdown("---")

    # ---------------- KPI CARDS ----------------
    c1, c2, c3 = st.columns(3)

    c1.metric("📊 AQI", f"{prediction:.2f}")
    c2.metric("🌡️ Category", category)
    c3.metric("🧠 Model", "ML Prediction")

    # ---------------- PROGRESS ----------------
    st.progress(min(prediction / 500, 1.0))

    # ---------------- POLLUTANT CHART ----------------
    st.markdown("### 📊 Pollutant Distribution")

    df = pd.DataFrame({
        "Pollutant": labels,
        "Value": values
    })

    st.bar_chart(df.set_index("Pollutant"))

    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown("### 🧠 Key Pollution Drivers")

    importance = np.array(values) / (np.sum(values) + 1e-6)

    imp_df = pd.DataFrame({
        "Pollutant": labels,
        "Impact": importance
    }).sort_values(by="Impact", ascending=False)

    st.dataframe(imp_df.head(5), use_container_width=True)

    # ---------------- AI INSIGHTS ----------------
    st.markdown("### 🤖 AI Insights")

    if prediction > 250:
        st.error("🚨 Critical pollution detected. Immediate precautions required.")
    elif prediction > 150:
        st.warning("⚠️ Moderate pollution. Limit outdoor exposure.")
    else:
        st.success("✅ Air quality is safe.")

    # ---------------- SMART RECOMMENDATIONS ----------------
    st.markdown("### 💡 Smart Recommendations")

    tips = []
    if values[0] > 100:
        tips.append("High PM2.5 → Wear mask 😷")
    if values[1] > 150:
        tips.append("High PM10 → Avoid outdoor activity")
    if values[6] > 1:
        tips.append("High CO → Ensure ventilation")

    if tips:
        for tip in tips:
            st.info(tip)
    else:
        st.info("Environment looks healthy 🌿")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🌿 AI-based Environmental Intelligence • Built with Streamlit")