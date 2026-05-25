import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("RidgeModel.pkl")

# -----------------------------
# Load and clean locations
# -----------------------------
@st.cache_data
def load_locations():
    data = pd.read_csv("Bengaluru_House_Data.csv")

    data["location"] = data["location"].fillna("Sarjapur Road")
    data["location"] = data["location"].apply(lambda x: x.strip())

    location_count = data["location"].value_counts()
    location_cnt_less_10 = location_count[location_count <= 10]
    data["location"] = data["location"].apply(
        lambda x: "others" if x in location_cnt_less_10.index else x
    )

    locations = sorted(data["location"].unique().tolist())
    return locations

model = load_model()
locations = load_locations()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="padding:18px;border-radius:16px;background:linear-gradient(90deg,#111827,#374151);color:white;">
        <h1 style="margin:0;">🏠 Bangalore House Price Predictor</h1>
        <p style="margin:6px 0 0 0;">Select property details and get an estimated price instantly.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("🏡 Property Inputs")

location = st.sidebar.selectbox(
    "Location",
    locations,
    index=0
)

total_sqft = st.sidebar.slider(
    "Total Sqft",
    min_value=300,
    max_value=10000,
    value=1000,
    step=50
)

bath = st.sidebar.slider(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

bhk = st.sidebar.slider(
    "BHK",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📋 Selected Details")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Location", location)
    c2.metric("Sqft", f"{total_sqft}")
    c3.metric("Baths", f"{bath}")
    c4.metric("BHK", f"{bhk}")

    input_df = pd.DataFrame(
        [[location, total_sqft, bath, bhk]],
        columns=["location", "total_sqft", "bath", "bhk"]
    )

    st.dataframe(input_df, use_container_width=True, hide_index=True)

    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

with right:
    st.subheader("📈 Prediction Dashboard")
    st.info("The gauge shows the final predicted price. The bar chart shows a small range around it.")

# -----------------------------
# Prediction
# -----------------------------
if predict_btn:
    if location not in locations:
        location = "others"

    prediction = model.predict(input_df)[0]

    low = prediction * 0.90
    high = prediction * 1.10

    st.success(f"🏷️ Estimated Price: ₹ {prediction:.2f} Lakhs")

    # -------------------------
    # Gauge Meter
    # -------------------------
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Final Predicted Price (Lakhs)"},
            gauge={
                "axis": {"range": [0, max(high * 1.2, 1)]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, low], "color": "#dbeafe"},
                    {"range": [low, high], "color": "#93c5fd"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": prediction
                }
            }
        )
    )

    st.plotly_chart(gauge_fig, use_container_width=True)

    # -------------------------
    # Bar Chart
    # -------------------------
    # fig, ax = plt.subplots(figsize=(7, 4))
    # labels = ["Low", "Predicted", "High"]
    # values = [low, prediction, high]

    # ax.bar(labels, values)
    # ax.set_title("Prediction Range")
    # ax.set_ylabel("Price in Lakhs")
    # ax.grid(axis="y", linestyle="--", alpha=0.3)

    # st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = ["Low", "Predicted", "High"]
    values = [low, prediction, high]

    ax.plot(labels, values, marker="o", linewidth=3)
    ax.fill_between(labels, values, alpha=0.15)

    ax.set_title("Prediction Range")
    ax.set_ylabel("Price in Lakhs")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    st.pyplot(fig)

    st.caption("Low/High values are a simple ±10% estimate around the model output.")
else:
    st.warning("Choose the inputs and click **Predict Price** to see the result.")
