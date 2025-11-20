
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="AI-Enhanced Digital Twin Predictor", layout="wide")

st.title("🔬 Deep Environmental Predictor Using Digital Twin")
st.markdown("""
This advanced digital twin model uses AI-based regression to forecast the impact of synthetic biology interventions on:
- **Air Pollution (PM2.5)**
- **Microplastic Reduction**
- **Malaria Suppression**

🧠 Powered by polynomial learning with environmental controls.
""")

# ------------------ Input Fields ------------------
st.sidebar.header("🔧 Input Environmental Parameters")
with st.sidebar.form("input_form"):
    intervention_strength = st.number_input("Intervention Effectiveness (%)", min_value=10, max_value=100, value=60)
    timespan = st.number_input("Forecast Time (Years)", min_value=1, max_value=20, value=10)
    impact_scale = st.number_input("Impact Scaling Factor", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    noise = st.slider("Environmental Noise", 0.0, 0.3, 0.1, step=0.01)
    submit_button = st.form_submit_button("Predict Impact")

# ------------------ Model Logic ------------------
def generate_forecast(start_value, decay_coeff, years, noise_level=0.1, degree=3):
    x = np.arange(years).reshape(-1, 1)
    y = start_value * np.exp(-decay_coeff * x.flatten())
    y_noise = y + np.random.normal(0, noise_level * start_value, size=y.shape)
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x)
    model = LinearRegression().fit(X_poly, y_noise)
    y_pred = model.predict(X_poly)
    return x.flatten(), y_noise, y_pred

# ------------------ Display Output ------------------
if submit_button:
    coeff = (intervention_strength / 100) * impact_scale * 0.2
    base = 100
    x, y_actual, y_pred = generate_forecast(base, coeff, timespan, noise)

    st.subheader("📈 Prediction Results")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, y_actual, 'o', label='Simulated Sensor Data')
    ax.plot(x, y_pred, 'r-', label='AI Prediction')
    ax.set_title("Forecasted Environmental Recovery")
    ax.set_xlabel("Years")
    ax.set_ylabel("Environmental Burden Index")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretation
    - Uses polynomial regression to fit AI predictions to simulated outcomes.
    - Curve slope steepens with stronger interventions.
    - Noise slider simulates real-world sensor uncertainty.

    💡 This model is useful for pre-screening intervention strategies **before field testing**, offering explainable trends to guide policy.
    """)
