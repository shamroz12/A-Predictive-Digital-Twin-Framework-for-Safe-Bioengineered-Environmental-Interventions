
# Reconstructed advanced app with model switcher, CSV upload, and 3 forecast domains
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🌍 Advanced Digital Twin Predictor", layout="wide")
st.title("🌱 AI-Powered Environmental Digital Twin Dashboard")

st.markdown("""
This advanced app forecasts environmental outcomes from biological interventions:

- PM2.5 Air Pollution ↘️ (Biofilters)
- Ocean Plastics ↘️ (PETase)
- Malaria ↘️ (Gene Drives)

🔬 Select from multiple AI models and test interventions interactively.
""")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("🧪 Environmental Inputs")
with st.sidebar.form("inputs"):
    model_choice = st.selectbox("Prediction Model", ["Polynomial Regression", "Random Forest", "Support Vector Regressor"])
    years = st.slider("Forecast Duration (Years)", 5, 30, 15)
    intervention_strength = st.slider("Intervention Strength (%)", 10, 100, 60)
    impact_scale = st.slider("Intervention Impact Multiplier", 0.1, 2.0, 1.0, step=0.1)
    noise_level = st.slider("Environmental Variability", 0.0, 0.3, 0.1)
    disease_resistance = st.slider("Mosquito Resistance", 0.0, 0.5, 0.1)
    uploaded_file = st.file_uploader("Upload CSV (Years, Values) for Custom Forecast", type=["csv"])
    submit = st.form_submit_button("Run Simulation")

# ------------------- Forecasting Function -------------------
def predict_with_model(x, y_obs, model_type):
    if model_type == "Polynomial Regression":
        poly = PolynomialFeatures(3)
        x_poly = poly.fit_transform(x)
        model = LinearRegression().fit(x_poly, y_obs)
        y_pred = model.predict(x_poly)
    elif model_type == "Random Forest":
        model = RandomForestRegressor().fit(x, y_obs)
        y_pred = model.predict(x.reshape(-1, 1))
    elif model_type == "Support Vector Regressor":
        model = SVR(kernel='rbf').fit(x, y_obs)
        y_pred = model.predict(x)
    else:
        y_pred = y_obs
    return y_pred

# ------------------- Main Model Execution -------------------
if submit:
    base = 100
    decay_rate = 0.2 * (intervention_strength / 100) * impact_scale
    x = np.arange(years).reshape(-1, 1)
    x_flat = x.flatten()

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        y_pm25 = data.iloc[:, 1].values[:years] if len(data) >= years else np.interp(x_flat, np.arange(len(data)), data.iloc[:, 1])
        y_pm25 += np.random.normal(0, noise_level * base, size=y_pm25.shape)
    else:
        y_pm25 = base * np.exp(-decay_rate * x_flat) + np.random.normal(0, noise_level * base, size=years)

    # Plastics and malaria
    y_plast = base * np.exp(-decay_rate * 0.8 * x_flat) + np.random.normal(0, noise_level * base * 0.8, size=years)
    y_malaria = base * np.exp(-decay_rate * (1 - disease_resistance) * x_flat) + np.random.normal(0, noise_level * base * 0.5, size=years)

    pred_pm25 = predict_with_model(x, y_pm25, model_choice)
    pred_plast = predict_with_model(x, y_plast, model_choice)
    pred_malaria = predict_with_model(x, y_malaria, model_choice)

    st.subheader("📈 Forecast Visualizations")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(x_flat, y_pm25, 'o-', label='Observed PM2.5')
        ax.plot(x_flat, pred_pm25, 'r--', label='Predicted')
        ax.set_title("Air Pollution")
        ax.set_xlabel("Years"); ax.set_ylabel("PM2.5 µg/m³")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(x_flat, y_plast, 'b-o', label='Observed Plastics')
        ax.plot(x_flat, pred_plast, 'orange', label='Predicted')
        ax.set_title("Plastic Degradation")
        ax.set_xlabel("Years"); ax.set_ylabel("Residual (%)")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        ax.plot(x_flat, y_malaria, 'purple', label='Observed Malaria')
        ax.plot(x_flat, pred_malaria, 'lime', label='Predicted')
        ax.set_title("Malaria Incidence")
        ax.set_xlabel("Years"); ax.set_ylabel("Relative Load")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretation
    - Model: `{}`  
    - PM2.5, plastics, and malaria curves simulate real-world interventions.
    - Customize with your own CSV to personalize outcomes.

    ✅ Ideal for biology + public health researchers  
    ✅ Visualizes cross-domain impact from synthetic solutions
    """.format(model_choice))
