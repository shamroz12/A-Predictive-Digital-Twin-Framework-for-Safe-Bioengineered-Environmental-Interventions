
"""
Digital Twin for Biological Environmental Forecasting Web App
-------------------------------------------------------------
This Streamlit app simulates environmental outcomes from synthetic biology interventions.
Modules: PM2.5 reduction, plastic degradation, malaria suppression.
Adjust inputs and visualize predictive outcomes interactively.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digital Twin for Bioenvironment", layout="wide")

st.title("Digital Twin for Biological Environmental Forecasting")
st.markdown("""
This software simulates how biological interventions impact the environment. Use it to test and visualize predictions for:
- **PM2.5 Air Pollution** reduction using biofilters
- **Microplastic Degradation** using PETase enzymes
- **Malaria Incidence** drop via gene drives

 Suitable for researchers in environmental science, synthetic biology, and public health.
""")

col1, col2 = st.columns(2)
with col1:
    pm_start = st.slider("Initial PM2.5 (µg/m³)", 40, 150, 90)
    pm_eff = st.slider("Filter Efficiency", 0.1, 0.9, 0.5)
    seasonality = st.slider("Seasonal Variation", 0.0, 0.5, 0.1)
with col2:
    plastic_load = st.slider("Initial Plastic Load (%)", 50, 150, 100)
    decay_rate = st.slider("PETase Decay Rate", 0.1, 0.6, 0.3)
    temp_factor = st.slider("Ocean Temperature Factor", 0.8, 1.5, 1.0)
    malaria_cases = st.slider("Baseline Malaria Cases", 50, 200, 100)
    gene_drive = st.slider("Gene Drive Efficacy", 0.1, 0.5, 0.2)
    resistance = st.slider("Mosquito Resistance", 0.0, 0.4, 0.1)

# Modular forecast functions
def forecast_pm25(t, start, eff, seasonality):
    decay = -np.log(1 - eff) / 12
    return start * np.exp(-decay * t) * (1 + seasonality * np.sin(2 * np.pi * t / 12))

def forecast_plastics(t, load, rate, temp):
    return load * np.exp(-rate * temp * t)

def forecast_malaria(t, base, drive, resist):
    return base * np.exp(-drive * (1 - resist) * t)

# Run simulations
def run_simulations():
    t1 = np.arange(13)
    t2 = np.arange(11)
    t3 = np.arange(16)
    pm_vals = forecast_pm25(t1, pm_start, pm_eff, seasonality)
    plast_decay = forecast_plastics(t2, plastic_load, decay_rate, temp_factor)
    malaria_effect = forecast_malaria(t3, malaria_cases, gene_drive, resistance)
    return t1, pm_vals, t2, plast_decay, t3, malaria_effect

t1, pm_vals, t2, plast, t3, malaria = run_simulations()

# Display outputs
st.subheader("Forecast Outputs")
chart1, chart2, chart3 = st.columns(3)

with chart1:
    fig, ax = plt.subplots()
    ax.plot(t1, pm_vals, color='green')
    ax.axhline(20, color='gray', linestyle='--')
    ax.set_title("PM2.5 Forecast")
    ax.set_xlabel("Months")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.text(0, pm_vals.max()*0.9, "↓ Biofilter impact", fontsize=9)
    st.pyplot(fig)

with chart2:
    fig, ax = plt.subplots()
    ax.plot(t2, plast, 'b-o')
    ax.set_title("Microplastic Degradation")
    ax.set_xlabel("Years")
    ax.set_ylabel("Remaining (%)")
    ax.text(0, plast[0]*0.85, "↓ PETase enzyme decay", fontsize=9)
    st.pyplot(fig)

with chart3:
    fig, ax = plt.subplots()
    ax.plot(t3, malaria, color='purple')
    ax.set_title("Malaria Forecast")
    ax.set_xlabel("Years")
    ax.set_ylabel("Relative Incidence")
    ax.text(0, malaria[0]*0.85, "↓ Gene Drive", fontsize=9)
    st.pyplot(fig)

st.markdown("""
### Insights:
- **PM2.5** levels decline rapidly with higher efficiency filters
- **Plastic decay** is faster with higher PETase rate and warmer waters
- **Malaria cases** drop under gene drive, but resistance slows effect

Researchers can apply this to test environmental interventions **before real-world trials**, reducing risk and optimizing deployment.
""")
