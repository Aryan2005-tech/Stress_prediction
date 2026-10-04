"""
SAHAYAK — AI-Based Predictive Personnel Stress & Welfare Monitoring System
Streamlit Dashboard for SIH 2026 | Team Caraxes | PS-26186
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAHAYAK",
    page_icon="🛡️",
    layout="wide",
)

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = "model/xgboost_stress_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/feature_names.pkl"
LE_PATH = "model/label_encoders.pkl"
HISTORY_PATH = "data/prediction_history.csv"

HIGH_THRESH = 0.7
MOD_THRESH = 0.4

ZONE_RISK_MAP = {
    "J&K (High Risk)": 0.85, "Northeast (High Risk)": 0.80,
    "Naxal Belt (High Risk)": 0.82, "Border Area (Moderate)": 0.55,
    "Urban Posting (Low)": 0.25, "Training Center (Low)": 0.20,
    "HQ / Admin (Low)": 0.15,
}

# ── Load artefacts ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    return model, scaler, feature_names, label_encoders

def save_to_history(data_dict, risk_score, category):
    # Prepare history record
    record = data_dict.copy()
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["predicted_risk_score"] = round(risk_score, 4)
    record["risk_category"] = category
    
    df_new = pd.DataFrame([record])
    
    if os.path.exists(HISTORY_PATH):
        df_history = pd.read_csv(HISTORY_PATH)
        df_history = pd.concat([df_history, df_new], ignore_index=True)
    else:
        df_history = df_new
        
    df_history.to_csv(HISTORY_PATH, index=False)

def risk_color(score):
    if score >= HIGH_THRESH: return "#ef4444"
    elif score >= MOD_THRESH: return "#f59e0b"
    return "#22c55e"

def risk_category(score):
    if score >= HIGH_THRESH: return "High"
    elif score >= MOD_THRESH: return "Moderate"
    return "Low"

def main():
    model, scaler, feature_names, label_encoders = load_model()
    
    # Simple navigation
    st.sidebar.title("🛡️ SAHAYAK Menu")
    mode = st.sidebar.radio("Select View:", ["Personnel Assessment", "Admin Dashboard"])
    
    if mode == "Personnel Assessment":
        st.title("🎯 Personnel Wellness Assessment")
        st.write("Please fill in your details to check your burnout risk score.")
        
        with st.form("prediction_form"):
            st.subheader("1. Demographics & Service")
            c1, c2, c3 = st.columns(3)
            with c1:
                inp_name = st.text_input("Name / ID", "User-001")
                inp_age = st.number_input("Age", 21, 58, 35)
                inp_gender = st.selectbox("Gender", ["Male", "Female"])
                inp_rank = st.selectbox("Rank", ["Constable", "Head Constable", "ASI", "SI", "Inspector", "Dy. Commandant", "Commandant", "DIG", "IG"])
            with c2:
                inp_unit = st.selectbox("Unit", ["CRPF", "BSF", "CISF", "ITBP", "SSB", "NSG", "AR", "RPF"])
                inp_yos = st.number_input("Years of Service", 1, 38, 10)
                inp_marital = st.selectbox("Marital Status", ["Single", "Married", "Widowed", "Divorced"])
                inp_deps = st.number_input("Dependents", 0, 10, 2)
            with c3:
                inp_zone = st.selectbox("Deployment Zone", list(ZONE_RISK_MAP.keys()))
                inp_dep_dur = st.number_input("Deployment Duration (months)", 1, 36, 12)
                inp_total_dep = st.number_input("Total Deployments", 1, 20, 5)
                inp_transfer = st.number_input("Transfers (last 5yr)", 0, 10, 2)
                
            st.subheader("2. Workload & Leave")
            c4, c5, c6 = st.columns(3)
            with c4:
                inp_fam_sep = st.number_input("Family Separation (months)", 0, 48, 12)
                inp_dist = st.number_input("Distance from Home (km)", 50, 3500, 800)
                inp_duty_hrs = st.number_input("Avg Duty Hours/Week", 35.0, 90.0, 52.0, step=1.0)
                inp_night = st.number_input("Night Duties/Month", 0, 20, 6)
            with c5:
                inp_train = st.number_input("Training Hours/Month", 0.0, 60.0, 15.0, step=1.0)
                inp_workload = st.slider("Workload Score (1-10)", 1.0, 10.0, 5.5, step=0.5)
                inp_leave_ent = st.selectbox("Leaves Entitled", [30, 45, 60])
                inp_leave_tak = st.number_input("Leaves Taken", 0, 60, 15)
            with c6:
                inp_sick = st.number_input("Sick Leave Count", 0, 15, 3)
                inp_promo_gap = st.number_input("Years Since Promotion", 0, 12, 3)
                inp_comm = st.number_input("Commendations", 0, 10, 1)
                inp_disc = st.number_input("Disciplinary Actions", 0, 5, 0)
                
            st.subheader("3. Wellness (Self-Reported)")
            c7, c8, c9 = st.columns(3)
            with c7:
                inp_incident = st.number_input("Incident Exposure Count", 0, 25, 3)
                inp_wellness = st.slider("General Wellness Score (1-10)", 1.0, 10.0, 6.0, step=0.5)
            with c8:
                inp_sleep = st.slider("Sleep Quality (1-5)", 1.0, 5.0, 3.0, step=0.5)
                inp_fitness = st.slider("Physical Fitness (1-10)", 1.0, 10.0, 6.5, step=0.5)
            with c9:
                inp_mood = st.slider("Mood Score (1-10)", 1.0, 10.0, 5.5, step=0.5)
                inp_social = st.slider("Social Connectedness (1-10)", 1.0, 10.0, 5.0, step=0.5)
                
            submitted = st.form_submit_button("Submit Assessment", use_container_width=True)
            
        if submitted:
            # Prepare data
            inp_zone_risk_val = ZONE_RISK_MAP.get(inp_zone, 0.5)
            leave_ratio = inp_leave_tak / max(inp_leave_ent, 1)
            
            person_dict = {
                "user_id": inp_name,
                "age": inp_age, "years_of_service": inp_yos,
                "gender": inp_gender, "rank": inp_rank, "unit": inp_unit,
                "marital_status": inp_marital, "num_dependents": inp_deps,
                "deployment_zone": inp_zone, "deployment_zone_risk": inp_zone_risk_val,
                "deployment_duration_months": inp_dep_dur,
                "total_deployments": inp_total_dep,
                "transfer_frequency_5yr": inp_transfer,
                "family_separation_months": inp_fam_sep,
                "distance_from_home_km": inp_dist,
                "avg_duty_hours_per_week": inp_duty_hrs,
                "night_duty_freq_per_month": inp_night,
                "training_hours_per_month": inp_train,
                "workload_score": inp_workload,
                "leaves_entitled": inp_leave_ent,
                "leaves_taken": inp_leave_tak,
                "leave_utilization_ratio": leave_ratio,
                "sick_leave_count": inp_sick,
                "years_since_last_promotion": inp_promo_gap,
                "commendations": inp_comm,
                "disciplinary_actions": inp_disc,
                "incident_exposure_count": inp_incident,
                "wellness_checkin_score": inp_wellness,
                "sleep_quality_score": inp_sleep,
                "physical_fitness_score": inp_fitness,
                "mood_score": inp_mood,
                "social_connectedness_score": inp_social,
            }
            
            # Predict
            X_input = pd.DataFrame([person_dict])[feature_names].copy()
            for col, le in label_encoders.items():
                if col in X_input.columns:
                    X_input[col] = le.transform(X_input[col].astype(str))
            X_inp_scaled = scaler.transform(X_input.values.astype(np.float32))
            pred_risk = float(model.predict_proba(X_inp_scaled)[:, 1][0])
            
            cat = risk_category(pred_risk)
            col = risk_color(pred_risk)
            
            # Save history
            save_to_history(person_dict, pred_risk, cat)
            
            st.divider()
            st.subheader("Results")
            
            # Show output
            r1, r2 = st.columns([1, 2])
            with r1:
                st.markdown(f"### Overall Risk Level: <span style='color:{col}'>{cat}</span>", unsafe_allow_html=True)
                st.metric("Burnout Risk Score", f"{pred_risk:.1%}")
                if cat == "High":
                    st.error("Priority Wellness Review Recommended.")
                elif cat == "Moderate":
                    st.warning("Monitor closely and follow wellness protocols.")
                else:
                    st.success("Status Healthy. Keep up the good work.")
            
            with r2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_risk * 100,
                    number=dict(suffix="%"),
                    title=dict(text="Risk Score"),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color=col),
                        steps=[
                            dict(range=[0, MOD_THRESH * 100], color="lightgreen"),
                            dict(range=[MOD_THRESH * 100, HIGH_THRESH * 100], color="lightyellow"),
                            dict(range=[HIGH_THRESH * 100, 100], color="lightpink"),
                        ],
                    ),
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=30, b=20, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)


    elif mode == "Admin Dashboard":
        st.title("📊 Admin Dashboard")
        st.write("History of all personnel assessments.")
        
        if os.path.exists(HISTORY_PATH):
            df_history = pd.read_csv(HISTORY_PATH)
            
            # Sort by newest first
            df_history = df_history.sort_values(by="timestamp", ascending=False)
            
            # High-level metrics
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Total Assessments", len(df_history))
            h2.metric("High Risk Clients", len(df_history[df_history['risk_category'] == 'High']))
            h3.metric("Moderate Risk Clients", len(df_history[df_history['risk_category'] == 'Moderate']))
            h4.metric("Low Risk Clients", len(df_history[df_history['risk_category'] == 'Low']))
            
            st.divider()
            
            st.subheader("Client History")
            # Only show relevant columns for easy viewing
            display_cols = ["timestamp", "user_id", "rank", "unit", "deployment_zone", "predicted_risk_score", "risk_category"]
            st.dataframe(df_history[display_cols], use_container_width=True)
            
            # Simple chart
            st.subheader("Risk Category Distribution")
            dist = df_history['risk_category'].value_counts().reset_index()
            dist.columns = ['Category', 'Count']
            
            import plotly.express as px
            fig = px.pie(dist, names='Category', values='Count', color='Category',
                         color_discrete_map={'High':'red', 'Moderate':'orange', 'Low':'green'})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No assessments have been submitted yet. Go to the 'Personnel Assessment' page to submit one.")

if __name__ == "__main__":
    main()
