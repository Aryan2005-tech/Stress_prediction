# SAHAYAK — AI-Based Predictive Personnel Stress & Welfare Monitoring System

> **SIH 2026 | PS-26186 | Team Caraxes**  
> Theme: MedTech / BioTech / HealthTech

## 🚀 Quick Start

### Option 1: Automated Setup (Windows)
```batch
setup.bat
```

### Option 2: Manual Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data/generate_synthetic_data.py

# Train XGBoost model
python model/train_model.py

# Run Streamlit app
streamlit run app.py
```

## 📁 Project Structure
```
Stress_prediction/
├── .env                          # Environment variables
├── .gitignore
├── .streamlit/config.toml        # Streamlit dark theme
├── requirements.txt              # Python dependencies
├── setup.bat                     # One-click setup (Windows)
├── app.py                        # Main Streamlit dashboard
├── data/
│   ├── generate_synthetic_data.py  # Synthetic CRPF HRMS generator
│   └── synthetic_crpf_data.csv     # Generated dataset (2000 records)
└── model/
    ├── train_model.py              # XGBoost training pipeline
    ├── xgboost_stress_model.pkl    # Trained model weights
    ├── scaler.pkl                  # Fitted StandardScaler
    ├── feature_names.pkl           # Feature name list
    └── label_encoders.pkl          # Categorical encoders
```

## 🏗️ Architecture
- **HR/Context Branch**: XGBoost classifier on HRMS features
- **Explainability**: Feature importance (SHAP-style)
- **Dashboard**: Streamlit with Plotly visualizations
- **Privacy**: RBAC, pseudonymous analytics, .env configuration

## ⚠️ Disclaimer
This prototype uses **entirely synthetic data**. No real CRPF personnel data is used.
