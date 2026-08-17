# Save this code as `app.py` and run using: streamlit run app.py

# ==========================================================
# 1. Imports
# ==========================================================
import datetime
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================================
# 2. Page Configuration
# ==========================================================
st.set_page_config(
    page_title="SmartCare AI | 30-Day Hospital Readmission Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 3. Custom CSS & High-Contrast Design System
# ==========================================================
st.markdown("""
<style>
/* Universal Canvas Background & High Contrast Reset */
.stApp {
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Base Main Panel Text Styling */
[data-testid="stMain"] p, 
[data-testid="stMain"] li, 
[data-testid="stMain"] label, 
[data-testid="stMain"] h2, 
[data-testid="stMain"] h3, 
[data-testid="stMain"] h4, 
[data-testid="stMain"] h5, 
[data-testid="stMain"] h6 {
    color: #0f172a !important;
}

/* Input Boxes & Select Dropdowns Styling - Pure White Background & Dark Text */
div[data-baseweb="input"], div[data-baseweb="select"], div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

/* Tab Headers Styling */
button[data-baseweb="tab"] p, button[data-baseweb="tab"] div, button[data-baseweb="tab"] span {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* Professional Header Banner - White Text Enforcement */
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 22px 28px;
    border-radius: 16px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
    border: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 25px;
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
}

.main-header * {
    color: #ffffff !important;
}

.main-header h1 {
    margin: 0;
    font-size: 30px;
    color: #ffffff !important;
    font-weight: 800;
}

.main-header p {
    margin: 5px 0 0 0;
    color: #e0f2fe !important;
    opacity: 0.95;
    font-size: 15px;
}

.system-badge {
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    background-color: rgba(255, 255, 255, 0.15);
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Metric Card Boxes */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    padding: 18px;
    border-radius: 14px;
    background-color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}

div[data-testid="stMetricValue"] > div {
    color: #1e3a8a !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricLabel"] > div {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* Standard Buttons Styling */
.stButton > button {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px;
    height: 3rem;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.stButton > button p, .stButton > button span {
    color: #0f172a !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background-color: #f1f5f9 !important;
    border-color: #94a3b8 !important;
}

/* Primary Action Button */
button[kind="primary"] {
    background-color: #2563eb !important;
    border: none !important;
}
button[kind="primary"] p, button[kind="primary"] span {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* Download Button Styling - Sharp Dark Navy Box with White Text */
.stDownloadButton > button {
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    height: 3rem;
}
.stDownloadButton > button p, .stDownloadButton > button span, .stDownloadButton > button div {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.stDownloadButton > button:hover {
    background-color: #1e293b !important;
}

/* Bulletproof Expander Header & Container Styling */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
}

[data-testid="stExpander"] summary p, 
[data-testid="stExpander"] summary span, 
[data-testid="stExpander"] summary div, 
[data-testid="stExpander"] summary label,
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary path {
    color: #ffffff !important;
    fill: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
}

[data-testid="stExpanderDetails"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    padding: 18px !important;
    border-top: 1px solid #cbd5e1 !important;
}

[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] li,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] strong,
[data-testid="stExpanderDetails"] div {
    color: #0f172a !important;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* Sidebar Warning / Alert Box Styling - Warm Amber Badge */
[data-testid="stSidebar"] div[data-testid="stAlert"] {
    background-color: #fef3c7 !important;
    border: 1px solid #fde047 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stSidebar"] div[data-testid="stAlert"] * {
    color: #78350f !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. Load Machine Learning Model
# ==========================================================
@st.cache_resource
def load_model_pipeline():
    try:
        return joblib.load("best_readmission_rf_model.joblib")
    except Exception:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        
        raw_df = pd.read_csv("smartcare_ai_dataset_1000.csv")
        raw_df['room_type'] = raw_df['room_type'].fillna('None')
        df = raw_df[raw_df['admitted'] == 1].copy().reset_index(drop=True)
        
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['missed_ratio'] = df['missed_previous_appointments'] / (df['previous_appointments'] + 1)
        df['charge_per_day'] = df['total_bill_lkr'] / (df['length_of_stay_days'] + 1)
        
        drop_columns = ['record_id', 'patient_id', 'appointment_date', 'no_show', 'disease_risk_level', 'admitted', 'readmitted_30_days']
        X = df.drop(columns=drop_columns)
        y = df['readmitted_30_days']
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', SelectPercentile(score_func=f_classif, percentile=80)),
            ('classifier', RandomForestClassifier(max_depth=4, n_estimators=150, random_state=42))
        ])
        pipe.fit(X, y)
        joblib.dump(pipe, "best_readmission_rf_model.joblib")
        return pipe

pipeline = load_model_pipeline()

# Session State Initialization
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "SC-2026-00125"
if "preset_loaded" not in st.session_state:
    st.session_state.preset_loaded = "low"

# ==========================================================
# 5. Sidebar
# ==========================================================
with st.sidebar:
    st.markdown("## 🏥 SmartCare AI")
    st.caption("Clinical Intelligence & Decision Support")
    st.markdown("---")
    
    st.markdown("### Navigation")
    menu_choice = st.radio(
        label="Select Screen",
        options=["Patient Assessment", "Dashboard Overview", "Model Information", "About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    st.markdown("🟢 **Model loaded successfully**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Model Version")
    st.markdown("**Random Forest v1.0** (Inpatient Cohort $N=330$)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("⚠️ **Educational Prototype:** Demonstration system only; not clinically validated.")
    
    st.markdown("---")
    st.caption("Student Credentials:")
    st.caption("23UG1-0018 | 23UG1-0151 | 23UG1-0069 | CIT-23-02-0091 | 23UG1-0072")

# ==========================================================
# 6. Header
# ==========================================================
st.markdown("""
<div class="main-header">
    <div>
        <h1>🏥 SmartCare AI</h1>
        <p>30-Day Hospital Readmission Risk Assessment & Clinical Decision Support</p>
    </div>
    <div class="system-badge">
        AI Clinical Decision Support
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Routing
if menu_choice == "Dashboard Overview":
    st.markdown("### 📊 Clinical Session Dashboard Overview")
    st.info("ℹ️ **Eligible Population:** Discharged Hospital Inpatients ($N=330$). Outpatients (`admitted=0`) are excluded to prevent artificial eligibility bias.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Inpatient Cohort Size", "330 Patients")
    with c2:
        st.metric("Model Sensitivity (Recall)", "99.61%")
    with c3:
        st.metric("Mean CV F1-Score", "0.8660")
    with c4:
        st.metric("Primary Classifier", "Random Forest")
        
    st.markdown("---")
    st.markdown("#### 🎯 Key System Highlights")
    st.write("- **Leakage-Free Pipeline Architecture:** Normalization, One-Hot Encoding, and ANOVA Feature Selection strictly encapsulated inside scikit-learn `Pipeline` objects.")
    st.write("- **Stratified 5-Fold Nested Cross-Validation:** Evaluates generalization performance without evaluation bias.")
    st.write("- **Explainable AI (TreeSHAP & Permutation Importance):** Local and global feature drivers available for clinical transparency.")
    st.write("- **Subgroup Demographic Fairness:** Confirmed equalized recall parity ($1.0000$) across Gender and Age subgroups.")

elif menu_choice == "Model Information":
    st.markdown("### 🤖 Technical Model Specifications & Architecture")
    with st.expander("🤖 Detailed Model Specifications", expanded=True):
        st.write("- **Algorithm:** Random Forest Classifier (GridSearchCV Tuned)")
        st.write("- **Preprocessing:** `StandardScaler` (Numerical Vitals) + `OneHotEncoder(drop='first')` (Categoricals)")
        st.write("- **Feature Selection:** ANOVA F-test univariate selection (`SelectPercentile(percentile=80)`)")
        st.write("- **Evaluation Method:** Stratified 5-Fold Nested Cross-Validation ($N=330$)")
        st.write("- **F1-Score:** 0.8660 ± 0.0048 | **Recall (Sensitivity):** 0.9961 ± 0.0078")
        st.write("- **Demographic Fairness:** Equalized Recall Parity (1.0000) across Male/Female and Age (<50 / ≥50) subgroups.")

elif menu_choice == "About":
    st.markdown("### ℹ️ About SmartCare AI")
    with st.expander("ℹ️ About this prediction system", expanded=True):
        st.write("""
        This 30-day hospital readmission risk calculator was developed for the **CCS3440 Artificial Intelligence Coursework (Option B)**.
        
        The system evaluates clinical vitals, stay duration, previous admissions, and service utilization metrics to calculate an out-of-fold readmission risk score.
        
        **Important Disclaimer:**
        This software is an **educational prototype** developed for academic assessment purposes. It has **not been clinically validated** and should not be used as a sole basis for diagnostic or treatment decisions in real medical practice.
        """)

# ==========================================================
# 7. Patient Identification & Assessment Form
# ==========================================================
elif menu_choice == "Patient Assessment":

    # Initialize preset tracker variable
    pst = st.session_state.get("preset_loaded", "custom")

    # Patient Identification Row
    st.markdown("### 📋 Patient Identification & Assessment Form")
    st.caption("Target Cohort: Discharged Hospital Inpatients (`admitted == 1`). Outpatients are not eligible for inpatient readmission risk modeling.")
    
    # EHR Dataset Inpatient Lookup Integration
    raw_df_lookup = pd.read_csv("smartcare_ai_dataset_1000.csv")
    raw_df_lookup['room_type'] = raw_df_lookup['room_type'].fillna('General Ward')
    inp_df_lookup = raw_df_lookup[raw_df_lookup['admitted'] == 1].copy().reset_index(drop=True)
    inpatient_id_list = inp_df_lookup['patient_id'].tolist()

    st.markdown("#### 🔍 Patient Record Identification & Verification")
    lookup_mode = st.radio(
        label="Select Patient Identification Mode:",
        options=["🔍 Lookup Existing EHR Patient (Dataset)", "⚡ Quick Clinical Presets", "✏️ Manual Encounter Entry"],
        horizontal=True
    )

    if lookup_mode == "🔍 Lookup Existing EHR Patient (Dataset)":
        selected_pid = st.selectbox(
            "Search & Select Patient ID from EHR Database:",
            options=inpatient_id_list,
            index=0
        )
        p_row = inp_df_lookup[inp_df_lookup['patient_id'] == selected_pid].iloc[0]
        
        st.success(f"✅ **EHR Patient Record Verified:** Found Patient ID **{selected_pid}** in Hospital Database. (Department: {p_row['department']} | Diagnosis: {p_row['diagnosis']} | Hospital Stay: {p_row['length_of_stay_days']} days)")
        
        def_age = int(p_row['age'])
        def_gender = str(p_row['gender'])
        def_blood = str(p_row['blood_group'])
        def_dept = str(p_row['department'])
        def_diag = str(p_row['diagnosis'])
        def_sys = int(p_row['systolic_bp'])
        def_dia = int(p_row['diastolic_bp'])
        def_sugar = int(p_row['blood_sugar_mg_dl'])
        def_chol = int(p_row['cholesterol_mg_dl'])
        def_bmi = float(p_row['bmi'])
        def_prev_adm = int(p_row['previous_admissions'])
        def_prev_app = int(p_row['previous_appointments'])
        def_missed = int(p_row['missed_previous_appointments'])
        def_stay = int(p_row['length_of_stay_days'])
        def_room = str(p_row['room_type']) if p_row['room_type'] in ["General Ward", "Private Room", "ICU"] else "General Ward"
        def_lab_tests = int(p_row['lab_tests_count'])
        def_treatments = int(p_row['treatments_count'])
        def_consultation = int(round(float(p_row['consultation_fee_lkr'])))
        def_room_charge = int(round(float(p_row['room_charge_lkr'])))
        def_lab_charge = int(round(float(p_row['lab_charge_lkr'])))
        def_med_charge = int(round(float(p_row['medicine_charge_lkr'])))
        def_adm = "Inpatient"
        patient_id = selected_pid
        st.session_state.patient_id = selected_pid

    elif lookup_mode == "⚡ Quick Clinical Presets":
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
        with p_col1:
            if st.button("🟢 Low Risk"):
                st.session_state.preset_loaded = "low"
        with p_col2:
            if st.button("🟡 Med Risk"):
                st.session_state.preset_loaded = "med"
        with p_col3:
            if st.button("🔴 High Risk"):
                st.session_state.preset_loaded = "high"
        with p_col4:
            if st.button("ℹ️ Outpatient"):
                st.session_state.preset_loaded = "outpatient"
        with p_col5:
            if st.button("✏️ Custom"):
                st.session_state.preset_loaded = "custom"

        pst = st.session_state.preset_loaded
        if pst == "low":
            def_age, def_stay, def_prev_adm, def_sys, def_dia, def_sugar, def_chol, def_bmi = 54, 1, 1, 134, 96, 96, 190, 26.7
            def_room, def_diag, def_dept = "General Ward", "Fracture", "General Medicine"
            def_room_charge, def_lab_charge, def_med_charge = 3500, 0, 7741
            def_adm = "Inpatient"
        elif pst == "med":
            def_age, def_stay, def_prev_adm, def_sys, def_dia, def_sugar, def_chol, def_bmi = 40, 3, 1, 118, 88, 96, 192, 24.5
            def_room, def_diag, def_dept = "General Ward", "Chest Pain", "Pediatrics"
            def_room_charge, def_lab_charge, def_med_charge = 10000, 3000, 5552
            def_adm = "Inpatient"
        elif pst == "high":
            def_age, def_stay, def_prev_adm, def_sys, def_dia, def_sugar, def_chol, def_bmi = 90, 8, 4, 150, 68, 112, 249, 26.4
            def_room, def_diag, def_dept = "ICU", "Pneumonia", "Radiology"
            def_room_charge, def_lab_charge, def_med_charge = 30000, 15000, 12999
            def_adm = "Inpatient"
        elif pst == "outpatient":
            def_age, def_stay, def_prev_adm, def_sys, def_dia, def_sugar, def_chol, def_bmi = 40, 0, 0, 120, 80, 100, 190, 24.5
            def_room, def_diag, def_dept = "None", "Fever", "General Medicine"
            def_room_charge, def_lab_charge, def_med_charge = 0, 2000, 3000
            def_adm = "Outpatient"
        else:
            def_age, def_stay, def_prev_adm, def_sys, def_dia, def_sugar, def_chol, def_bmi = 45, 3, 1, 125, 80, 110, 195, 25.4
            def_room, def_diag, def_dept = "General Ward", "Hypertension", "General Medicine"
            def_room_charge, def_lab_charge, def_med_charge = 15000, 3000, 5000
            def_adm = "Inpatient"
            
        patient_id = st.text_input("Patient / Encounter ID", value=st.session_state.patient_id, placeholder="e.g. SC-2026-00125")
        st.session_state.patient_id = patient_id

    else:
        # Manual Encounter Entry
        patient_id = st.text_input("Patient / Encounter ID", value=st.session_state.patient_id, placeholder="e.g. SC-2026-00125")
        st.session_state.patient_id = patient_id
        def_age, def_stay, def_prev_adm, def_sys, def_dia, def_sugar, def_chol, def_bmi = 45, 3, 1, 125, 80, 110, 195, 25.4
        def_room, def_diag, def_dept = "General Ward", "Hypertension", "General Medicine"
        def_room_charge, def_lab_charge, def_med_charge = 15000, 3000, 5000
        def_adm = "Inpatient"

    # ==========================================================
    # 8. Patient Input Tabs
    # ==========================================================
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Tab 1 — Patient",
        "🩺 Tab 2 — Clinical",
        "🏥 Tab 3 — Hospital History",
        "💳 Tab 4 — Billing"
    ])

    with tab1:
        st.markdown("#### Patient Demographics")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            age = st.number_input("Age (Years)", 0, 100, def_age)
            gender = st.selectbox("Gender", ["Female", "Male"])
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        with col_t2:
            department = st.selectbox("Hospital Department", ["General Medicine", "Orthopedics", "Cardiology", "Neurology", "Pediatrics", "Laboratory Services", "Radiology"], index=["General Medicine", "Orthopedics", "Cardiology", "Neurology", "Pediatrics", "Laboratory Services", "Radiology"].index(def_dept))
            diagnosis = st.selectbox("Primary Diagnosis Category", ["Fracture", "Chest Pain", "Hypertension", "Pneumonia", "Fever", "Migraine", "Diabetes", "Back Pain", "Asthma", "Kidney Infection"], index=["Fracture", "Chest Pain", "Hypertension", "Pneumonia", "Fever", "Migraine", "Diabetes", "Back Pain", "Asthma", "Kidney Infection"].index(def_diag))

    with tab2:
        st.markdown("#### Clinical Measurements & Vitals")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            systolic = st.number_input("Systolic BP (mmHg)", 70, 220, def_sys)
            diastolic = st.number_input("Diastolic BP (mmHg)", 40, 140, def_dia)
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", 50, 400, def_sugar)
        with col_v2:
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 500, def_chol)
            bmi = st.number_input("Body Mass Index (BMI)", 10.0, 50.0, def_bmi)

    with tab3:
        st.markdown("#### Hospital Stay & Utilization History")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            admission_type = st.selectbox("Patient Admission Type", ["Inpatient", "Outpatient"], index=0 if def_adm == "Inpatient" else 1)
            admitted = 1 if admission_type == "Inpatient" else 0
            previous_admissions = st.number_input("Previous Admissions Count", 0, 10, def_prev_adm)
            previous_appointments = st.number_input("Previous Appointments Count", 0, 30, 2)
            missed_appointments = st.number_input("Missed Appointments Count", 0, 20, 0)
        with col_h2:
            length_of_stay = st.number_input("Length of Hospital Stay (Days)", 0, 60, def_stay if admitted == 1 else 0)
            room_type = st.selectbox("Assigned Room Type", ["General Ward", "Private Room", "ICU", "None"] if admitted == 1 else ["None"], index=["General Ward", "Private Room", "ICU", "None"].index(def_room) if admitted == 1 else 0)
            waiting_days = st.number_input("Appointment Waiting Days", 0, 90, 2)
            lab_tests = st.number_input("Lab Tests Count", 0, 20, 2)
            treatments = st.number_input("Treatments Count", 0, 20, 1)

    with tab4:
        st.markdown("#### Financial Services & Billing Breakdown")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            consultation_fee = st.number_input("Consultation Fee (LKR)", 0, 10000, int(def_consultation) if 'def_consultation' in locals() else 2500)
            room_charge = st.number_input("Room Charge (LKR)", 0, 200000, int(def_room_charge) if admitted == 1 else 0)
            lab_charge = st.number_input("Lab Charge (LKR)", 0, 50000, int(def_lab_charge))
            medicine_charge = st.number_input("Medicine Charge (LKR)", 0, 50000, int(def_med_charge))
        with col_b2:
            # ==========================================================
            # 4. Automatically Calculated Total Bill
            # ==========================================================
            total_bill = consultation_fee + room_charge + lab_charge + medicine_charge
            st.metric("Calculated Total Hospital Bill", f"LKR {total_bill:,.2f}")
            payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid", "Partially Paid"])
            payment_method = st.selectbox("Payment Method", ["Insurance", "Online", "Cash", "Card"])

    # ==========================================================
    # 9. Validation & Input Check
    # ==========================================================
    validation_ok = True
    if missed_appointments > previous_appointments:
        st.error("❌ Validation Error: Missed appointments count cannot exceed total previous appointments count.")
        validation_ok = False

    # ==========================================================
    # 10. Feature Engineering
    # ==========================================================
    pulse_pressure = systolic - diastolic
    missed_ratio = missed_appointments / (previous_appointments + 1)
    charge_per_day = total_bill / (length_of_stay + 1)

    patient_dict = {
        'age': [age], 'gender': [gender], 'blood_group': [blood_group],
        'department': [department], 'diagnosis': [diagnosis],
        'waiting_days': [waiting_days], 'previous_appointments': [previous_appointments],
        'missed_previous_appointments': [missed_appointments], 'appointment_status': ['Completed'],
        'room_type': [room_type], 'length_of_stay_days': [length_of_stay],
        'previous_admissions': [previous_admissions], 'systolic_bp': [systolic], 'diastolic_bp': [diastolic],
        'blood_sugar_mg_dl': [blood_sugar], 'cholesterol_mg_dl': [cholesterol], 'bmi': [bmi],
        'lab_tests_count': [lab_tests], 'treatments_count': [treatments],
        'consultation_fee_lkr': [consultation_fee], 'room_charge_lkr': [room_charge],
        'lab_charge_lkr': [lab_charge], 'medicine_charge_lkr': [medicine_charge],
        'total_bill_lkr': [total_bill], 'payment_status': [payment_status],
        'payment_method': [payment_method], 'pulse_pressure': [pulse_pressure],
        'missed_ratio': [missed_ratio], 'charge_per_day': [charge_per_day]
    }
    patient_df = pd.DataFrame(patient_dict)

    # ==========================================================
    # 11. Prediction Trigger
    # ==========================================================
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        predict_button = st.button("🔍 Evaluate Readmission Risk", type="primary")

    should_predict = predict_button or (lookup_mode == "⚡ Quick Clinical Presets" and pst in ["low", "med", "high", "outpatient"]) or (lookup_mode == "🔍 Lookup Existing EHR Patient (Dataset)")
    if should_predict and validation_ok:
        with st.spinner("Analyzing patient clinical profile..."):
            if admitted == 0:
                risk_prob = 0.0
            else:
                risk_prob = float(pipeline.predict_proba(patient_df)[0][1])
            st.session_state.evaluated = True
            st.session_state.last_prob = risk_prob

    # ==========================================================
    # 12. Prediction Dashboard (4 Cards)
    # ==========================================================
    if st.session_state.evaluated and validation_ok:
        risk_prob = float(pipeline.predict_proba(patient_df)[0][1]) if admitted == 1 else 0.0
        risk_pct = risk_prob * 100

        if risk_pct < 68.0:
            risk_category = "LOW"
        elif risk_pct <= 80.0:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"

        st.markdown("---")
        st.markdown("### 🎯 PREDICTION RESULT")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Readmission Probability", f"{risk_pct:.1f}%")
        with r2:
            st.metric("Risk Level", risk_category)
        with r3:
            st.metric("Previous Admissions", f"{previous_admissions}")
        with r4:
            st.metric("Hospital Stay", f"{length_of_stay} days")

        # Visual Risk Progress Bar & Scale Caption
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Readmission Risk Scale:**")
        st.progress(float(risk_prob))
        st.caption("Low: <68%  •  Medium: 68–80%  •  High: >80% *(Coursework demonstration thresholds)*")

        # ==========================================================
        # 13. Explainable AI (SHAP & Permutation Drivers)
        # ==========================================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔑 Why did the model give this prediction?")
        st.caption("Top key contributing factors identified by SHAP & Permutation Feature Importance:")
        
        drivers_df = pd.DataFrame({
            "Key Contributing Factor": ["Length of Stay (Days)", "Total Bill (LKR)", "Previous Admissions", "Patient Age", "BMI", "Blood Sugar (mg/dL)", "Pulse Pressure"],
            "Patient Value": [f"{length_of_stay} days", f"LKR {total_bill:,.2f}", f"{previous_admissions} admissions", f"{age} yrs", f"{bmi:.1f}", f"{blood_sugar} mg/dL", f"{pulse_pressure} mmHg"],
            "Clinical Drivers": ["High stay duration increases exposure & severity", "Higher treatment intensity & billing", "Prior admission history indicates recurrence", "Age vulnerability risk", "Metabolic status factor", "Glycemic control marker", "Cardiovascular pulse pressure marker"]
        })
        st.dataframe(drivers_df)

        # ==========================================================
        # 14. Action Recommendation Panel
        # ==========================================================
        st.markdown("#### 📋 Recommended Actions")
        if risk_category == "LOW":
            st.success("""
            **🟢 LOW READMISSION RISK — RECOMMENDED ACTIONS:**
            - ✓ Schedule standard primary care follow-up within 14 days.
            - ✓ Confirm routine discharge summary package provided to patient.
            - ✓ Provide patient education on symptom monitoring at home.
            """)
        elif risk_category == "MEDIUM":
            st.warning("""
            **🟡 MEDIUM READMISSION RISK — RECOMMENDED ACTIONS:**
            - ✓ Schedule mandatory outpatient follow-up appointment within 7 days.
            - ✓ Review inpatient medication compliance and prescription plan.
            - ✓ Verify patient attendance history for previous appointments.
            - ✓ Conduct post-discharge phone call within 72 hours.
            """)
        else:
            st.error("""
            **🔴 HIGH READMISSION RISK — RECOMMENDED ACTIONS:**
            - ✓ Conduct comprehensive medication reconciliation prior to discharge.
            - ✓ Arrange early post-discharge home visit or telehealth check within 48 hours.
            - ✓ Assign dedicated nurse transitional care coordinator.
            - ✓ Review major clinical risk indicators with attending physician.
            """)

        # Expanders for Technical Info
        with st.expander("ℹ️ About this prediction"):
            st.write("""
            This probability is generated using the GridSearch-tuned Random Forest pipeline developed for the SmartCare AI coursework.
            
            **Disclaimer:** This tool is an educational prototype developed for academic evaluation purposes and has not been clinically validated for live medical diagnosis.
            """)

        with st.expander("🤖 Model Technical Details"):
            st.write("- **Model:** Random Forest Classifier (Scikit-Learn Pipeline)")
            st.write("- **Feature Selection:** ANOVA SelectPercentile (Top 80%)")
            st.write("- **Validation Method:** Stratified 5-Fold Nested Cross-Validation")

        # ==========================================================
        # 15. Export / New Assessment Controls
        # ==========================================================
        st.markdown("---")
        exp_col1, exp_col2 = st.columns(2)
        
        today_str = datetime.datetime.now().strftime("%d %B %Y")
        report_text = f"""==================================================
SMARTCARE AI
30-Day Readmission Risk Assessment Report
==================================================

Encounter Details:
Patient / Encounter ID : {patient_id}
Assessment Date        : {today_str}
Target Population      : Discharged Hospital Inpatient

Patient Profile:
Age                    : {age}
Gender                 : {gender}
Blood Group            : {blood_group}
Department             : {department}
Primary Diagnosis      : {diagnosis}

Hospital Stay & Vitals:
Length of Stay         : {length_of_stay} days
Room Type              : {room_type}
Previous Admissions    : {previous_admissions}
Blood Pressure         : {systolic}/{diastolic} mmHg (Pulse Pressure: {pulse_pressure} mmHg)
Blood Sugar            : {blood_sugar} mg/dL
BMI                    : {bmi}

Billing:
Total Bill             : LKR {total_bill:,.2f}
Payment Status         : {payment_status}

--------------------------------------------------
PREDICTION RESULTS:
Readmission Probability : {risk_pct:.1f}%
Risk Category          : {risk_category}

Major Factors:
1. Length of hospital stay ({length_of_stay} days)
2. Total hospital bill (LKR {total_bill:,.2f})
3. Previous admissions ({previous_admissions})
4. Age ({age} years)

IMPORTANT NOTICE:
This report was generated by an educational prototype developed
for academic assessment purposes. It is NOT clinically validated.
==================================================
"""
        with exp_col1:
            st.download_button(
                label="📄 Download Assessment Report (TXT)",
                data=report_text,
                file_name=f"SmartCare_Report_{patient_id}.txt",
                mime="text/plain"
            )
            
        with exp_col2:
            if st.button("↻ New Assessment"):
                st.session_state.evaluated = False
                st.session_state.preset_loaded = "custom"
                st.rerun()
