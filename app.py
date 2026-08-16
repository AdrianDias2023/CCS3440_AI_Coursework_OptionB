# Save this code as `app.py` and run using: streamlit run app.py
import streamlit as st
import pandas as pd
import joblib

# Page layout configuration
st.set_page_config(page_title="SmartCare AI Predictor", page_icon="🏥", layout="centered")

st.title("🏥 SmartCare Readmission Risk Calculator")
st.write("Clinical Decision Support Dashboard - Artificial Intelligence Coursework (Option B)")

# Sidebar: Model metadata information
st.sidebar.title("Model Information")
st.sidebar.markdown("""
**Target Population:**
* Discharged Hospital Inpatients (`admitted == 1`)

**Prediction Target:**
* 30-day Hospital Readmission (`readmitted_30_days`)

**Pipeline Classifier:**
* Random Forest Classifier (Optimized)

**Pipeline Features Selection:**
* SelectPercentile (ANOVA F-test, Top 80%)

**Evaluation Method:**
* Stratified 5-Fold Nested CV & Holdout Test Set ($N=66$)

**XAI & Fairness:**
* Permutation Importance, SHAP, Subgroup Metrics

**Environment Dependency:**
* `scikit-learn == 1.5.0`

---
**Group Members:**
* 23UG1-0018
* 23UG1-0151
* 23UG1-0069
* CIT-23-02-0091
* 23UG1-0072
""")

# Load production model pipeline
@st.cache_resource
def load_model_pipeline():
    return joblib.load("best_readmission_rf_model.joblib")

try:
    pipeline = load_model_pipeline()
    model_ok = True
except FileNotFoundError:
    st.error("⚠️ Model pipeline file 'best_readmission_rf_model.joblib' was not found. Please train and save the model in the notebook first.")
    model_ok = False
except AttributeError as e:
    st.error(f"⚠️ Model Deserialization Error: {str(e)}\n\nThis model was serialized using **scikit-learn 1.5.0**. Running on a different scikit-learn version can cause deserialization failures due to internal namespace changes. Please install the exact version using: `pip install scikit-learn==1.5.0`")
    model_ok = False
except Exception as e:
    st.error(f"⚠️ Unexpected Error loading model: {str(e)}")
    model_ok = False

if model_ok:
    st.markdown("---")
    st.header("Patient Clinical Profile Inputs")
    
    # Inputs split into two columns
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (Years)", 0, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A-", "B-", "B+", "AB-", "O+", "A+", "AB+", "O-"])
        department = st.selectbox("Department", ["General Medicine", "Orthopedics", "Cardiology", "Neurology", "Pediatrics", "Laboratory Services", "Radiology"])
        diagnosis = st.selectbox("Primary Diagnosis", ["Migraine", "Diabetes", "Back Pain", "Asthma", "Hypertension", "Fracture", "Kidney Infection", "Fever", "Chest Pain", "Pneumonia"])
        appointment_status = st.selectbox("Appointment Status", ["Completed", "No-Show", "Scheduled", "Cancelled"])
        payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid", "Partially Paid"])
        payment_method = st.selectbox("Payment Method", ["Insurance", "Online", "Cash", "Card"])
        
    with col2:
        admitted = st.radio("Admitted? (0=No, 1=Yes)", [0, 1], index=1)
        
        # Interactive room type
        room_options = ["None", "General Ward", "Private Room", "ICU"]
        room_type = st.selectbox("Assigned Room Type", room_options, index=1 if admitted==1 else 0)
        
        # Interactive length of stay
        length_of_stay = st.slider("Hospital Stay (Days)", 0, 30, 3 if admitted==1 else 0)
        
        previous_admissions = st.slider("Previous Admissions Count", 0, 10, 0)
        previous_appointments = st.slider("Previous Appointments Count", 0, 20, 2)
        missed_appointments = st.slider("Missed Previous Appointments", 0, 10, 0)
        
        st.markdown("**Clinical Vitals**")
        systolic = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        diastolic = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
        blood_sugar = st.slider("Blood Sugar (mg/dL)", 50, 300, 100)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 190)
        bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 24.5)

    st.header("Billing & Services")
    col3, col4 = st.columns(2)
    with col3:
        lab_tests = st.slider("Lab Tests Count", 0, 15, 2)
        treatments = st.slider("Treatments Count", 0, 15, 1)
        waiting_days = st.slider("Waiting Days", 0, 60, 10)
    with col4:
        consultation_fee = st.number_input("Consultation Fee (LKR)", min_value=0, value=2500)
        room_charge = st.number_input("Room Charge (LKR)", min_value=0, value=15000 if admitted==1 else 0)
        lab_charge = st.number_input("Lab Charge (LKR)", min_value=0, value=3000)
        medicine_charge = st.number_input("Medicine Charge (LKR)", min_value=0, value=5000)
        total_bill = st.number_input("Total Bill (LKR)", min_value=0, value=int(consultation_fee + room_charge + lab_charge + medicine_charge))

    # Form Validation and Impossible State Prevention
    validation_ok = True
    
    # 1. Admitted = No checks
    if admitted == 0:
        st.info("ℹ️ Note: This predictive model is trained on discharged hospital inpatients. Outpatients (Admitted=No) do not have hospital admissions and have 0.00% inpatient readmission risk.")
        if length_of_stay > 0 or room_type != "None" or room_charge > 0:
            st.warning("⚠️ Operational Constraint: Outpatients (Admitted=No) cannot have a hospital stay length > 0, room charges, or room assignments. Overriding these inputs to 0 / 'None'.")
            length_of_stay = 0
            room_type = "None"
            room_charge = 0
            # Recalculate total bill
            total_bill = int(consultation_fee + lab_charge + medicine_charge)
            
    # 2. Missed appointments count check
    if missed_appointments > previous_appointments:
        st.error("❌ Input Error: Missed appointments cannot exceed the total previous appointments count.")
        validation_ok = False



    # Apply engineered calculations
    pulse_pressure = systolic - diastolic
    missed_ratio = missed_appointments / (previous_appointments + 1)
    charge_per_day = total_bill / (length_of_stay + 1)

    # Prepare prediction dictionary
    patient_dict = {
        'age': [age], 'gender': [gender], 'blood_group': [blood_group],
        'department': [department], 'diagnosis': [diagnosis],
        'waiting_days': [waiting_days], 'previous_appointments': [previous_appointments],
        'missed_previous_appointments': [missed_appointments], 'appointment_status': [appointment_status],
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
    
    st.markdown("### Prediction Evaluation")
    if st.button("📊 Evaluate Patient Risk"):
        if validation_ok:
            if admitted == 0:
                risk_prob = 0.0
                risk_category = "Low"
                alert_func = st.success
                alert_msg = "✅ **Outpatient Status:** Outpatients have zero hospital stay duration and zero 30-day inpatient readmission risk."
            else:
                risk_prob = pipeline.predict_proba(patient_df)[0][1]
                
                # Determine Risk Category
                if risk_prob < 0.30:
                    risk_category = "Low"
                    alert_func = st.success
                    alert_msg = "✅ **Low Risk:** The patient has a low probability of readmission. Standard discharge protocol is sufficient."
                elif risk_prob <= 0.70:
                    risk_category = "Medium"
                    alert_func = st.warning
                    alert_msg = "⚠️ **Medium Risk Warning:** The patient shows moderate readmission indicators. Recommend arranging a standard follow-up appointment within 7 days and verifying outpatient compliance barriers."
                else:
                    risk_category = "High"
                    alert_func = st.error
                    alert_msg = "🚨 **High Risk Clinical Alert:** The patient is highly likely to be readmitted within 30 days. It is strongly recommended to review the discharge checklist, reconcile medications, and schedule a 48-hour post-discharge follow-up."
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(label="Patient Readmission Risk Probability", value=f"{risk_prob * 100:.2f}%")
            with col_m2:
                st.metric(label="Risk Category", value=risk_category)
                
            # Render clinical alert
            alert_func(alert_msg)
        else:
            st.error("Please resolve the input validation errors before evaluating risk.")
