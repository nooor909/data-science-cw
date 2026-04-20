import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Grant Application Predictor", layout="centered")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")

model = joblib.load(MODEL_PATH)
model_columns = model.feature_names_in_

st.title("Grant Application Approval Predictor")
st.write("Enter the project details below to predict whether the application is likely to be approved.")

teacher_prefix = st.number_input("Teacher Prefix (encoded)", value=0.0)
project_grade_category = st.number_input("Project Grade Category (encoded)", value=0)
teacher_number_of_previously_posted_projects = st.number_input(
    "Teacher Number of Previously Posted Projects", value=0.0
)
submission_year = st.number_input("Submission Year", value=2017)
submission_month = st.number_input("Submission Month", min_value=1, max_value=12, value=1)
submission_weekday = st.number_input("Submission Weekday", min_value=0, max_value=6, value=0)
experience_bin = st.number_input("Experience Bin", value=0)

# Encoded state columns
school_state_0 = st.number_input("school_state_0", value=0)
school_state_1 = st.number_input("school_state_1", value=0)
school_state_2 = st.number_input("school_state_2", value=0)
school_state_3 = st.number_input("school_state_3", value=0)
school_state_4 = st.number_input("school_state_4", value=0)
school_state_5 = st.number_input("school_state_5", value=0)

# Encoded subject category columns
project_subject_categories_0 = st.number_input("project_subject_categories_0", value=0)
project_subject_categories_1 = st.number_input("project_subject_categories_1", value=0)
project_subject_categories_2 = st.number_input("project_subject_categories_2", value=0)
project_subject_categories_3 = st.number_input("project_subject_categories_3", value=0)
project_subject_categories_4 = st.number_input("project_subject_categories_4", value=0)
project_subject_categories_5 = st.number_input("project_subject_categories_5", value=0)

if st.button("Predict"):
    input_data = {
        "teacher_prefix": teacher_prefix,
        "school_state_0": school_state_0,
        "school_state_1": school_state_1,
        "school_state_2": school_state_2,
        "school_state_3": school_state_3,
        "school_state_4": school_state_4,
        "school_state_5": school_state_5,
        "project_grade_category": project_grade_category,
        "project_subject_categories_0": project_subject_categories_0,
        "project_subject_categories_1": project_subject_categories_1,
        "project_subject_categories_2": project_subject_categories_2,
        "project_subject_categories_3": project_subject_categories_3,
        "project_subject_categories_4": project_subject_categories_4,
        "project_subject_categories_5": project_subject_categories_5,
        "teacher_number_of_previously_posted_projects": teacher_number_of_previously_posted_projects,
        "submission_year": submission_year,
        "submission_month": submission_month,
        "submission_weekday": submission_weekday,
        "experience_bin": experience_bin
    }

    input_df = pd.DataFrame([input_data])

    # Force exact training column order
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write("Prediction:", "Approved" if prediction == 1 else "Not Approved")
    st.write("Approval Probability:", round(float(probability), 4))