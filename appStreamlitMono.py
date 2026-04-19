import streamlit as st
import joblib
import pandas as pd

# Load the machine learning model and encode
clf_model = joblib.load("classification_model.pkl")
reg_model = joblib.load("regression_model.pkl")

st.set_page_config(
    page_title="Student Placement and Salary Prediction",
    page_icon="🎓",
    layout="wide"
)

def main():
    with st.sidebar:
        st.title("🎓 Student Placement & Salary Prediction")
        st.caption("Predict placement status and estimated salary package.")
        st.info("Fill in the student information, then click the prediction button to start the prediction.")
        st.divider()
        st.write("**Creator:**")
        st.write("Muhammad Akhdan Athallah")
        st.write("2802446560")
        st.write("")
        st.write("")
        st.write("**Placement Prediction Model:**")
        st.write("🚀 XGBoost Classifier")
        st.write("")
        st.write("")
        st.write("**Salary Prediction Model:**")
        st.write("📈 Linear Regression")

    with st.form("prediction_form"):
        st.subheader("Student Information")

        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

        col1, col2 = st.columns(2)

        with col1:
            ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, step=0.01)
            hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, step=0.01)
            degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, step=0.01)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
            entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, step=0.01)
            technical_skill_score = st.number_input("Technical Skill Score", min_value=0.0, max_value=100.0, step=0.01)
            soft_skill_score = st.number_input("Soft Skill Score", min_value=0.0, max_value=100.0, step=0.01)

        with col2:
            internship_count = st.slider("Internship Count", min_value=0, max_value=10, step=1)
            live_projects = st.slider("Live Projects", min_value=0, max_value=10, step=1)
            work_experience_months = st.number_input("Work Experience (Months)", min_value=0, max_value=72, step=1)
            certifications = st.slider("Certifications", min_value=0, max_value=10, step=1)
            attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.01)
            backlogs = st.slider("Backlogs", min_value=0, max_value=10, step=1)
            extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        submit = st.form_submit_button("Predict Placement & Salary", use_container_width=True)

    if submit:
        input_df = pd.DataFrame([{
            "gender": gender,
            "ssc_percentage": ssc_percentage,
            "hsc_percentage": hsc_percentage,
            "degree_percentage": degree_percentage,
            "cgpa": cgpa,
            "entrance_exam_score": entrance_exam_score,
            "technical_skill_score": technical_skill_score,
            "soft_skill_score": soft_skill_score,
            "internship_count": internship_count,
            "live_projects": live_projects,
            "work_experience_months": work_experience_months,
            "certifications": certifications,
            "attendance_percentage": attendance_percentage,
            "backlogs": backlogs,
            "extracurricular_activities": extracurricular_activities
        }])

        placement_pred = clf_model.predict(input_df)[0]

        st.divider()

        st.header("**Prediction Result**")

        if placement_pred == 1:
            salary_pred = reg_model.predict(input_df)[0]
            st.success("Placement Status: Placed")
            st.metric("Predicted Salary Package:", f"{salary_pred:.2f} LPA")
            st.caption("*LPA stands for Lakhs Per Annum.")
        else:
            st.error("Placement Status: Not Placed")
            st.metric("Predicted Salary Package:", "Not Available")
            st.caption("*Salary prediction is only provided for students classified as Placed.")

if __name__ == "__main__":
    main()
