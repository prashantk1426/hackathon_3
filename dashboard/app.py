#importing all libraries

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- SETTINGS ----------------
st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/student_data.csv")

# ---------------- LOAD MODEL ----------------
model_files = glob.glob("models/saved_models/*.pkl")
model = joblib.load(max(model_files, key=lambda x: x))

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>🎓 Student Academic Risk Prediction System</h1>
""", unsafe_allow_html=True)

# 🔥 LONG DESCRIPTION (FIXED BACK)
st.markdown("""
<div style="text-align:center;font-size:18px;">
This dashboard uses machine learning to identify students who may be 
<strong>academically at risk</strong> based on study habits, academic performance,
and lifestyle patterns.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- TOGGLE ----------------
if "show_info" not in st.session_state:
    st.session_state.show_info = False

col1, col2 = st.columns([8,1])
with col2:
    if st.button("ℹ More Info"):
        st.session_state.show_info = not st.session_state.show_info

# ---------------- INPUT ----------------
st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance (%)", 40, 100)
    study_hours = st.slider("Study Hours", 1, 6)
    assignment = st.slider("Assignment Score", 40, 100)

with col2:
    gpa = st.slider("Previous GPA", 0.0, 10.0)
    participation = st.slider("Participation", 0, 100)
    sleep = st.slider("Sleep Hours", 4, 9)

# ---------------- PREDICTION ----------------
if st.button("Predict Academic Risk"):

    features = np.array([[attendance, study_hours, assignment, gpa, participation, sleep]])
    prediction = model.predict(features)[0]

    try:
        prob = model.predict_proba(features)[0][1]
        confidence = round(prob * 100, 2)
    except:
        confidence = None

    st.markdown("---")

    if prediction == 1:
        st.error("⚠ Student is Academically At Risk")
    else:
        st.success("✅ Student is Academically Safe")

    if confidence:
        st.info(f"📊 Risk Probability: {confidence}%")

    # Recommendations
    st.markdown("### 💡 Recommendations")

    rec = []
    if attendance < 60: rec.append("Improve attendance above 75%")
    if study_hours < 3: rec.append("Increase study hours")
    if gpa < 6: rec.append("Improve GPA")
    if participation < 50: rec.append("Participate more")
    if sleep < 6: rec.append("Maintain healthy sleep")

    if rec:
        for r in rec:
            st.write("📌", r)
    else:
        st.success("Great performance! Keep it up.")

# ---------------- MORE INFO ----------------
if st.session_state.show_info:

    st.markdown("---")

    # Dataset Overview
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Average GPA", round(df["previous_gpa"].mean(),2))
    col3.metric("Average Attendance", round(df["attendance"].mean(),2))

    st.markdown("---")

    # ---------------- GRAPHS (UNCHANGED PROFESSIONAL STYLE) ----------------
    st.header("📈 Data Insights")

    col1, col2 = st.columns(2)

    # Attendance
    with col1:
        fig, ax = plt.subplots(figsize=(5,3))
        bins = [40,50,60,70,80,90,100]
        ax.hist(df["attendance"], bins=bins, color="#4FA3FF", edgecolor="white")

        ax.set_title("Attendance Distribution", color="white")
        ax.set_xlabel("Attendance (%)", color="white")
        ax.set_ylabel("Frequency", color="white")

        ax.tick_params(colors='white')
        ax.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")

        st.pyplot(fig)

    # Study Hours
    with col2:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x="study_hours", data=df, color="#4FA3FF", ax=ax)

        ax.set_title("Study Hours Distribution", color="white")
        ax.set_xlabel("Study Hours", color="white")
        ax.set_ylabel("Frequency", color="white")

        ax.tick_params(colors='white')
        ax.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")

        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # GPA
    with col3:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(df["previous_gpa"], bins=10, kde=True,
                     color="#4FA3FF", edgecolor="white", ax=ax)

        ax.set_xlim(0,10)

        ax.set_title("GPA Distribution", color="white")
        ax.set_xlabel("GPA (0–10)", color="white")
        ax.set_ylabel("Frequency", color="white")

        ax.tick_params(colors='white')
        ax.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")

        st.pyplot(fig)

    # Sleep
    with col4:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(df["sleep_hours"], bins=6, kde=True,
                     color="#4FA3FF", edgecolor="white", ax=ax)

        ax.set_title("Sleep Hours Distribution", color="white")
        ax.set_xlabel("Sleep Hours", color="white")
        ax.set_ylabel("Frequency", color="white")

        ax.tick_params(colors='white')
        ax.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")

        st.pyplot(fig)

    st.markdown("---")

    # 🔥 PROPER EXPLANATION (VISIBLE + CLEAN)
    st.header("📘 Project Explanation")

    st.markdown("""
### 🔍 What This System Does
This system predicts whether a student is academically at risk using machine learning.

---

### 🧠 How It Works
• Data is collected  
• Models are trained  
• Best model is selected  
• Prediction is made  
• Probability + recommendations are shown  

---

### 📊 Features Used
• Attendance  
• Study Hours  
• GPA  
• Participation  
• Sleep  

---

### 📈 Graph Meaning
X-axis = values  
Y-axis = number of students  

---

### ⚙ Workflow
Dataset → Training → Model → Dashboard
""")

    st.caption("Hackathon 3 • Student Academic Risk Prediction System")

# final update