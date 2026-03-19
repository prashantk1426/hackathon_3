# 🎓 Student Academic Risk Prediction System

## 📸 Dashboard Preview
<img width="1919" height="935" alt="Screenshot 2026-03-19 122328" src="https://github.com/user-attachments/assets/8fa60036-fae4-4385-9079-ab36d9362c88" />


---

## 📌 Project Overview
This project predicts whether a student is academically at risk using machine learning.

It analyzes student performance based on:
- Attendance
- Study Hours
- Assignment Scores
- Previous GPA
- Participation
- Sleep Hours

The goal is to identify students early and provide recommendations for improvement.

---

## 🚀 Features
- Predict student academic risk (Safe / At Risk)
- Interactive dashboard using Streamlit
- Data visualization (graphs & distributions)
- Smart recommendations based on input
- Probability score (model confidence)

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- SQLite (Database)

---

## ⚙️ How to Run the Project

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python -m training.train_model

### 3. Run the dashboard
streamlit run dashboard/app.py

---

## 📂 Project Structure
hackathon_3/
│
├── dashboard/        # Streamlit dashboard
├── training/         # Model training code
├── database/         # Database handling
├── models/           # Saved models
├── data/             # Dataset
├── requirements.txt
└── README.md

---

## 🧠 Machine Learning Approach
- Models used:
  - Logistic Regression
  - Decision Tree
  - Random Forest

- Best model selected based on accuracy

- Output:
  - Risk prediction (Safe / At Risk)
  - Probability score

---

## 🧠 How it works
The system takes student input, processes it using a trained ML model, and predicts risk along with recommendations.

---

## 📊 Dashboard
The dashboard allows users to:
- Input student details
- Get prediction results
- View probability score
- Get improvement suggestions
- View data graphs

---

## 🎯 Use Case
This system can be used in:
- Schools & Colleges
- Academic monitoring systems
- Early intervention programs

---

## 👨‍💻 Author
Prashant Kumar  
Admission No: 2024SEPVUGP0005  
Course: B.Tech AI  

---

## ⭐ Note
This project demonstrates a basic end-to-end Machine Learning pipeline (MLOps), including data processing, model training, and deployment.
