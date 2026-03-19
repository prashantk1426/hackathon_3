# 🎓 Student Academic Risk Prediction System

## 📸 Dashboard Preview
<img width="1919" height="935" alt="Screenshot 2026-03-19 122328" src="https://github.com/user-attachments/assets/8fa60036-fae4-4385-9079-ab36d9362c88" />

This is the interactive dashboard used for predicting student academic risk in real-time.


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

<img width="1536" height="1024" alt="repo_fc" src="https://github.com/user-attachments/assets/bc6a3ab5-ae9d-47d0-b638-cac078e850b6" />


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

## 🔄 Pipeline
Data → Preprocessing → Model Training → Prediction → Dashboard


<img width="1024" height="1536" alt="flwchrt" src="https://github.com/user-attachments/assets/dc148356-2b2f-47d9-b7ed-972c96434736" />



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

<img width="1841" height="822" alt="Screenshot 2026-03-19 221848" src="https://github.com/user-attachments/assets/fbf6d277-bab8-4cd0-9df6-e04e8c4db1a1" />

<img width="1748" height="637" alt="Screenshot 2026-03-19 221040" src="https://github.com/user-attachments/assets/791e3d15-a1b1-44fa-a9dc-6b7c85d8082b" />

<img width="1753" height="602" alt="Screenshot 2026-03-19 221021" src="https://github.com/user-attachments/assets/9610671c-57f9-45d8-bb76-7a5ceed3f5e6" />




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

---

## ⭐ Note
This project demonstrates a basic end-to-end Machine Learning pipeline (MLOps), including data processing, model training, and deployment.

---

## 📊 Future Improvements
- Add more features
- Improve model accuracy
- improve UI
- Deploy online
