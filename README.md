# 👨‍💼 Employee Attrition Prediction & HR Analytics Platform

An end-to-end Machine Learning and Data Analytics project designed to predict employee attrition, uncover organizational trends, and provide strategic decision-support tools for Human Resources. 

**Live Application:** [View the Streamlit App Here](https://employeeattritionproject-u5sup4llhb6xrmgvdwvzpz.streamlit.app)

---

## 🎯 Project Overview
Employee attrition is one of the most critical and financially significant challenges modern organizations face, with replacement costs ranging from 50% to 200% of an employee's annual salary. This project replaces reactive HR monitoring with a proactive, data-driven early warning system.

By analyzing 12 key HR parameters, this system utilizes a **Random Forest Classifier** (achieving **81.2% accuracy** and **85.6% recall**) to predict flight risk and prescribe actionable retention strategies.

## ✨ Key Features
* **Real-Time Prediction Engine:** Input an employee's profile to instantly receive a probability risk score and a Low/Moderate/High risk classification.
* **Automated HR Recommendations:** The system generates context-aware retention advice based on the specific risk factors flagged by the model.
* **What-If Simulator:** Allows HR to test interventions (e.g., removing overtime or offering a raise) and immediately see how much the attrition risk drops.
* **Attrition Cost Calculator:** Estimates the exact financial impact (recruitment, onboarding, productivity loss) of losing an employee to calculate Retention ROI.
* **Interactive EDA Dashboard:** Visual deep-dives into 10,000 HR records, uncovering insights like the severe impact of overtime and salary stagnation on attrition.

## 🛠️ Technology Stack
* **Core Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `LogisticRegression`, `StandardScaler`)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud, GitHub

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Faizan49-tech/EmployeeAttritionProject.git](https://github.com/Faizan49-tech/EmployeeAttritionProject.git)
   cd EmployeeAttritionProject

2. Install the required dependencies:

pip install -r requirements.txt

3. Run the Streamlit application:

streamlit run app.py

Note: If the .pkl model files or the dataset are missing locally, the custom startup.py script will automatically generate the 10,000-record dataset and retrain the models upon the first launch.

📊 Model Performance
The Random Forest model was selected as the primary predictive engine due to its superior ability to handle the non-linear realities of HR data.

Accuracy: 81.2%

Recall (Sensitivity): 85.6%

Precision: 78.4%

F1-Score: 81.8%

👨‍💻 Developer
Patwa Faizan Akhtar Hussain Bachelor of Computer Application (BCA) Final Year Narmada College of Science and Commerce (VNSGU)

Academic Year: 2025–2026