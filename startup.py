# ─────────────────────────────────────────────────────────────
# startup.py — Auto Setup for Streamlit Cloud Deployment
# Employee Attrition Prediction | BCA Major Project
# Developer: Patwa Faizan
#
# This file is called by app.py on first load.
# It checks if dataset and model files exist.
# If not → generates dataset → trains models → saves .pkl files.
# On next load everything is ready instantly.
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import joblib

def run_setup():
    """
    Returns True if setup was needed and completed.
    Returns False if all files already exist (skip setup).
    """

    required_files = [
        "employee_data_10000.csv",
        "logistic_model.pkl",
        "random_forest_model.pkl",
        "scaler.pkl",
        "columns.pkl"
    ]

    all_exist = all(os.path.exists(f) for f in required_files)
    if all_exist:
        return False  # nothing to do

    print("=" * 55)
    print("  STARTUP SETUP — Streamlit Cloud Deployment")
    print("  Generating dataset and training models...")
    print("=" * 55)

    # ─────────────────────────────────────────────
    # STEP 1 — GENERATE DATASET (10,000 records)
    # ─────────────────────────────────────────────
    if not os.path.exists("employee_data_10000.csv"):
        print("\n[1] Generating HR dataset (10,000 records)...")
        _generate_dataset()
        print("    Done — employee_data_10000.csv saved.")
    else:
        print("\n[1] Dataset already exists — skipping generation.")

    # ─────────────────────────────────────────────
    # STEP 2 — TRAIN MODELS
    # ─────────────────────────────────────────────
    pkl_missing = any(not os.path.exists(f) for f in required_files[1:])
    if pkl_missing:
        print("\n[2] Training ML models...")
        _train_models()
        print("    Done — all .pkl files saved.")
    else:
        print("\n[2] Model files already exist — skipping training.")

    print("\n" + "=" * 55)
    print("  SETUP COMPLETE — App is ready!")
    print("=" * 55)
    return True


# ─────────────────────────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────────────────────────
def _generate_dataset():
    np.random.seed(42)
    n = 10000

    age                       = np.random.randint(22, 58, n)
    gender                    = np.random.choice(["Male","Female"], n, p=[0.6,0.4])
    department                = np.random.choice(["Sales","Research & Development","Human Resources"], n, p=[0.35,0.55,0.10])
    job_role                  = np.random.choice(["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"], n)
    education                 = np.random.randint(1, 6, n)
    monthly_income            = np.random.randint(1500, 20000, n)
    job_satisfaction          = np.random.randint(1, 5, n)
    work_life_balance         = np.random.randint(1, 5, n)
    overtime_raw              = np.random.choice([0, 1], n, p=[0.68, 0.32])
    years_at_company          = np.random.randint(0, 35, n)
    years_in_current_role     = np.clip(np.random.randint(0, 18, n), 0, years_at_company)
    years_since_last_promotion= np.clip(np.random.randint(0, 15, n), 0, years_at_company)
    years_with_curr_manager   = np.clip(np.random.randint(0, 17, n), 0, years_at_company)
    total_working_years       = np.clip(years_at_company + np.random.randint(0, 15, n), years_at_company, 40)
    num_companies_worked      = np.random.randint(0, 10, n)
    distance_from_home        = np.random.randint(1, 30, n)
    environment_satisfaction  = np.random.randint(1, 5, n)
    job_involvement           = np.random.randint(1, 5, n)
    performance_rating        = np.random.choice([3, 4], n, p=[0.85, 0.15])
    relationship_satisfaction = np.random.randint(1, 5, n)
    stock_option_level        = np.random.randint(0, 4, n)
    training_times_last_year  = np.random.randint(0, 7, n)
    business_travel           = np.random.choice(["Non-Travel","Travel_Rarely","Travel_Frequently"], n, p=[0.19,0.71,0.10])

    # ── Attrition probability based on HR research signals ──
    attrition_score = np.zeros(n, dtype=float)

    # Overtime — strongest predictor
    attrition_score += overtime_raw * 0.30

    # Low job satisfaction
    attrition_score += np.where(job_satisfaction == 1, 0.20,
                       np.where(job_satisfaction == 2, 0.12,
                       np.where(job_satisfaction == 3, 0.04, 0.0)))

    # Low work life balance
    attrition_score += np.where(work_life_balance == 1, 0.15,
                       np.where(work_life_balance == 2, 0.08, 0.0))

    # Low income
    income_norm = (monthly_income - 1500) / (20000 - 1500)
    attrition_score += (1 - income_norm) * 0.15

    # Short tenure
    attrition_score += np.where(years_at_company <= 2, 0.15,
                       np.where(years_at_company <= 5, 0.07, 0.0))

    # No recent promotion
    attrition_score += np.where(years_since_last_promotion >= 5, 0.10,
                       np.where(years_since_last_promotion >= 3, 0.06, 0.0))

    # Job hopper
    attrition_score += np.where(num_companies_worked >= 5, 0.08, 0.0)

    # Long commute
    attrition_score += np.where(distance_from_home >= 20, 0.06, 0.0)

    # Sales department slightly higher
    attrition_score += np.where(department == "Sales", 0.04, 0.0)

    # Add noise
    attrition_score += np.random.normal(0, 0.05, n)
    attrition_score  = np.clip(attrition_score, 0, 1)

    attrition = np.where(attrition_score > 0.45, "Yes", "No")

    df = pd.DataFrame({
        "Age":                     age,
        "Gender":                  gender,
        "Department":              department,
        "JobRole":                 job_role,
        "Education":               education,
        "MonthlyIncome":           monthly_income,
        "JobSatisfaction":         job_satisfaction,
        "WorkLifeBalance":         work_life_balance,
        "OverTime":                np.where(overtime_raw == 1, "Yes", "No"),
        "YearsAtCompany":          years_at_company,
        "YearsInCurrentRole":      years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager":    years_with_curr_manager,
        "TotalWorkingYears":       total_working_years,
        "NumCompaniesWorked":      num_companies_worked,
        "DistanceFromHome":        distance_from_home,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement":          job_involvement,
        "PerformanceRating":       performance_rating,
        "RelationshipSatisfaction":relationship_satisfaction,
        "StockOptionLevel":        stock_option_level,
        "TrainingTimesLastYear":   training_times_last_year,
        "BusinessTravel":          business_travel,
        "Attrition":               attrition,
        "EmployeeCount":           1,
        "StandardHours":           80,
        "EmployeeNumber":          np.arange(1, n+1),
    })

    df.to_csv("employee_data_10000.csv", index=False)


# ─────────────────────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────────────────────
def _train_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("employee_data_10000.csv")

    # ── Feature engineering ──
    df["OverTime"]       = df["OverTime"].map({"Yes": 1, "No": 0})
    df["Attrition_Label"]= df["Attrition"].map({"Yes": 1, "No": 0})

    df["SatisfactionScore"] = (df["JobSatisfaction"] + df["WorkLifeBalance"]) / 2
    df["IncomePerYear"]     = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["YearsPerCompany"]   = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["ExperienceGap"]     = df["TotalWorkingYears"] - df["YearsAtCompany"]
    df["LoyaltyScore"]      = df["YearsAtCompany"] - df["YearsWithCurrManager"]
    df["OverTimeFlag"]      = df["OverTime"]

    FEATURES = [
        "Age", "MonthlyIncome", "OverTime", "JobSatisfaction",
        "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion",
        "DistanceFromHome", "TotalWorkingYears", "NumCompaniesWorked",
        "YearsInCurrentRole", "YearsWithCurrManager",
        "SatisfactionScore", "IncomePerYear", "YearsPerCompany",
        "ExperienceGap", "LoyaltyScore", "OverTimeFlag"
    ]

    X = df[FEATURES]
    y = df["Attrition_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale for Logistic Regression ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Logistic Regression ──
    print("    Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    print(f"    LR Accuracy: {lr_acc:.4f}")

    # ── Random Forest ──
    print("    Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"    RF Accuracy: {rf_acc:.4f}")

    # ── Save all files ──
    joblib.dump(lr,      "logistic_model.pkl")
    joblib.dump(rf,      "random_forest_model.pkl")
    joblib.dump(scaler,  "scaler.pkl")
    joblib.dump(FEATURES,"columns.pkl")
    print("    Saved: logistic_model.pkl")
    print("    Saved: random_forest_model.pkl")
    print("    Saved: scaler.pkl")
    print("    Saved: columns.pkl")


# ─────────────────────────────────────────────────────────────
# Run directly if called as script
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_setup()
    print("\nAll files ready. You can now run: streamlit run app.py")