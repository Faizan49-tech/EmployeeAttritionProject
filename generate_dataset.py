import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────
# ROOT CAUSE FIX:
#
# The old generate_dataset.py used GaussianCopulaSynthesizer
# which copies statistical distributions of each column
# INDEPENDENTLY — it does NOT preserve the relationship
# between features and the Attrition column.
#
# Result: OverTime=Yes had 15.8% attrition, same as No.
# The model trained on pure noise and always predicted ~16%
# regardless of what inputs you gave it.
#
# This script generates 10,000 rows where attrition is
# determined by REAL HR logic based on IBM HR Analytics
# research — so the model actually learns something useful.
# ─────────────────────────────────────────────────────────

np.random.seed(42)
n = 10000

print("Generating dataset with realistic attrition signal...")

# ── STEP 1: Generate feature distributions ───────────────
age                       = np.random.randint(18, 61, n)
department                = np.random.choice(
    ["Sales", "Research & Development", "Human Resources"],
    n, p=[0.30, 0.65, 0.05]
)
education                 = np.random.randint(1, 6, n)
education_field           = np.random.choice(
    ["Life Sciences","Medical","Marketing",
     "Technical Degree","Human Resources","Other"], n
)
environment_satisfaction  = np.random.randint(1, 5, n)
gender                    = np.random.choice(["Male","Female"], n, p=[0.60, 0.40])
job_involvement           = np.random.randint(1, 5, n)
job_level                 = np.random.randint(1, 6, n)
job_role                  = np.random.choice([
    "Sales Executive","Research Scientist","Laboratory Technician",
    "Manufacturing Director","Healthcare Representative","Manager",
    "Sales Representative","Research Director","Human Resources"
], n)
job_satisfaction          = np.random.randint(1, 5, n)
marital_status            = np.random.choice(
    ["Single","Married","Divorced"], n, p=[0.32, 0.46, 0.22]
)
monthly_income            = np.random.randint(2000, 20000, n)
monthly_rate              = np.random.randint(2000, 27000, n)
daily_rate                = np.random.randint(100, 1500, n)
hourly_rate               = np.random.randint(30, 100, n)
num_companies_worked      = np.random.randint(0, 10, n)
overtime                  = np.random.choice(["Yes","No"], n, p=[0.28, 0.72])
percent_salary_hike       = np.random.randint(11, 26, n)
performance_rating        = np.random.choice([3, 4], n, p=[0.85, 0.15])
relationship_satisfaction = np.random.randint(1, 5, n)
stock_option_level        = np.random.randint(0, 4, n)
total_working_years       = np.clip(np.random.randint(0, 41, n), 0, 40)
training_times_last_year  = np.random.randint(0, 7, n)
work_life_balance         = np.random.randint(1, 5, n)
years_at_company          = np.minimum(np.random.randint(0, 41, n), total_working_years)
years_in_current_role     = np.minimum(np.random.randint(0, 19, n), years_at_company)
years_since_last_promo    = np.minimum(np.random.randint(0, 16, n), years_at_company)
years_with_curr_manager   = np.minimum(np.random.randint(0, 18, n), years_at_company)
distance_from_home        = np.random.randint(1, 30, n)
business_travel           = np.random.choice(
    ["Travel_Rarely","Travel_Frequently","Non-Travel"],
    n, p=[0.71, 0.19, 0.10]
)
employee_count            = np.ones(n, dtype=int)
employee_number           = np.arange(1, n + 1)
over18                    = np.array(["Y"] * n)
standard_hours            = np.array([80] * n)

# ── STEP 2: Build attrition probability with real HR logic ─
prob = np.zeros(n)

# --- RISK FACTORS (increase probability) ---

# Overtime: strongest predictor (+22%)
prob += np.where(overtime == "Yes", 0.22, 0.0)

# Low job satisfaction
prob += np.where(job_satisfaction == 1, 0.14, 0.0)
prob += np.where(job_satisfaction == 2, 0.07, 0.0)

# Poor work-life balance
prob += np.where(work_life_balance == 1, 0.12, 0.0)
prob += np.where(work_life_balance == 2, 0.05, 0.0)

# Low income
prob += np.where(monthly_income < 3000, 0.13, 0.0)
prob += np.where((monthly_income >= 3000) & (monthly_income < 5000), 0.07, 0.0)

# Job hopper pattern
prob += np.where(num_companies_worked >= 7, 0.12, 0.0)
prob += np.where((num_companies_worked >= 4) & (num_companies_worked < 7), 0.06, 0.0)

# No recent promotion
prob += np.where(years_since_last_promo >= 5, 0.10, 0.0)
prob += np.where((years_since_last_promo >= 3) & (years_since_last_promo < 5), 0.05, 0.0)

# Long commute
prob += np.where(distance_from_home >= 20, 0.08, 0.0)

# Young + early career employees leave more
prob += np.where((age < 30) & (total_working_years <= 5), 0.10, 0.0)

# Brand new to company
prob += np.where(years_at_company <= 1, 0.09, 0.0)

# Single employees more mobile
prob += np.where(marital_status == "Single", 0.08, 0.0)

# Poor environment
prob += np.where(environment_satisfaction == 1, 0.07, 0.0)

# No stock options
prob += np.where(stock_option_level == 0, 0.06, 0.0)

# Low job involvement
prob += np.where(job_involvement == 1, 0.07, 0.0)

# Sales roles have higher turnover
prob += np.where(job_role == "Sales Representative", 0.08, 0.0)

# Low salary hike despite working
prob += np.where(percent_salary_hike <= 12, 0.05, 0.0)

# Frequent travel increases attrition
prob += np.where(business_travel == "Travel_Frequently", 0.07, 0.0)

# Low relationship satisfaction
prob += np.where(relationship_satisfaction == 1, 0.05, 0.0)

# Base attrition rate
prob += 0.05

# --- PROTECTIVE FACTORS (reduce probability) ---
prob -= np.where(stock_option_level >= 2, 0.06, 0.0)
prob -= np.where(job_level >= 4, 0.07, 0.0)
prob -= np.where(total_working_years >= 15, 0.05, 0.0)
prob -= np.where(years_at_company >= 10, 0.05, 0.0)
prob -= np.where(monthly_income >= 10000, 0.06, 0.0)
prob -= np.where(work_life_balance == 4, 0.04, 0.0)
prob -= np.where(job_satisfaction == 4, 0.05, 0.0)
prob -= np.where(marital_status == "Married", 0.04, 0.0)
prob -= np.where(years_since_last_promo == 0, 0.03, 0.0)

# Keep probability in valid range
prob = np.clip(prob, 0.02, 0.95)

# ── STEP 3: Generate Attrition column ────────────────────
attrition = np.where(np.random.random(n) < prob, "Yes", "No")

# ── STEP 4: Build final DataFrame ────────────────────────
df = pd.DataFrame({
    "Age":                      age,
    "Attrition":                attrition,
    "BusinessTravel":           business_travel,
    "DailyRate":                daily_rate,
    "Department":               department,
    "DistanceFromHome":         distance_from_home,
    "Education":                education,
    "EducationField":           education_field,
    "EmployeeCount":            employee_count,
    "EmployeeNumber":           employee_number,
    "EnvironmentSatisfaction":  environment_satisfaction,
    "Gender":                   gender,
    "HourlyRate":               hourly_rate,
    "JobInvolvement":           job_involvement,
    "JobLevel":                 job_level,
    "JobRole":                  job_role,
    "JobSatisfaction":          job_satisfaction,
    "MaritalStatus":            marital_status,
    "MonthlyIncome":            monthly_income,
    "MonthlyRate":              monthly_rate,
    "NumCompaniesWorked":       num_companies_worked,
    "Over18":                   over18,
    "OverTime":                 overtime,
    "PercentSalaryHike":        percent_salary_hike,
    "PerformanceRating":        performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StandardHours":            standard_hours,
    "StockOptionLevel":         stock_option_level,
    "TotalWorkingYears":        total_working_years,
    "TrainingTimesLastYear":    training_times_last_year,
    "WorkLifeBalance":          work_life_balance,
    "YearsAtCompany":           years_at_company,
    "YearsInCurrentRole":       years_in_current_role,
    "YearsSinceLastPromotion":  years_since_last_promo,
    "YearsWithCurrManager":     years_with_curr_manager,
})

# ── STEP 5: Save ──────────────────────────────────────────
df.to_csv("employee_data_10000.csv", index=False)

# ── STEP 6: Verify signal is strong ──────────────────────
print(f"\nDataset saved: employee_data_10000.csv")
print(f"Shape: {df.shape}")
print(f"\nAttrition distribution:")
print(df["Attrition"].value_counts())
print(f"Attrition rate: {(df['Attrition']=='Yes').mean()*100:.1f}%")

print("\n=== SIGNAL VERIFICATION ===")
ot_yes = df[df["OverTime"]=="Yes"]["Attrition"].value_counts(normalize=True).get("Yes",0)
ot_no  = df[df["OverTime"]=="No"]["Attrition"].value_counts(normalize=True).get("Yes",0)
print(f"OverTime=Yes → Attrition: {ot_yes:.1%}   OverTime=No → {ot_no:.1%}   Diff: {abs(ot_yes-ot_no):.1%}")

low_s  = df[df["JobSatisfaction"]==1]["Attrition"].value_counts(normalize=True).get("Yes",0)
high_s = df[df["JobSatisfaction"]==4]["Attrition"].value_counts(normalize=True).get("Yes",0)
print(f"JobSat=1 Low → {low_s:.1%}   JobSat=4 High → {high_s:.1%}   Diff: {abs(low_s-high_s):.1%}")

low_i  = df[df["MonthlyIncome"]<3000]["Attrition"].value_counts(normalize=True).get("Yes",0)
high_i = df[df["MonthlyIncome"]>10000]["Attrition"].value_counts(normalize=True).get("Yes",0)
print(f"Income<3000 → {low_i:.1%}   Income>10000 → {high_i:.1%}   Diff: {abs(low_i-high_i):.1%}")

print("\n✅ Dataset generated successfully with strong attrition signal!")
print("Next step: Run train_model.py to retrain models on this dataset.")