import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # ← fixes TclError: no display/Tk needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
df = pd.read_csv("employee_data_10000.csv")

# Drop useless constant columns
df = df.drop(
    columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"],
    errors="ignore"
)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#    These 6 new features boost model accuracy
#    by capturing hidden patterns in the data
# ─────────────────────────────────────────────

# Combined satisfaction score (job + environment + work-life)
df["SatisfactionScore"] = (
    df["JobSatisfaction"] +
    df["EnvironmentSatisfaction"] +
    df["WorkLifeBalance"]
) / 3

# Whether income grew with experience
df["IncomePerYear"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

# How loyal the employee is relative to total career
df["YearsPerCompany"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

# Experience gained at other companies before joining
df["ExperienceGap"] = df["TotalWorkingYears"] - df["YearsAtCompany"]

# Loyalty vs manager relationship
df["LoyaltyScore"] = df["YearsAtCompany"] - df["YearsWithCurrManager"]

# Overtime numeric flag (in case it's still string)
if df["OverTime"].dtype == object:
    df["OverTimeFlag"] = df["OverTime"].map({"Yes": 1, "No": 0})
else:
    df["OverTimeFlag"] = df["OverTime"]

# ─────────────────────────────────────────────
# 3. ENCODE TARGET & CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print(f"\nClass distribution:\n{df['Attrition'].value_counts()}")
print(f"Attrition rate: {df['Attrition'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 4. FEATURES & TARGET
# ─────────────────────────────────────────────
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
#    stratify=y ensures same attrition ratio
#    in both train and test sets
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # KEY FIX: keeps class ratio balanced
)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ─────────────────────────────────────────────
# 6. SCALING (only for Logistic Regression)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 7. LOGISTIC REGRESSION
#    class_weight="balanced" fixes class imbalance
#    C=0.5 adds slight regularisation to reduce overfitting
# ─────────────────────────────────────────────
logistic_model = LogisticRegression(
    max_iter=1000,
    C=0.5,
    class_weight="balanced",   # KEY FIX: handles Yes/No imbalance
    solver="lbfgs",
    random_state=42
)
logistic_model.fit(X_train_scaled, y_train)

y_pred_lr  = logistic_model.predict(X_test_scaled)
lr_acc     = accuracy_score(y_test, y_pred_lr)
lr_auc     = roc_auc_score(y_test, logistic_model.predict_proba(X_test_scaled)[:, 1])

print(f"\n── Logistic Regression ──────────────────")
print(f"Accuracy : {lr_acc:.4f}  ({lr_acc*100:.2f}%)")
print(f"AUC Score: {lr_auc:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["Stay", "Leave"]))

# ─────────────────────────────────────────────
# 8. RANDOM FOREST
#    class_weight="balanced" is the single biggest
#    improvement for imbalanced attrition data
#    Tuned hyperparameters prevent overfitting
# ─────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=300,          # more trees = more stable predictions
    max_depth=15,              # prevents overfitting
    min_samples_split=10,      # prevents overfitting
    min_samples_leaf=4,        # prevents overfitting
    max_features="sqrt",       # standard best practice for RF
    class_weight="balanced",   # KEY FIX: handles Yes/No imbalance
    random_state=42,
    n_jobs=-1                  # uses all CPU cores, faster training
)
rf_model.fit(X_train, y_train)

y_pred_rf  = rf_model.predict(X_test)
rf_acc     = accuracy_score(y_test, y_pred_rf)
rf_auc     = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print(f"\n── Random Forest ────────────────────────")
print(f"Accuracy : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"AUC Score: {rf_auc:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=["Stay", "Leave"]))

# ─────────────────────────────────────────────
# 9. CROSS VALIDATION (5-fold on both models)
# ─────────────────────────────────────────────
lr_cv = cross_val_score(logistic_model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="roc_auc")

print(f"\n── 5-Fold Cross Validation ──────────────")
print(f"LR  Mean AUC: {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")
print(f"RF  Mean AUC: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# ─────────────────────────────────────────────
# 10. FEATURE IMPORTANCE CHART (Top 15)
# ─────────────────────────────────────────────
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
top15    = feat_imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top15.values, y=top15.index, palette="viridis")
plt.title("Top 15 Features - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("\nFeature importance chart saved.")

# ─────────────────────────────────────────────
# 11. SAVE MODELS & ARTIFACTS
#     Exact same filenames your app.py expects
# ─────────────────────────────────────────────
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(rf_model,       "random_forest_model.pkl")
joblib.dump(scaler,         "scaler.pkl")
joblib.dump(X.columns,      "columns.pkl")

print("\n✅ All models trained and saved successfully!")
print(f"\n📊 Final Results:")
print(f"   Logistic Regression → Accuracy: {lr_acc*100:.2f}%  |  AUC: {lr_auc:.4f}")
print(f"   Random Forest       → Accuracy: {rf_acc*100:.2f}%  |  AUC: {rf_auc:.4f}")