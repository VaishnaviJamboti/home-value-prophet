"""
=============================================================
  LABMENTIX INTERNSHIP — Real Estate Investment Advisor
  Step 3: Machine Learning Models
  ✓ Works with india_housing_cleaned.csv
  ✓ No MLflow, No XGBoost (scikit-learn only)
=============================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, classification_report
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
OUTPUT_DIR = "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*70)
print("STEP 3 — ML MODELS (Classification + Regression)")
print("="*70)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv("india_housing_cleaned.csv")
print(f"✓ Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────────
print("[2] Preparing features...")

# Select NUMERIC features only
numeric_features = [
    "Size_in_SqFt",
    "BHK",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Parking_Space",
    "Transport_Score",
    "Infrastructure_Score",
]

# Keep only columns that exist in the CSV
numeric_features = [col for col in numeric_features if col in df.columns]

# Select CATEGORICAL features
categorical_features = ["Property_Type", "Furnished_Status", "Owner_Type"]
categorical_features = [col for col in categorical_features if col in df.columns]

# Create X (features)
X = df[numeric_features].copy()

# One-hot encode categorical features
if len(categorical_features) > 0:
    X = pd.concat([X, pd.get_dummies(df[categorical_features], drop_first=True)], axis=1)

# Fill NaN values
X = X.fillna(X.median(numeric_only=True))

print(f"✓ Features prepared: {X.shape[1]} variables\n")

# ─────────────────────────────────────────────
# 3A. CLASSIFICATION — Good Investment?
# ─────────────────────────────────────────────
print("="*70)
print("TASK 1: CLASSIFICATION (Good Investment = Yes/No)")
print("="*70)

y_class = df["Good_Investment"].values
print(f"\nClass distribution:")
for label in [0, 1]:
    count = (y_class == label).sum()
    pct = (count / len(y_class)) * 100
    print(f"  • Class {label}: {count:,} ({pct:.1f}%)")

# Split data
print("\n[3A-1] Train/Test Split (80/20)...")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
print(f"✓ Train: {len(X_train_c):,} | Test: {len(X_test_c):,}")

# Scale features
print("\n[3A-2] Scaling features...")
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)
print("✓ StandardScaler applied")

# Model 1: Logistic Regression
print("\n[3A-3] Training: Logistic Regression")
lr_clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_clf.fit(X_train_c_scaled, y_train_c)
y_pred_lr = lr_clf.predict(X_test_c_scaled)
y_pred_proba_lr = lr_clf.predict_proba(X_test_c_scaled)[:, 1]

acc_lr = accuracy_score(y_test_c, y_pred_lr)
prec_lr = precision_score(y_test_c, y_pred_lr, zero_division=0)
rec_lr = recall_score(y_test_c, y_pred_lr, zero_division=0)
f1_lr = f1_score(y_test_c, y_pred_lr, zero_division=0)
auc_lr = roc_auc_score(y_test_c, y_pred_proba_lr)

print(f"  - Accuracy:  {acc_lr:.4f}")
print(f"  - Precision: {prec_lr:.4f}")
print(f"  - Recall:    {rec_lr:.4f}")
print(f"  - F1-Score:  {f1_lr:.4f}")
print(f"  - ROC-AUC:   {auc_lr:.4f}")

# Model 2: Random Forest Classifier
print("\n[3A-4] Training: Random Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict(X_test_c)
y_pred_proba_rf = rf_clf.predict_proba(X_test_c)[:, 1]

acc_rf = accuracy_score(y_test_c, y_pred_rf)
prec_rf = precision_score(y_test_c, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test_c, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test_c, y_pred_rf, zero_division=0)
auc_rf = roc_auc_score(y_test_c, y_pred_proba_rf)

print(f"  - Accuracy:  {acc_rf:.4f}")
print(f"  - Precision: {prec_rf:.4f}")
print(f"  - Recall:    {rec_rf:.4f}")
print(f"  - F1-Score:  {f1_rf:.4f}")
print(f"  - ROC-AUC:   {auc_rf:.4f}")

# Select BEST classifier
if acc_rf >= acc_lr:
    best_clf = rf_clf
    best_clf_name = "Random Forest"
    y_pred_best_c = y_pred_rf
else:
    best_clf = lr_clf
    best_clf_name = "Logistic Regression"
    y_pred_best_c = y_pred_lr

print(f"\n✓ BEST: {best_clf_name} (Accuracy: {max(acc_rf, acc_lr):.4f})")

# ─────────────────────────────────────────────
# 3B. REGRESSION — Price After 5 Years
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("TASK 2: REGRESSION (Predict Price After 5 Years)")
print("="*70)

y_regr = df["Price_After_5Yrs"].values
print(f"\nPrice After 5 Years:")
print(f"  - Min:    Rs {y_regr.min():.2f} Lakhs")
print(f"  - Max:    Rs {y_regr.max():.2f} Lakhs")
print(f"  - Mean:   Rs {y_regr.mean():.2f} Lakhs")
print(f"  - Median: Rs {np.median(y_regr):.2f} Lakhs")

# Split data
print("\n[3B-1] Train/Test Split (80/20)...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_regr, test_size=0.2, random_state=42
)
print(f"✓ Train: {len(X_train_r):,} | Test: {len(X_test_r):,}")

# Scale features
print("\n[3B-2] Scaling features...")
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)
print("✓ StandardScaler applied")

# Model 1: Linear Regression
print("\n[3B-3] Training: Linear Regression")
lr_regr = LinearRegression()
lr_regr.fit(X_train_r_scaled, y_train_r)
y_pred_lr_regr = lr_regr.predict(X_test_r_scaled)

mae_lr = mean_absolute_error(y_test_r, y_pred_lr_regr)
rmse_lr = np.sqrt(mean_squared_error(y_test_r, y_pred_lr_regr))
r2_lr = r2_score(y_test_r, y_pred_lr_regr)

print(f"  - MAE:  Rs {mae_lr:.2f} Lakhs")
print(f"  - RMSE: Rs {rmse_lr:.2f} Lakhs")
print(f"  - R2:   {r2_lr:.4f}")

# Model 2: Random Forest Regressor
print("\n[3B-4] Training: Random Forest Regressor")
rf_regr = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_regr.fit(X_train_r, y_train_r)
y_pred_rf_regr = rf_regr.predict(X_test_r)

mae_rf = mean_absolute_error(y_test_r, y_pred_rf_regr)
rmse_rf = np.sqrt(mean_squared_error(y_test_r, y_pred_rf_regr))
r2_rf = r2_score(y_test_r, y_pred_rf_regr)

print(f"  - MAE:  Rs {mae_rf:.2f} Lakhs")
print(f"  - RMSE: Rs {rmse_rf:.2f} Lakhs")
print(f"  - R2:   {r2_rf:.4f}")

# Select BEST regressor
if r2_rf >= r2_lr:
    best_regr = rf_regr
    best_regr_name = "Random Forest"
    y_pred_best_r = y_pred_rf_regr
else:
    best_regr = lr_regr
    best_regr_name = "Linear Regression"
    y_pred_best_r = y_pred_lr_regr

print(f"\n✓ BEST: {best_regr_name} (R2: {max(r2_rf, r2_lr):.4f})")

# ─────────────────────────────────────────────
# 4. SAVE MODELS
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

print("\n[4-1] Saving classifier...")
with open(f"{OUTPUT_DIR}/best_classifier.pkl", "wb") as f:
    pickle.dump(best_clf, f)
print(f"✓ {OUTPUT_DIR}/best_classifier.pkl")

print("\n[4-2] Saving regressor...")
with open(f"{OUTPUT_DIR}/best_regressor.pkl", "wb") as f:
    pickle.dump(best_regr, f)
print(f"✓ {OUTPUT_DIR}/best_regressor.pkl")

print("\n[4-3] Saving scalers...")
with open(f"{OUTPUT_DIR}/scaler_classification.pkl", "wb") as f:
    pickle.dump(scaler_c, f)
with open(f"{OUTPUT_DIR}/scaler_regression.pkl", "wb") as f:
    pickle.dump(scaler_r, f)
print(f"✓ {OUTPUT_DIR}/scaler_classification.pkl")
print(f"✓ {OUTPUT_DIR}/scaler_regression.pkl")

# ─────────────────────────────────────────────
# 5. CREATE VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ML Model Performance Comparison", fontsize=16, fontweight="bold")

# [1] Classification Accuracy
ax = axes[0, 0]
models = ["Logistic\nRegression", "Random\nForest"]
scores = [acc_lr, acc_rf]
colors = ["#FF6B6B", "#4ECDC4"]
bars = ax.bar(models, scores, color=colors, edgecolor="black", linewidth=1.5)
ax.set_ylabel("Accuracy", fontweight="bold", fontsize=11)
ax.set_title("Classification: Accuracy", fontweight="bold", fontsize=12)
ax.set_ylim([0, 1.0])
ax.grid(axis="y", alpha=0.3)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{score:.4f}", ha="center", fontsize=10, fontweight="bold")

# [2] Classification ROC-AUC
ax = axes[0, 1]
scores = [auc_lr, auc_rf]
bars = ax.bar(models, scores, color=colors, edgecolor="black", linewidth=1.5)
ax.set_ylabel("ROC-AUC", fontweight="bold", fontsize=11)
ax.set_title("Classification: ROC-AUC Score", fontweight="bold", fontsize=12)
ax.set_ylim([0, 1.0])
ax.grid(axis="y", alpha=0.3)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{score:.4f}", ha="center", fontsize=10, fontweight="bold")

# [3] Regression R2
ax = axes[1, 0]
models_r = ["Linear\nRegression", "Random\nForest"]
scores = [r2_lr, r2_rf]
colors_r = ["#95E1D3", "#F38181"]
bars = ax.bar(models_r, scores, color=colors_r, edgecolor="black", linewidth=1.5)
ax.set_ylabel("R2 Score", fontweight="bold", fontsize=11)
ax.set_title("Regression: R2 Score (Higher is Better)", fontweight="bold", fontsize=12)
ax.set_ylim([0, 1.0])
ax.grid(axis="y", alpha=0.3)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{score:.4f}", ha="center", fontsize=10, fontweight="bold")

# [4] Regression MAE
ax = axes[1, 1]
scores = [mae_lr, mae_rf]
bars = ax.bar(models_r, scores, color=colors_r, edgecolor="black", linewidth=1.5)
ax.set_ylabel("MAE (Rs Lakhs)", fontweight="bold", fontsize=11)
ax.set_title("Regression: Mean Absolute Error (Lower is Better)", fontweight="bold", fontsize=12)
ax.grid(axis="y", alpha=0.3)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores)*0.02,
            f"Rs {score:.2f}L", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=300, bbox_inches="tight")
print(f"\n✓ {OUTPUT_DIR}/model_comparison.png")
plt.close()

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_clf.feature_importances_
}).sort_values("Importance", ascending=True).tail(15)

importance_df.plot(kind="barh", ax=ax, legend=False, color="#4ECDC4", edgecolor="black")
ax.set_xlabel("Importance Score", fontweight="bold")
ax.set_title("Top 15 Features - Classification Model", fontweight="bold", fontsize=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300, bbox_inches="tight")
print(f"✓ {OUTPUT_DIR}/feature_importance.png")
plt.close()

# ─────────────────────────────────────────────
# 6. SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

summary_text = f"""
LABMENTIX INTERNSHIP - STEP 3: ML MODELS
==========================================

CLASSIFICATION (Good Investment = Yes/No)
  Best Model    : {best_clf_name}
  Accuracy      : {max(acc_rf, acc_lr):.4f}
  Precision     : {max(prec_rf, prec_lr):.4f}
  Recall        : {max(rec_rf, rec_lr):.4f}
  F1-Score      : {max(f1_rf, f1_lr):.4f}
  ROC-AUC       : {max(auc_rf, auc_lr):.4f}

REGRESSION (Price After 5 Years)
  Best Model    : {best_regr_name}
  R2 Score      : {max(r2_rf, r2_lr):.4f}
  MAE           : Rs {min(mae_rf, mae_lr):.2f} Lakhs
  RMSE          : Rs {min(rmse_rf, rmse_lr):.2f} Lakhs

SAVED FILES
  - ml_outputs/best_classifier.pkl
  - ml_outputs/best_regressor.pkl
  - ml_outputs/scaler_classification.pkl
  - ml_outputs/scaler_regression.pkl
  - ml_outputs/model_comparison.png
  - ml_outputs/feature_importance.png
  - ml_outputs/model_summary.txt

READY FOR DEPLOYMENT!
"""

print(summary_text)

with open(f"{OUTPUT_DIR}/model_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
print(f"\n✓ {OUTPUT_DIR}/model_summary.txt")

print("\n" + "="*70)
print("SUCCESS! Step 3 complete.")
print("="*70 + "\n")