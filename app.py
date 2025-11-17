# ============================================================
# All-in-One Analytics Script for Luxury 1–2h Delivery
# - Data generation (or CSV load)
# - Classification (many models) + metrics + ROC + confusion matrices
# - Clustering (KMeans) + silhouette + human-readable profiles
# - Association Rule Mining (Apriori) with Top-N rules (support/conf/lift)
# - Regression (Linear, Ridge, Lasso) + MAE, RMSE, R2 + best model plot
# ============================================================
# Requirements: numpy, pandas, scikit-learn, matplotlib
# Optional for ARM: mlxtend
# ------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO

# ---- SKLEARN ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    silhouette_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ---- Optional (ARM) ----
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

# ---------------- CONFIG ----------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_SYNTHETIC = True         # Set False to load your CSV
DATA_PATH = "your_data.csv"  # Path if using real CSV

RANDOM_STATE = 42
N_SAMPLES = 1200             # synthetic rows if needed

# ARM thresholds
MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.50
TOP_N_RULES = 10

# KMeans settings
KMEANS_K = 4
# ---------------------------------------


# =============== Utilities ===============
def save_fig(fig, basename, dpi=180):
    png = os.path.join(OUTPUT_DIR, f"{basename}.png")
    jpg = os.path.join(OUTPUT_DIR, f"{basename}.jpg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(jpg, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def describe_clusters_means(df, cluster_col, cols):
    """
    Return a 'human-readable' profile of clusters:
    mean values and relative labels (Low/Med/High vs global mean)
    """
    prof = df.groupby(cluster_col)[cols].mean().round(2)
    overall = df[cols].mean()
    labels = pd.DataFrame(index=prof.index)
    for c in cols:
        labels[c] = np.where(prof[c] >= overall[c] * 1.15, "High",
                      np.where(prof[c] <= overall[c] * 0.85, "Low", "Medium"))
    return prof, labels


# =============== Data ===============
def generate_synthetic(n=N_SAMPLES, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(index=range(n))

    df["Customer_ID"] = [f"CUST_{i+1:05d}" for i in range(n)]
    df["Age"] = rng.integers(21, 70, size=n)
    df["City_Tier"] = rng.choice([1,2,3], size=n, p=[0.5,0.35,0.15])

    base_income = rng.lognormal(mean=10.5, sigma=0.45, size=n) / 1000.0
    df["Annual_Income"] = np.clip(base_income * df["City_Tier"].map({1:1.25,2:1.0,3:0.8}), 20000, 350000)

    spend_ratio = np.clip(rng.normal(0.065, 0.025, size=n), 0.015, 0.18)
    df["Annual_Luxury_Spend"] = df["Annual_Income"] * spend_ratio
    df["Monthly_Luxury_Spend"] = df["Annual_Luxury_Spend"] / 12.0
    df["Luxury_Purchases_Per_Year"] = np.clip(
        np.round(df["Annual_Luxury_Spend"] / 3500.0 + rng.normal(0, 1.2, size=n)), 0, 40
    )

    df["Brand_Loyalty_Score"] = np.clip(
        2.5
        + 3.0 * (df["Luxury_Purchases_Per_Year"] / max(1.0, df["Luxury_Purchases_Per_Year"].max()))
        + 2.2 * (df["Annual_Luxury_Spend"] / max(1e-9, df["Annual_Luxury_Spend"].max()))
        + rng.normal(0, 1, size=n),
        1, 10
    ).round(1)

    df["Digital_Shopping_Comfort"] = rng.choice([1,2,3,4,5], size=n, p=[0.06,0.1,0.22,0.35,0.27])

    interest_base = (
        1.1
        + 0.8 * (df["Digital_Shopping_Comfort"] / 5)
        + 0.45 * (df["Brand_Loyalty_Score"] / 10)
        + 0.45 * (df["Annual_Luxury_Spend"] / max(1e-9, df["Annual_Luxury_Spend"].max()))
        + rng.normal(0, 0.5, size=n)
    )
    df["Interest_1_2hr_Service"] = np.clip(np.round(interest_base * 5 / 3), 1, 5).astype(int)

    df["WTP_1hr_Delivery"] = np.clip(
        10 + 0.00045 * df["Annual_Income"] + 0.7 * df["Interest_1_2hr_Service"] + 0.25 * df["Digital_Shopping_Comfort"]
        + rng.normal(0, 12, size=n),
        0, None
    ).round(2)

    # Multi-select
    cats = [
        "Interested_Handbags","Interested_Watches","Interested_Fine_Jewelry",
        "Interested_Limited_Sneakers","Interested_Beauty","Interested_Fragrances",
        "Interested_Tech_Gadgets"
    ]
    for c in cats: df[c] = 0

    inc_max = max(1e-9, df["Annual_Income"].max())
    for i in range(n):
        base_p = 0.28 + 0.2 * (df.loc[i, "Annual_Income"] / inc_max)
        prefs = {
            "Interested_Handbags": base_p + (0.12 if 25 <= df.loc[i,"Age"] <= 45 else -0.05),
            "Interested_Watches": base_p + (0.06 if df.loc[i,"Age"] >= 30 else 0.0),
            "Interested_Fine_Jewelry": base_p + (0.12 if df.loc[i,"Annual_Income"] > 100000 else -0.05),
            "Interested_Limited_Sneakers": base_p + (0.16 if df.loc[i,"Age"] <= 35 else -0.04),
            "Interested_Beauty": base_p + 0.06,
            "Interested_Fragrances": base_p + 0.10,
            "Interested_Tech_Gadgets": base_p + (0.16 if df.loc[i,"Annual_Income"] > 70000 else 0.0),
        }
        for k,v in prefs.items(): df.loc[i,k] = int(np.random.random() < np.clip(v, 0.05, 0.92))

    feats = ["Feat_PersonalizedPackaging","Feat_DedicatedConcierge","Feat_SameDayTailoringSizing","Feat_HighValueInsurance"]
    for f in feats: df[f] = 0

    med_m = df["Monthly_Luxury_Spend"].median()
    for i in range(n):
        p = {"Feat_PersonalizedPackaging":0.52,"Feat_DedicatedConcierge":0.36,"Feat_SameDayTailoringSizing":0.26,"Feat_HighValueInsurance":0.41}
        if df.loc[i,"Interested_Handbags"]: p["Feat_PersonalizedPackaging"] += 0.16; p["Feat_HighValueInsurance"] += 0.10
        if df.loc[i,"Interested_Fine_Jewelry"]: p["Feat_HighValueInsurance"] += 0.22
        if df.loc[i,"Interested_Limited_Sneakers"]: p["Feat_SameDayTailoringSizing"] += 0.16
        if df.loc[i,"Monthly_Luxury_Spend"] > med_m: p["Feat_DedicatedConcierge"] += 0.06
        for k,v in p.items(): df.loc[i,k] = int(np.random.random() < np.clip(v, 0.05, 0.97))

    df["n_categories_interested"] = df[[c for c in df.columns if c.startswith("Interested_")]].sum(axis=1)
    df["n_features_valued"] = df[[c for c in df.columns if c.startswith("Feat_")]].sum(axis=1)

    # Likely_to_Use target for classification
    df["Likely_to_Use"] = (df["Interest_1_2hr_Service"] >= 4).astype(int)

    # Base_Price (if absent in real data we’ll create later too)
    df["Base_Price"] = np.round(
        rng.choice([350, 600, 900, 1500, 2500, 4000, 6500, 9500], size=n,
                   p=[0.10,0.12,0.15,0.20,0.18,0.12,0.08,0.05]) *
        (1 + rng.normal(0, 0.05, size=n)), 2
    )

    return df


# Load data
if USE_SYNTHETIC:
    df = generate_synthetic()
else:
    df = pd.read_csv(DATA_PATH)

# Ensure helper columns if missing
if "n_categories_interested" not in df.columns:
    df["n_categories_interested"] = df[[c for c in df.columns if c.startswith("Interested_")]].sum(axis=1)
if "n_features_valued" not in df.columns:
    df["n_features_valued"] = df[[c for c in df.columns if c.startswith("Feat_")]].sum(axis=1)
if "Likely_to_Use" not in df.columns:
    if "Interest_1_2hr_Service" in df.columns:
        df["Likely_to_Use"] = (df["Interest_1_2hr_Service"] >= 4).astype(int)
    else:
        med = df["WTP_1hr_Delivery"].median() if "WTP_1hr_Delivery" in df.columns else 20
        df["Likely_to_Use"] = (df["WTP_1hr_Delivery"] > med).astype(int) if "WTP_1hr_Delivery" in df.columns else 0
if "Base_Price" not in df.columns:
    rng = np.random.default_rng(7)
    df["Base_Price"] = np.round(
        rng.choice([350,600,900,1500,2500,4000,6500,9500], size=len(df),
                   p=[0.10,0.12,0.15,0.20,0.18,0.12,0.08,0.05]) * (1 + rng.normal(0,0.05,size=len(df))), 2
    )

# Save a peek at data
df.head(20).to_csv(os.path.join(OUTPUT_DIR, "sample_data_head.csv"), index=False)

# ============================================================
# A) CLASSIFICATION — multiple models, metrics & plots
# ============================================================
print("\n=== Classification ===")
target = "Likely_to_Use"

features_num = [
    "Age","Annual_Income","Monthly_Luxury_Spend","Luxury_Purchases_Per_Year",
    "Brand_Loyalty_Score","Digital_Shopping_Comfort","Interest_1_2hr_Service",
    "WTP_1hr_Delivery","City_Tier","n_categories_interested","n_features_valued"
]
features_cat = [c for c in df.columns if c.startswith("Interested_")] + [c for c in df.columns if c.startswith("Feat_")]

X_cols = [c for c in (features_num + features_cat) if c in df.columns and c != target]
data_clf = df.dropna(subset=[target]).copy()
for c in features_cat:
    if c in data_clf.columns:
        data_clf[c] = data_clf[c].astype(int)

X = data_clf[X_cols].copy()
y = data_clf[target].astype(int)

# Guard against single-class target
if y.nunique() < 2:
    raise SystemExit("Classification target has one class only; cannot evaluate classifiers.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=RANDOM_STATE
)

# Scale numerics for models that need it
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train[num_cols])
X_test_s  = scaler.transform(X_test[num_cols])

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=800, solver="lbfgs", random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVC-RBF": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
}

results = []
roc_fig = plt.figure(figsize=(6,5))
for name, model in models.items():
    # Prepare X input (scaled numerics for linear/logistic/SVC/KNN; tree models can use raw mix)
    if name in ["LogisticRegression","SVC-RBF","KNN","GradientBoosting","DecisionTree","RandomForest"]:
        # For simplicity, feed scaled numerics + raw binaries concatenated (won't hurt trees)
        Xtr = np.hstack([X_train_s, X_train.drop(columns=num_cols).values]) if len(num_cols) > 0 else X_train.values
        Xte = np.hstack([X_test_s,  X_test.drop(columns=num_cols).values])  if len(num_cols) > 0 else X_test.values
    else:
        Xtr = X_train.values
        Xte = X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xte)[:,1]
    else:
        # Decision function fallback
        if hasattr(model, "decision_function"):
            raw = model.decision_function(Xte)
            # Min-max to [0,1]
            y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            y_proba = np.zeros_like(y_pred, dtype=float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = np.nan

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc})

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    except Exception:
        pass

# Finalize ROC plot
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Classification)")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
save_fig(plt.gcf(), "classification_roc_curves")

# Confusion matrices
for name, model in models.items():
    if name in ["LogisticRegression","SVC-RBF","KNN","GradientBoosting","DecisionTree","RandomForest"]:
        Xte = np.hstack([X_test_s, X_test.drop(columns=num_cols).values]) if len(num_cols) > 0 else X_test.values
    else:
        Xte = X_test.values
    y_pred = model.predict(Xte)
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    save_fig(fig, f"cm_{name}")

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"), index=False)
print(results_df.head())

# ============================================================
# B) CLUSTERING — KMeans + silhouette + profiles
# ============================================================
print("\n=== Clustering (KMeans) ===")
clust_cols = [
    "Age","Annual_Income","Monthly_Luxury_Spend","Luxury_Purchases_Per_Year",
    "Brand_Loyalty_Score","Digital_Shopping_Comfort","Interest_1_2hr_Service",
    "WTP_1hr_Delivery","City_Tier","n_categories_interested","n_features_valued"
]
clust_cols = [c for c in clust_cols if c in df.columns]
if len(clust_cols) < 3:
    print("Not enough numeric columns for clustering.")
else:
    Xc = df[clust_cols].copy()
    scaler_c = StandardScaler()
    Xc_s = scaler_c.fit_transform(Xc)

    kmeans = KMeans(n_clusters=KMEANS_K, n_init=20, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(Xc_s)
    df["Cluster"] = labels

    # Silhouette
    sil = silhouette_score(Xc_s, labels) if len(set(labels)) > 1 else np.nan
    print(f"Silhouette score (K={KMEANS_K}): {sil:.3f}")

    # Profiles (means + qualitative labels)
    prof_means, prof_labels = describe_clusters_means(df, "Cluster", clust_cols)
    prof_means.to_csv(os.path.join(OUTPUT_DIR, "kmeans_profile_means.csv"))
    prof_labels.to_csv(os.path.join(OUTPUT_DIR, "kmeans_profile_labels.csv"))

    # Simple 2D viz (first 2 principal-like axes via scaler only; optional: PCA)
    fig = plt.figure(figsize=(5,4))
    plt.scatter(Xc_s[:,0], Xc_s[:,1], c=labels, s=8, alpha=0.6, cmap="tab10")
    plt.title(f"KMeans Clusters (K={KMEANS_K}) — Silhouette={sil:.3f}")
    plt.xlabel(clust_cols[0]); plt.ylabel(clust_cols[1])
    plt.grid(True, alpha=0.3)
    save_fig(fig, "kmeans_scatter")

# ============================================================
# C) ASSOCIATION RULE MINING — Apriori (optional)
# ============================================================
print("\n=== Association Rule Mining ===")
if not MLXTEND_AVAILABLE:
    print("mlxtend not installed; skipping ARM. Install: pip install mlxtend")
else:
    basket_cols = [c for c in df.columns if c.startswith("Interested_") or c.startswith("Feat_")]
    if len(basket_cols) < 2:
        print("Not enough basket columns for ARM.")
    else:
        basket = df[basket_cols].copy().fillna(0).astype(bool)
        freq = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
        if freq.empty:
            print("No frequent itemsets at current support.")
        else:
            rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
            if rules.empty:
                print("No rules at current confidence.")
            else:
                rules["antecedent_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
                rules["consequent_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
                rules = rules.sort_values(["lift","confidence","support"], ascending=False)
                top = rules[["antecedent_str","consequent_str","support","confidence","lift"]].head(TOP_N_RULES)
                top.to_csv(os.path.join(OUTPUT_DIR, "arm_top_rules.csv"), index=False)
                print(top)

# ============================================================
# D) REGRESSION — Linear, Ridge, Lasso
# ============================================================
print("\n=== Regression (Linear, Ridge, Lasso) ===")
target_r = "Monthly_Luxury_Spend"
reg_num = [
    "Age","Annual_Income","Luxury_Purchases_Per_Year","Brand_Loyalty_Score",
    "Digital_Shopping_Comfort","Interest_1_2hr_Service","WTP_1hr_Delivery",
    "City_Tier","n_categories_interested","n_features_valued"
]
reg_cat = [c for c in df.columns if c.startswith("Interested_")] + [c for c in df.columns if c.startswith("Feat_")]
reg_cols = [c for c in (reg_num + reg_cat) if c in df.columns]

data_reg = df.dropna(subset=[target_r]).copy()
for c in reg_cat:
    if c in data_reg.columns:
        data_reg[c] = data_reg[c].astype(int)

Xr = data_reg[reg_cols].copy()
yr = data_reg[target_r].copy()

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=RANDOM_STATE)

# Scale numerics
numr = Xr_train.select_dtypes(include=[np.number]).columns.tolist()
scaler_r = StandardScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train[numr])
Xr_test_s  = scaler_r.transform(Xr_test[numr])

# For simplicity, run models on scaled **numeric-only** subset
models_r = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
}

res_reg = []
preds = {}
for name, mdl in models_r.items():
    mdl.fit(Xr_train_s, yr_train)
    pred = mdl.predict(Xr_test_s)
    preds[name] = pred
    mae = mean_absolute_error(yr_test, pred)
    rmse = mean_squared_error(yr_test, pred, squared=False)
    r2 = r2_score(yr_test, pred)
    res_reg.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

res_reg_df = pd.DataFrame(res_reg).sort_values("RMSE")
res_reg_df.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)
print(res_reg_df)

best_name = res_reg_df.iloc[0]["Model"]
best_pred = preds[best_name]
best_r2 = res_reg_df.iloc[0]["R2"]

# Plot predicted vs actual for best model
fig = plt.figure(figsize=(6,5))
plt.scatter(yr_test, best_pred, s=14, alpha=0.6)
mn, mx = min(yr_test.min(), best_pred.min()), max(yr_test.max(), best_pred.max())
plt.plot([mn, mx], [mn, mx], "k--", lw=1)
plt.xlabel("Actual Monthly Spend"); plt.ylabel("Predicted Monthly Spend")
plt.title(f"Predicted vs Actual — {best_name} (R²={best_r2:.3f})")
plt.grid(True, alpha=0.3)
save_fig(fig, "regression_pred_vs_actual_best")

print(f"\nAll outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
