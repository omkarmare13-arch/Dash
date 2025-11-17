
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import requests
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score
)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

try:
    from kmodes.kmodes import KModes
    KMODES_AVAILABLE = True
except ImportError:
    KMODES_AVAILABLE = False

st.set_page_config(page_title="Luxury Delivery Dashboard", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;}
.sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_data(n=1000, seed=42):
    np.random.seed(seed)
    data = []
    for i in range(n):
        c = {}
        c["Customer_ID"] = f"CUST_{i+1:05d}"
        c["Age"] = np.random.randint(18, 75)
        c["Gender"] = np.random.choice(["Male", "Female", "Other"], p=[0.47, 0.50, 0.03])
        c["City_Tier"] = np.random.choice([1, 2, 3], p=[0.50, 0.35, 0.15])

        base_income = np.random.choice([75000, 125000, 200000, 300000, 500000, 1000000],
                                       p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10])
        c["Annual_Income"] = base_income * np.random.uniform(0.9, 1.3)
        income_tier = [75000, 125000, 200000, 300000, 500000, 1000000].index(base_income)

        freq_options = [0.5, 1, 3, 6, 12, 30, 52]
        if income_tier >= 4:
            freq = np.random.choice(freq_options[3:], p=[0.2, 0.3, 0.3, 0.2])
        else:
            freq = np.random.choice(freq_options[:5], p=[0.1, 0.2, 0.3, 0.25, 0.15])
        c["Luxury_Purchases_Per_Year"] = freq

        base_spend = (c["Annual_Income"] * 0.05) / 12
        freq_mult = 1.0 + (freq / 52) * 0.5
        city_mult = {1: 1.2, 2: 1.0, 3: 0.85}[c["City_Tier"]]
        c["Monthly_Luxury_Spend"] = base_spend * freq_mult * city_mult * np.random.uniform(0.8, 1.4)

        c["Brand_Loyalty_Score"] = int(np.clip(np.random.normal(6 + income_tier * 0.3, 1.5), 1, 10))
        c["Digital_Shopping_Comfort"] = int(np.clip(np.random.normal(3.5, 1), 1, 5))
        c["Interest_1_2hr_Service"] = int(np.clip(np.random.normal(3 + income_tier * 0.2, 1), 1, 5))

        wtp_base = 20 + (income_tier * 15)
        if c["Interest_1_2hr_Service"] >= 4:
            wtp_base += 30
        c["WTP_1hr_Delivery"] = int(np.clip(np.random.normal(wtp_base, 15), 5, 200))

        categories = {
            "Interested_Handbags": 0.45, "Interested_Watches": 0.40,
            "Interested_Fine_Jewelry": 0.35, "Interested_Limited_Sneakers": 0.18,
            "Interested_Beauty": 0.38, "Interested_Fragrances": 0.32,
            "Interested_Tech_Gadgets": 0.28
        }

        if c["Gender"] == "Female":
            categories["Interested_Handbags"] += 0.25
            categories["Interested_Beauty"] += 0.25
        elif c["Gender"] == "Male":
            categories["Interested_Watches"] += 0.25
            categories["Interested_Tech_Gadgets"] += 0.15

        for cat, prob in categories.items():
            c[cat] = 1 if np.random.random() < min(prob, 0.95) else 0

        features = {
            "Feat_Personalized_Packaging": 0.45,
            "Feat_Dedicated_Concierge": 0.35,
            "Feat_SameDay_Tailoring_Sizing": 0.25,
            "Feat_HighValue_Insurance": 0.50
        }

        if income_tier >= 4:
            for feat in features:
                features[feat] = min(0.90, features[feat] + 0.20)

        for feat, prob in features.items():
            c[feat] = 1 if np.random.random() < min(prob, 0.95) else 0

        if c["Monthly_Luxury_Spend"] >= 10000:
            c["Base_Price"] = np.random.uniform(500, 2000)
        elif c["Monthly_Luxury_Spend"] >= 5000:
            c["Base_Price"] = np.random.uniform(300, 800)
        else:
            c["Base_Price"] = np.random.uniform(50, 500)

        data.append(c)

    return pd.DataFrame(data)

def load_from_upload(file):
    try:
        return pd.read_csv(file), None
    except Exception as e:
        return None, str(e)

def load_from_github(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text)), None
    except Exception as e:
        return None, str(e)

def engineer_features(df):
    df = df.copy()

    cat_cols = [c for c in df.columns if "Interested_" in c]
    df["n_categories_interested"] = df[cat_cols].sum(axis=1) if cat_cols else 0

    feat_cols = [c for c in df.columns if "Feat_" in c]
    df["n_features_valued"] = df[feat_cols].sum(axis=1) if feat_cols else 0

    if "Likely_to_Use" not in df.columns:
        if "Interest_1_2hr_Service" in df.columns:
            df["Likely_to_Use"] = (df["Interest_1_2hr_Service"] >= 4).astype(int)
        elif "WTP_1hr_Delivery" in df.columns:
            df["Likely_to_Use"] = (df["WTP_1hr_Delivery"] >= df["WTP_1hr_Delivery"].median()).astype(int)
        else:
            df["Likely_to_Use"] = 0

    if "Base_Price" not in df.columns:
        if "Monthly_Luxury_Spend" in df.columns:
            df["Base_Price"] = df["Monthly_Luxury_Spend"] * np.random.uniform(0.05, 0.15, len(df))
        else:
            df["Base_Price"] = np.random.uniform(100, 1000, len(df))

    return df

def get_numeric_features(df):
    base = ["Age", "Annual_Income", "Monthly_Luxury_Spend", "Luxury_Purchases_Per_Year",
            "Brand_Loyalty_Score", "Digital_Shopping_Comfort", "Interest_1_2hr_Service",
            "WTP_1hr_Delivery", "City_Tier"]
    return [c for c in base if c in df.columns]

def fig_to_bytes(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

def download_button(df, filename, label="Download CSV"):
    st.download_button(label=label, data=df.to_csv(index=False), file_name=filename, mime="text/csv")

st.sidebar.title("🎛️ Data Controls")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Select Data Source:", ["Use Synthetic Data", "Upload CSV", "GitHub Raw URL"])

df = None
data_loaded = False

if data_source == "Use Synthetic Data":
    st.sidebar.subheader("Synthetic Data Settings")
    n = st.sidebar.slider("Sample Size", 100, 2000, 1000, 100)
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

    if st.sidebar.button("Generate Data") or "main_df" not in st.session_state:
        with st.spinner("Generating..."):
            df = generate_synthetic_data(n, seed)
            df = engineer_features(df)
            st.session_state["main_df"] = df
            data_loaded = True
            st.sidebar.success(f"✅ {len(df)} records")
    else:
        df = st.session_state.get("main_df")
        if df is not None:
            data_loaded = True
            st.sidebar.success(f"✅ {len(df)} records")

elif data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df, error = load_from_upload(uploaded)
        if df is not None:
            df = engineer_features(df)
            data_loaded = True
            st.sidebar.success(f"✅ {len(df)} records")
        else:
            st.sidebar.error(f"❌ {error}")

elif data_source == "GitHub Raw URL":
    url = st.sidebar.text_input("GitHub Raw CSV URL:")
    if st.sidebar.button("Load from GitHub"):
        if url:
            with st.spinner("Fetching..."):
                df, error = load_from_github(url)
                if df is not None:
                    df = engineer_features(df)
                    data_loaded = True
                    st.sidebar.success(f"✅ {len(df)} records")
                else:
                    st.sidebar.error(f"❌ {error}")

if data_loaded and df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"Rows: {len(df)}")
    st.sidebar.write(f"Columns: {len(df.columns)}")

st.title("🚀 Luxury Delivery ML Dashboard")
st.markdown("**Classification • Clustering • Association Rules • Dynamic Pricing**")
st.markdown("---")

if not data_loaded or df is None:
    st.info("👈 Select a data source from sidebar")
    st.stop()

tabs = st.tabs(["📊 Overview", "🎯 Classification", "🔍 Clustering", "🛒 Association Rules", "💰 Pricing"])

with tabs[0]:
    st.markdown("<h2 class="sub-header">Overview & Personas</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Avg Income", f"${df["Annual_Income"].mean():,.0f}" if "Annual_Income" in df.columns else "N/A")
    col3.metric("Avg Spend", f"${df["Monthly_Luxury_Spend"].mean():,.0f}" if "Monthly_Luxury_Spend" in df.columns else "N/A")
    col4.metric("Adoption", f"{(df["Likely_to_Use"].sum()/len(df)*100):.1f}%" if "Likely_to_Use" in df.columns else "N/A")

    st.markdown("---")
    st.subheader("🎭 K-Means Personas")
    k = st.slider("Number of Personas (K)", 2, 8, 4, key="persona_k")

    numeric = get_numeric_features(df)

    if len(numeric) >= 3:
        X = df[numeric].fillna(df[numeric].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        df["Persona"] = kmeans.fit_predict(X_scaled)

        spend_order = df.groupby("Persona")["Monthly_Luxury_Spend"].mean().sort_values()
        names = {}
        for i, (pid, _) in enumerate(spend_order.items()):
            if i == 0:
                names[pid] = "Budget-Conscious"
            elif i == len(spend_order) - 1:
                names[pid] = "Ultra-Premium"
            elif i == len(spend_order) - 2:
                names[pid] = "Affluent"
            else:
                names[pid] = "Aspirational"

        df["Persona_Name"] = df["Persona"].map(names)

        st.markdown("#### Persona Profiles")
        profile_cols = numeric + ["n_categories_interested", "n_features_valued"]
        profile_cols = [c for c in profile_cols if c in df.columns]

        profile = df.groupby("Persona_Name")[profile_cols].mean().round(2)
        profile["Count"] = df.groupby("Persona_Name").size()
        profile["Percentage"] = (profile["Count"] / len(df) * 100).round(1)

        st.dataframe(profile, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            download_button(df[["Customer_ID", "Persona_Name"] + numeric], "persona_labels.csv", "📥 Download Labels")
        with col2:
            download_button(profile, "persona_profiles.csv", "📥 Download Profiles")
    else:
        st.warning("⚠️ Insufficient numeric features")

with tabs[1]:
    st.markdown("<h2 class="sub-header">Classification</h2>", unsafe_allow_html=True)

    class_upload = st.file_uploader("Upload CSV (optional)", type=["csv"], key="class_upload")

    if class_upload:
        df_class, error = load_from_upload(class_upload)
        if df_class is not None:
            df_class = engineer_features(df_class)
            st.success(f"✅ Using uploaded ({len(df_class)} records)")
        else:
            st.error(f"❌ {error}")
            df_class = df
    else:
        df_class = df

    numeric = get_numeric_features(df_class)
    cats = [c for c in df_class.columns if "Interested_" in c]
    feats = [c for c in df_class.columns if "Feat_" in c]

    X_cols = numeric + cats + feats + ["n_categories_interested", "n_features_valued"]
    X_cols = [c for c in X_cols if c in df_class.columns and c != "Monthly_Luxury_Spend"]

    if "Likely_to_Use" in df_class.columns and len(X_cols) > 0:
        X = df_class[X_cols].fillna(df_class[X_cols].median())
        y = df_class["Likely_to_Use"]

        if len(X) > 20 and y.nunique() == 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            lr = LogisticRegression(random_state=42, max_iter=1000)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)

            lr.fit(X_train_s, y_train)
            rf.fit(X_train_s, y_train)

            results = []
            for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
                y_pred = model.predict(X_test_s)
                y_proba = model.predict_proba(X_test_s)[:, 1]

                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_test, y_proba)
                })

            results_df = pd.DataFrame(results)

            st.markdown("#### Model Performance")
            st.dataframe(results_df, use_container_width=True)

            best_idx = results_df["ROC-AUC"].idxmax()
            best_name = results_df.loc[best_idx, "Model"]
            best_model = lr if best_name == "Logistic Regression" else rf

            st.success(f"🏆 Best: {best_name} (AUC: {results_df.loc[best_idx, "ROC-AUC"]:.4f})")

            st.markdown("#### ROC Curve")
            y_proba = best_model.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc_score(y_test, y_proba):.3f}")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlabel("False Positive Rate", fontweight="bold")
            ax.set_ylabel("True Positive Rate", fontweight="bold")
            ax.set_title("ROC Curve", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 PNG", fig_to_bytes(fig, "png"), "roc_curve.png", "image/png")
            with col2:
                st.download_button("📥 JPG", fig_to_bytes(fig, "jpg"), "roc_curve.jpg", "image/jpeg")
            plt.close()

            pred_df = pd.DataFrame({
                "Customer_ID": df_class.loc[X_test.index, "Customer_ID"] if "Customer_ID" in df_class.columns else X_test.index,
                "True_Label": y_test,
                "Predicted": best_model.predict(X_test_s),
                "Probability": y_proba
            })

            download_button(pred_df, "predictions.csv", "📥 Download Predictions")
        else:
            st.warning("⚠️ Insufficient data")
    else:
        st.warning("⚠️ Target missing or no features")

with tabs[2]:
    st.markdown("<h2 class="sub-header">Clustering</h2>", unsafe_allow_html=True)

    st.subheader("K-Means")
    k = st.slider("K", 2, 10, 5, key="kmeans_k")

    numeric = get_numeric_features(df)

    if len(numeric) >= 3:
        X = df[numeric].fillna(df[numeric].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

        profile = df.groupby("KMeans_Cluster")[numeric].mean().round(2)
        profile["Count"] = df.groupby("KMeans_Cluster").size()
        profile["Percentage"] = (profile["Count"] / len(df) * 100).round(1)

        st.dataframe(profile, use_container_width=True)
        download_button(profile, "kmeans_profiles.csv", "📥 Download")
    else:
        st.warning("⚠️ Insufficient features")

    st.markdown("---")
    st.subheader("K-Modes")

    if KMODES_AVAILABLE:
        cats = [c for c in df.columns if ("Interested_" in c or "Feat_" in c) and df[c].dtype in ["int64", "object"]]

        if len(cats) >= 3:
            k = st.slider("K", 2, 8, 4, key="kmodes_k")

            X = df[cats].fillna(0).astype(int)

            try:
                km = KModes(n_clusters=k, random_state=42, n_init=10)
                df["KModes_Cluster"] = km.fit_predict(X)

                modes = pd.DataFrame(km.cluster_centroids_, columns=cats).T
                modes.columns = [f"Mode {i+1}" for i in range(k)]
                modes["Sizes"] = df.groupby("KModes_Cluster").size().values

                st.dataframe(modes, use_container_width=True)
                download_button(modes, "kmodes_modes.csv", "📥 Download")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.info("ℹ️ Insufficient categorical columns")
    else:
        st.info("ℹ️ K-Modes unavailable. Install: pip install kmodes")

with tabs[3]:
    st.markdown("<h2 class="sub-header">Association Rules</h2>", unsafe_allow_html=True)

    if not MLXTEND_AVAILABLE:
        st.error("❌ mlxtend unavailable. Install: pip install mlxtend")
    else:
        apriori_upload = st.file_uploader("Upload CSV (optional)", type=["csv"], key="apriori_upload")

        if apriori_upload:
            df_apriori, error = load_from_upload(apriori_upload)
            if df_apriori is not None:
                df_apriori = engineer_features(df_apriori)
                st.success(f"✅ Using uploaded ({len(df_apriori)} records)")
            else:
                st.error(f"❌ {error}")
                df_apriori = df
        else:
            df_apriori = df

        if len(df_apriori) > 10000:
            st.warning(f"⚠️ Large dataset ({len(df_apriori)} rows) may be slow")

        basket_cols = [c for c in df_apriori.columns if ("Interested_" in c or "Feat_" in c) and df_apriori[c].dtype in ["int64", "float64"]]

        if len(basket_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                min_sup = st.slider("Min Support (%)", 1, 50, 10) / 100
            with col2:
                min_conf = st.slider("Min Confidence (%)", 10, 90, 50) / 100

            top_n = st.slider("Top N Rules", 5, 50, 10)

            if st.button("🔍 Generate Rules", type="primary"):
                with st.spinner("Mining..."):
                    try:
                        basket = df_apriori[basket_cols].astype(bool)
                        freq = apriori(basket, min_support=min_sup, use_colnames=True)

                        if len(freq) > 0:
                            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

                            if len(rules) > 0:
                                rules["antecedent_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                                rules["consequent_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
                                rules = rules.sort_values("lift", ascending=False).head(top_n)

                                display = rules[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].copy()
                                display.columns = ["If", "Then", "Support", "Confidence", "Lift"]

                                st.markdown(f"#### Top {len(display)} Rules")
                                st.dataframe(display, use_container_width=True)

                                download_button(display, "association_rules.csv", "📥 Download")
                                st.success(f"✅ Found {len(rules)} rules")
                            else:
                                st.warning("⚠️ No rules found. Lower confidence.")
                        else:
                            st.warning("⚠️ No frequent itemsets. Lower support.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Insufficient basket columns")

with tabs[4]:
    st.markdown("<h2 class="sub-header">Regression & Pricing</h2>", unsafe_allow_html=True)

    if "Monthly_Luxury_Spend" not in df.columns:
        st.error("❌ Target missing")
    else:
        numeric = [c for c in get_numeric_features(df) if c != "Monthly_Luxury_Spend"]
        cats = [c for c in df.columns if "Interested_" in c]
        feats = [c for c in df.columns if "Feat_" in c]

        X_cols = numeric + cats + feats + ["n_categories_interested", "n_features_valued"]
        X_cols = [c for c in X_cols if c in df.columns]

        if len(X_cols) > 0:
            X = df[X_cols].fillna(df[X_cols].median())
            y = df["Monthly_Luxury_Spend"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            lr = LinearRegression()
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

            lr.fit(X_train_s, y_train)
            gb.fit(X_train_s, y_train)

            results = []
            for name, model in [("Linear Regression", lr), ("Gradient Boosting", gb)]:
                y_pred = model.predict(X_test_s)
                results.append({
                    "Model": name,
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R²": r2_score(y_test, y_pred)
                })

            results_df = pd.DataFrame(results).sort_values("RMSE")

            st.markdown("#### Model Performance")
            st.dataframe(results_df, use_container_width=True)

            best_name = results_df.iloc[0]["Model"]
            best_model = lr if best_name == "Linear Regression" else gb
            best_r2 = results_df.iloc[0]["R²"]

            st.success(f"🏆 Best: {best_name} (R²: {best_r2:.4f})")

            X_all_s = scaler.transform(X)
            df["Predicted_Spend"] = best_model.predict(X_all_s)

            st.markdown("---")
            st.markdown("### 💎 Dynamic Pricing")

            with st.expander("⚙️ Pricing Config"):
                col1, col2 = st.columns(2)
                with col1:
                    cap_w = st.slider("Capacity Weight", 0.0, 1.0, 0.30, 0.05)
                    loy_w = st.slider("Loyalty Weight", 0.0, 1.0, 0.25, 0.05)
                with col2:
                    min_mult = st.slider("Min Multiplier", 0.5, 1.0, 0.70, 0.05)
                    max_mult = st.slider("Max Multiplier", 1.0, 2.0, 1.25, 0.05)

            df["Capacity_Score"] = (df["Predicted_Spend"] / df["Predicted_Spend"].median()).clip(0.5, 2.0)
            df["Loyalty_Mult"] = df["Brand_Loyalty_Score"] / 10.0 if "Brand_Loyalty_Score" in df.columns else 0.5
            df["Price_Multiplier"] = (
                (df["Capacity_Score"] - 1.0) * cap_w +
                (df["Loyalty_Mult"] - 0.5) * loy_w +
                1.0
            ).clip(min_mult, max_mult)

            df["Personalized_Price"] = df["Base_Price"] * df["Price_Multiplier"]
            df["Price_Adj_Pct"] = ((df["Personalized_Price"] - df["Base_Price"]) / df["Base_Price"]) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Multiplier", f"{df["Price_Multiplier"].mean():.3f}x")
            col2.metric("Premium %", f"{(df["Price_Adj_Pct"].sum() / len(df) * 100):.1f}%")
            col3.metric("Revenue Lift", f"${(df["Personalized_Price"].sum() - df["Base_Price"].sum()):,.0f}")

            st.markdown("#### 📊 Charts")

            fig1, ax1 = plt.subplots(figsize=(8, 6))
            y_pred_test = best_model.predict(X_test_s)
            ax1.scatter(y_test, y_pred_test, alpha=0.5, s=30)
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
            ax1.set_xlabel("Actual", fontweight="bold")
            ax1.set_ylabel("Predicted", fontweight="bold")
            ax1.set_title(f"Predicted vs Actual (R²={best_r2:.3f})", fontweight="bold")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()

            st.pyplot(fig1)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 PNG", fig_to_bytes(fig1, "png"), "pred_actual.png", "image/png")
            with col2:
                st.download_button("📥 JPG", fig_to_bytes(fig1, "jpg"), "pred_actual.jpg", "image/jpeg")
            plt.close()

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            scatter = ax2.scatter(df["Base_Price"], df["Personalized_Price"],
                                 c=df["Price_Multiplier"], cmap="RdYlGn", alpha=0.6, s=30)
            min_p = min(df["Base_Price"].min(), df["Personalized_Price"].min())
            max_p = max(df["Base_Price"].max(), df["Personalized_Price"].max())
            ax2.plot([min_p, max_p], [min_p, max_p], "k--", linewidth=2)
            ax2.set_xlabel("Base Price", fontweight="bold")
            ax2.set_ylabel("Personalized Price", fontweight="bold")
            ax2.set_title("Personalized vs Base", fontweight="bold")
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label="Multiplier")
            plt.tight_layout()

            st.pyplot(fig2)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 PNG", fig_to_bytes(fig2, "png"), "price_comp.png", "image/png")
            with col2:
                st.download_button("📥 JPG", fig_to_bytes(fig2, "jpg"), "price_comp.jpg", "image/jpeg")
            plt.close()

            if "Brand_Loyalty_Score" in df.columns:
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                bins = pd.cut(df["Brand_Loyalty_Score"], bins=[0, 3, 6, 8, 10], labels=["Low", "Med", "High", "V.High"])
                mult_loy = df.groupby(bins, observed=True)["Price_Multiplier"].mean()

                if len(mult_loy) > 0:
                    ax3.bar(range(len(mult_loy)), mult_loy.values, color="#3498db", alpha=0.8)
                    ax3.axhline(y=1.0, color="r", linestyle="--", linewidth=2)
                    ax3.set_xticks(range(len(mult_loy)))
                    ax3.set_xticklabels(mult_loy.index)
                    ax3.set_ylabel("Avg Multiplier", fontweight="bold")
                    ax3.set_title("Multiplier by Loyalty", fontweight="bold")
                    ax3.grid(True, alpha=0.3, axis="y")
                    plt.tight_layout()

                    st.pyplot(fig3)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("📥 PNG", fig_to_bytes(fig3, "png"), "mult_loyalty.png", "image/png")
                    with col2:
                        st.download_button("📥 JPG", fig_to_bytes(fig3, "jpg"), "mult_loyalty.jpg", "image/jpeg")
                    plt.close()

            st.markdown("#### 📋 Digital Price Chart")

            chart_cols = ["Customer_ID", "Monthly_Luxury_Spend", "Predicted_Spend",
                         "Base_Price", "Price_Multiplier", "Personalized_Price", "Price_Adj_Pct"]
            chart_cols = [c for c in chart_cols if c in df.columns]

            price_chart = df[chart_cols].round(2)
            st.dataframe(price_chart.head(20), use_container_width=True)

            download_button(price_chart, "digital_price_chart.csv", "📥 Download Price Chart")
        else:
            st.warning("⚠️ No features available")

st.markdown("---")
st.markdown("<div style="text-align: center; color: gray;">Built with Streamlit • ML-powered pricing</div>",
           unsafe_allow_html=True)
