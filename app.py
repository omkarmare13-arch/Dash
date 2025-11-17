import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import requests
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, silhouette_score
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
st.markdown("**Complete ML Analytics: Classification • Clustering • Association Rules • Regression**")
st.markdown("---")

if not data_loaded or df is None:
    st.info("👈 Select a data source from sidebar")
    st.stop()

tabs = st.tabs(["📊 Overview", "🎯 Classification (All Algorithms)", "🔍 Clustering Analysis", "🛒 Association Rules", "📈 Regression (Linear, Ridge, Lasso)", "💰 Dynamic Pricing"])

with tabs[0]:
    st.markdown('<h2 class="sub-header">Overview & Personas</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Avg Income", f"${df['Annual_Income'].mean():,.0f}" if "Annual_Income" in df.columns else "N/A")
    col3.metric("Avg Spend", f"${df['Monthly_Luxury_Spend'].mean():,.0f}" if "Monthly_Luxury_Spend" in df.columns else "N/A")
    col4.metric("Adoption", f"{(df['Likely_to_Use'].sum()/len(df)*100):.1f}%" if "Likely_to_Use" in df.columns else "N/A")

    st.markdown("---")
    st.subheader("🎭 Quick K-Means Personas")
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
    st.markdown('<h2 class="sub-header">Classification - All Algorithms Comparison</h2>', unsafe_allow_html=True)
    st.markdown("**Comparing: Logistic Regression, Random Forest, Decision Tree, SVM, Naive Bayes, KNN, Gradient Boosting**")

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

            # Define all classification models
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
                "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
                "Naive Bayes": GaussianNB(),
                "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

            st.markdown("### 🔄 Training All Models...")
            progress_bar = st.progress(0)
            results = []
            
            for idx, (name, model) in enumerate(models.items()):
                with st.spinner(f"Training {name}..."):
                    # Train model
                    model.fit(X_train_s, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_s)
                    y_proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    # Calculate metrics
                    results.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, zero_division=0),
                        "Recall": recall_score(y_test, y_pred, zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else accuracy_score(y_test, y_pred)
                    })
                    
                progress_bar.progress((idx + 1) / len(models))

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("ROC-AUC", ascending=False)

            st.markdown("### 📊 Performance Comparison - All Algorithms")
            st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color='lightgreen'), use_container_width=True)

            # Download results
            download_button(results_df, "classification_results.csv", "📥 Download Results")

            # Best model
            best_idx = results_df["ROC-AUC"].idxmax()
            best_name = results_df.loc[best_idx, "Model"]
            best_model = models[best_name]
            best_model.fit(X_train_s, y_train)  # Retrain best model
            
            st.success(f"🏆 **Best Model: {best_name}** | ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f} | Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")

            # Visualization - Performance Comparison
            st.markdown("### 📈 Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
                x = np.arange(len(results_df))
                width = 0.2
                
                for i, metric in enumerate(metrics_to_plot):
                    ax1.bar(x + i*width, results_df[metric], width, label=metric)
                
                ax1.set_xlabel('Models', fontweight='bold')
                ax1.set_ylabel('Score', fontweight='bold')
                ax1.set_title('Classification Metrics Comparison', fontweight='bold', fontsize=14)
                ax1.set_xticks(x + width * 1.5)
                ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig1)
                st.download_button("📥 PNG", fig_to_bytes(fig1, "png"), "metrics_comparison.png", "image/png")
                plt.close()
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.barh(results_df['Model'], results_df['ROC-AUC'], color='steelblue')
                ax2.set_xlabel('ROC-AUC Score', fontweight='bold')
                ax2.set_title('ROC-AUC Comparison Across Models', fontweight='bold', fontsize=14)
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, v in enumerate(results_df['ROC-AUC']):
                    ax2.text(v + 0.01, i, f'{v:.4f}', va='center')
                
                plt.tight_layout()
                st.pyplot(fig2)
                st.download_button("📥 PNG", fig_to_bytes(fig2, "png"), "roc_comparison.png", "image/png")
                plt.close()

            # ROC Curves for all models
            st.markdown("### 📉 ROC Curves - All Models")
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            
            for name, model in models.items():
                if hasattr(model, 'predict_proba'):
                    model.fit(X_train_s, y_train)
                    y_proba = model.predict_proba(X_test_s)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc_score = roc_auc_score(y_test, y_proba)
                    ax3.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_score:.3f})')
            
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            ax3.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
            ax3.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
            ax3.set_title('ROC Curves - All Classification Models', fontweight='bold', fontsize=14)
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig3)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 PNG", fig_to_bytes(fig3, "png"), "all_roc_curves.png", "image/png")
            with col2:
                st.download_button("📥 JPG", fig_to_bytes(fig3, "jpg"), "all_roc_curves.jpg", "image/jpeg")
            plt.close()

            # Confusion Matrix for Best Model
            st.markdown(f"### 🎯 Confusion Matrix - {best_name}")
            y_pred_best = best_model.predict(X_test_s)
            cm = confusion_matrix(y_test, y_pred_best)
            
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar_kws={'label': 'Count'})
            ax4.set_xlabel('Predicted Label', fontweight='bold')
            ax4.set_ylabel('True Label', fontweight='bold')
            ax4.set_title(f'Confusion Matrix - {best_name}', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            st.pyplot(fig4)
            st.download_button("📥 PNG", fig_to_bytes(fig4, "png"), "confusion_matrix.png", "image/png")
            plt.close()

            # Classification Report
            st.markdown(f"### 📋 Detailed Classification Report - {best_name}")
            report = classification_report(y_test, y_pred_best, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            download_button(report_df, "classification_report.csv", "📥 Download Report")

            # Predictions export
            pred_df = pd.DataFrame({
                "Customer_ID": df_class.loc[X_test.index, "Customer_ID"] if "Customer_ID" in df_class.columns else X_test.index,
                "True_Label": y_test,
                "Predicted": y_pred_best,
                "Probability": best_model.predict_proba(X_test_s)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_best
            })
            download_button(pred_df, "predictions_best_model.csv", "📥 Download Predictions")

        else:
            st.warning("⚠️ Insufficient data or target not binary")
    else:
        st.warning("⚠️ Target 'Likely_to_Use' missing or no features available")

with tabs[2]:
    st.markdown('<h2 class="sub-header">Clustering Analysis with Interpretation</h2>', unsafe_allow_html=True)

    # K-Means Clustering
    st.markdown("### 🔵 K-Means Clustering")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        k = st.slider("Number of Clusters (K)", 2, 10, 5, key="kmeans_k_main")
    with col2:
        st.info("📊 Optimal K will be analyzed using Elbow & Silhouette methods")

    numeric = get_numeric_features(df)

    if len(numeric) >= 3:
        X = df[numeric].fillna(df[numeric].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow Method
        st.markdown("#### 📉 Elbow Method (Finding Optimal K)")
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(11, len(df)//10))
        
        for k_test in K_range:
            kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init=20)
            kmeans_test.fit(X_scaled)
            inertias.append(kmeans_test.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans_test.labels_))

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold')
        ax1.set_title('Elbow Method', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)', fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontweight='bold')
        ax2.set_title('Silhouette Score Analysis', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig1)
        st.download_button("📥 Download Chart", fig_to_bytes(fig1, "png"), "elbow_silhouette.png", "image/png")
        plt.close()

        # Perform K-Means with selected K
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, df["KMeans_Cluster"])
        st.success(f"✅ Silhouette Score for K={k}: **{silhouette_avg:.4f}** (Higher is better, >0.5 is good)")

        # Cluster Profiles
        st.markdown("#### 📊 Cluster Profiles & Characteristics")
        profile = df.groupby("KMeans_Cluster")[numeric].mean().round(2)
        profile["Count"] = df.groupby("KMeans_Cluster").size()
        profile["Percentage"] = (profile["Count"] / len(df) * 100).round(1)

        st.dataframe(profile.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), 
                     use_container_width=True)

        # Cluster Interpretation
        st.markdown("#### 🔍 Cluster Interpretation & Business Insights")
        
        for cluster_id in range(k):
            cluster_data = df[df["KMeans_Cluster"] == cluster_id]
            
            with st.expander(f"**Cluster {cluster_id}** ({len(cluster_data)} customers, {len(cluster_data)/len(df)*100:.1f}%)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Income", f"${cluster_data['Annual_Income'].mean():,.0f}" if "Annual_Income" in cluster_data.columns else "N/A")
                    st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f}" if "Age" in cluster_data.columns else "N/A")
                
                with col2:
                    st.metric("Monthly Spend", f"${cluster_data['Monthly_Luxury_Spend'].mean():,.0f}" if "Monthly_Luxury_Spend" in cluster_data.columns else "N/A")
                    st.metric("Loyalty Score", f"{cluster_data['Brand_Loyalty_Score'].mean():.1f}/10" if "Brand_Loyalty_Score" in cluster_data.columns else "N/A")
                
                with col3:
                    st.metric("Purchase Freq/Yr", f"{cluster_data['Luxury_Purchases_Per_Year'].mean():.1f}" if "Luxury_Purchases_Per_Year" in cluster_data.columns else "N/A")
                    st.metric("Adoption Rate", f"{cluster_data['Likely_to_Use'].mean()*100:.1f}%" if "Likely_to_Use" in cluster_data.columns else "N/A")
                
                # Interpretation
                avg_income = cluster_data['Annual_Income'].mean() if 'Annual_Income' in cluster_data.columns else 0
                avg_spend = cluster_data['Monthly_Luxury_Spend'].mean() if 'Monthly_Luxury_Spend' in cluster_data.columns else 0
                
                if avg_income > 300000 and avg_spend > 8000:
                    interpretation = "🌟 **Ultra-High-Value Segment**: Premium customers with high income and spending. Focus on exclusive experiences and concierge services."
                elif avg_income > 150000 and avg_spend > 4000:
                    interpretation = "💎 **Affluent Shoppers**: Regular luxury purchasers. Target with loyalty programs and personalized offers."
                elif avg_spend > 2000:
                    interpretation = "🎯 **Aspirational Buyers**: Moderate-to-high spenders. Upsell opportunities through flexible payment options."
                else:
                    interpretation = "📊 **Price-Conscious Segment**: Value-oriented customers. Emphasize convenience and competitive pricing."
                
                st.info(interpretation)

        # Visualization
        st.markdown("#### 📈 Cluster Visualization")
        
        if "Annual_Income" in df.columns and "Monthly_Luxury_Spend" in df.columns:
            fig2, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(df["Annual_Income"], df["Monthly_Luxury_Spend"], 
                               c=df["KMeans_Cluster"], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
            
            # Plot centroids
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            income_idx = numeric.index("Annual_Income")
            spend_idx = numeric.index("Monthly_Luxury_Spend")
            
            ax.scatter(centroids[:, income_idx], centroids[:, spend_idx], 
                      c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
            
            ax.set_xlabel('Annual Income ($)', fontweight='bold', fontsize=12)
            ax.set_ylabel('Monthly Luxury Spend ($)', fontweight='bold', fontsize=12)
            ax.set_title('K-Means Clustering: Income vs Spending', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plt.tight_layout()
            
            st.pyplot(fig2)
            st.download_button("📥 Download Visualization", fig_to_bytes(fig2, "png"), "cluster_visualization.png", "image/png")
            plt.close()

        download_button(profile, "kmeans_cluster_profiles.csv", "📥 Download Cluster Profiles")
        
        # Export cluster labels
        cluster_export = df[["Customer_ID"] + numeric + ["KMeans_Cluster"]] if "Customer_ID" in df.columns else df[numeric + ["KMeans_Cluster"]]
        download_button(cluster_export, "cluster_labels.csv", "📥 Download Cluster Labels")

    else:
        st.warning("⚠️ Insufficient numeric features for clustering")

    # K-Modes for Categorical Data
    st.markdown("---")
    st.markdown("### 🟠 K-Modes Clustering (Categorical Data)")

    if KMODES_AVAILABLE:
        cats = [c for c in df.columns if ("Interested_" in c or "Feat_" in c) and df[c].dtype in ["int64", "object"]]

        if len(cats) >= 3:
            k_modes = st.slider("Number of Modes (K)", 2, 8, 4, key="kmodes_k_main")

            X_cat = df[cats].fillna(0).astype(int)

            try:
                km = KModes(n_clusters=k_modes, random_state=42, n_init=10)
                df["KModes_Cluster"] = km.fit_predict(X_cat)

                st.success(f"✅ K-Modes clustering completed with {k_modes} clusters")

                # Mode profiles
                modes = pd.DataFrame(km.cluster_centroids_, columns=cats).T
                modes.columns = [f"Mode {i+1}" for i in range(k_modes)]
                
                sizes = df.groupby("KModes_Cluster").size()
                modes["Cluster_Sizes"] = sizes.values

                st.markdown("#### 📊 Categorical Modes (Most Common Values)")
                st.dataframe(modes, use_container_width=True)

                # Interpretation
                st.markdown("#### 💡 Categorical Cluster Insights")
                for mode_id in range(k_modes):
                    mode_data = df[df["KModes_Cluster"] == mode_id]
                    
                    interested_cols = [c for c in cats if "Interested_" in c]
                    top_interests = mode_data[interested_cols].sum().sort_values(ascending=False).head(3)
                    
                    with st.expander(f"**Mode {mode_id}** ({len(mode_data)} customers)"):
                        st.write("**Top Product Interests:**")
                        for interest, count in top_interests.items():
                            pct = (count / len(mode_data)) * 100
                            st.write(f"- {interest.replace('Interested_', '')}: {pct:.1f}%")

                download_button(modes, "kmodes_profiles.csv", "📥 Download Modes")
            except Exception as e:
                st.error(f"❌ Error in K-Modes: {str(e)}")
        else:
            st.info("ℹ️ Insufficient categorical columns for K-Modes clustering")
    else:
        st.info("ℹ️ K-Modes unavailable. Install with: `pip install kmodes`")

with tabs[3]:
    st.markdown('<h2 class="sub-header">Association Rule Mining</h2>', unsafe_allow_html=True)
    st.markdown("**Discover patterns in customer product interests and feature preferences**")

    if not MLXTEND_AVAILABLE:
        st.error("❌ mlxtend library not available. Install with: `pip install mlxtend`")
    else:
        apriori_upload = st.file_uploader("Upload CSV (optional)", type=["csv"], key="apriori_upload")

        if apriori_upload:
            df_apriori, error = load_from_upload(apriori_upload)
            if df_apriori is not None:
                df_apriori = engineer_features(df_apriori)
                st.success(f"✅ Using uploaded data ({len(df_apriori)} records)")
            else:
                st.error(f"❌ {error}")
                df_apriori = df
        else:
            df_apriori = df

        if len(df_apriori) > 10000:
            st.warning(f"⚠️ Large dataset ({len(df_apriori)} rows) - computation may take time")

        basket_cols = [c for c in df_apriori.columns if ("Interested_" in c or "Feat_" in c) and df_apriori[c].dtype in ["int64", "float64"]]

        if len(basket_cols) >= 2:
            st.markdown("#### ⚙️ Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_sup = st.slider("Minimum Support (%)", 1, 50, 10, help="How frequently items appear together") / 100
            with col2:
                min_conf = st.slider("Minimum Confidence (%)", 10, 90, 50, help="Strength of the rule") / 100
            with col3:
                top_n = st.slider("Top N Rules to Display", 5, 50, 15)

            if st.button("🔍 Mine Association Rules", type="primary"):
                with st.spinner("Mining frequent patterns and generating rules..."):
                    try:
                        # Convert to boolean basket
                        basket = df_apriori[basket_cols].astype(bool)
                        
                        # Find frequent itemsets
                        freq = apriori(basket, min_support=min_sup, use_colnames=True)

                        if len(freq) > 0:
                            st.success(f"✅ Found {len(freq)} frequent itemsets")
                            
                            # Generate association rules
                            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

                            if len(rules) > 0:
                                # Process rules
                                rules["antecedent_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)).replace("Interested_", "").replace("Feat_", ""))
                                rules["consequent_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)).replace("Interested_", "").replace("Feat_", ""))
                                
                                # Sort by lift
                                rules = rules.sort_values("lift", ascending=False)
                                
                                # Display top rules
                                display = rules[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].head(top_n).copy()
                                display.columns = ["If Customer Likes", "Then Likely Likes", "Support", "Confidence", "Lift"]
                                display["Support"] = display["Support"].round(4)
                                display["Confidence"] = display["Confidence"].round(4)
                                display["Lift"] = display["Lift"].round(4)

                                st.markdown(f"### 📋 Top {len(display)} Association Rules")
                                st.dataframe(display.style.highlight_max(subset=["Lift"], color='lightgreen'), use_container_width=True)

                                # Interpretation guide
                                st.markdown("#### 📖 How to Read These Rules:")
                                st.info("""
                                - **Support**: How frequently the items appear together (higher = more common)
                                - **Confidence**: How often the rule is correct (higher = more reliable)
                                - **Lift**: How much more likely items appear together vs. independently (>1 = positive correlation)
                                """)

                                # Key insights
                                st.markdown("#### 💡 Key Business Insights")
                                top_rule = display.iloc[0]
                                st.success(f"""
                                **Strongest Association** (Lift: {top_rule['Lift']:.2f}):  
                                Customers interested in **{top_rule['If Customer Likes']}** are {top_rule['Lift']:.1f}x more likely to be interested in **{top_rule['Then Likely Likes']}**
                                """)

                                # Visualization
                                st.markdown("#### 📊 Rule Visualization")
                                
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # Support vs Confidence scatter
                                scatter1 = ax1.scatter(rules['support'], rules['confidence'], 
                                                     c=rules['lift'], s=100, cmap='viridis', alpha=0.6, edgecolors='black')
                                ax1.set_xlabel('Support', fontweight='bold')
                                ax1.set_ylabel('Confidence', fontweight='bold')
                                ax1.set_title('Support vs Confidence (colored by Lift)', fontweight='bold')
                                ax1.grid(True, alpha=0.3)
                                plt.colorbar(scatter1, ax=ax1, label='Lift')
                                
                                # Top rules by lift
                                top_10 = rules.head(10)
                                ax2.barh(range(len(top_10)), top_10['lift'], color='steelblue')
                                ax2.set_yticks(range(len(top_10)))
                                ax2.set_yticklabels([f"{row['antecedent_str'][:20]}... → {row['consequent_str'][:20]}..." 
                                                     for _, row in top_10.iterrows()], fontsize=8)
                                ax2.set_xlabel('Lift', fontweight='bold')
                                ax2.set_title('Top 10 Rules by Lift', fontweight='bold')
                                ax2.grid(True, alpha=0.3, axis='x')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.download_button("📥 Download Chart", fig_to_bytes(fig, "png"), "association_rules_viz.png", "image/png")
                                plt.close()

                                # Download options
                                col1, col2 = st.columns(2)
                                with col1:
                                    download_button(display, "association_rules_top.csv", "📥 Download Top Rules")
                                with col2:
                                    full_export = rules[["antecedent_str", "consequent_str", "support", "confidence", "lift"]]
                                    full_export.columns = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
                                    download_button(full_export, "association_rules_all.csv", "📥 Download All Rules")

                                st.success(f"✅ Generated {len(rules)} total association rules")
                            else:
                                st.warning("⚠️ No rules meet the confidence threshold. Try lowering the minimum confidence.")
                        else:
                            st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support.")
                    except Exception as e:
                        st.error(f"❌ Error during mining: {str(e)}")
        else:
            st.warning("⚠️ Insufficient categorical/binary columns for association rule mining")

with tabs[4]:
    st.markdown('<h2 class="sub-header">Regression Analysis: Linear, Ridge & Lasso</h2>', unsafe_allow_html=True)
    st.markdown("**Predicting Monthly Luxury Spend using three regression techniques**")

    if "Monthly_Luxury_Spend" not in df.columns:
        st.error("❌ Target variable 'Monthly_Luxury_Spend' not found in dataset")
    else:
        numeric = [c for c in get_numeric_features(df) if c != "Monthly_Luxury_Spend"]
        cats = [c for c in df.columns if "Interested_" in c]
        feats = [c for c in df.columns if "Feat_" in c]

        X_cols = numeric + cats + feats + ["n_categories_interested", "n_features_valued"]
        X_cols = [c for c in X_cols if c in df.columns]

        if len(X_cols) > 0:
            X = df[X_cols].fillna(df[X_cols].median())
            y = df["Monthly_Luxury_Spend"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scaling
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Ridge and Lasso parameters
            st.markdown("### ⚙️ Regularization Parameters")
            col1, col2 = st.columns(2)
            with col1:
                ridge_alpha = st.slider("Ridge Alpha (L2 penalty)", 0.01, 10.0, 1.0, 0.1)
            with col2:
                lasso_alpha = st.slider("Lasso Alpha (L1 penalty)", 0.01, 10.0, 1.0, 0.1)

            st.info("💡 **Alpha** controls regularization strength. Higher values = more penalty on coefficients.")

            # Train all three models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=ridge_alpha),
                "Lasso Regression": Lasso(alpha=lasso_alpha, max_iter=10000)
            }

            results = []
            trained_models = {}

            st.markdown("### 🔄 Training Models...")
            progress_bar = st.progress(0)

            for idx, (name, model) in enumerate(models.items()):
                with st.spinner(f"Training {name}..."):
                    model.fit(X_train_s, y_train)
                    trained_models[name] = model
                    
                    y_pred = model.predict(X_test_s)
                    
                    results.append({
                        "Model": name,
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "R² Score": r2_score(y_test, y_pred),
                        "Explained Variance": r2_score(y_test, y_pred) * 100
                    })
                    
                progress_bar.progress((idx + 1) / len(models))

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("R² Score", ascending=False)

            st.markdown("### 📊 Model Performance Comparison")
            st.dataframe(results_df.style.highlight_max(subset=["R² Score"], color='lightgreen')
                        .highlight_min(subset=["MAE", "RMSE"], color='lightgreen'), 
                        use_container_width=True)

            best_model_name = results_df.iloc[0]["Model"]
            best_r2 = results_df.iloc[0]["R² Score"]
            best_model = trained_models[best_model_name]

            st.success(f"🏆 **Best Model: {best_model_name}** | R² Score: {best_r2:.4f} | RMSE: ${results_df.iloc[0]['RMSE']:,.2f}")

            download_button(results_df, "regression_comparison.csv", "📥 Download Results")

            # Metric explanations
            with st.expander("📖 Understanding the Metrics"):
                st.markdown("""
                - **MAE (Mean Absolute Error)**: Average prediction error in dollars. Lower is better.
                - **RMSE (Root Mean Squared Error)**: Penalizes large errors more. Lower is better.
                - **R² Score**: % of variance explained by model (0-1). Higher is better.
                - **Explained Variance**: How much of the target variation is captured (%).
                """)

            # Visualizations
            st.markdown("### 📈 Performance Visualizations")

            # 1. Metrics comparison bar chart
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            x = np.arange(len(results_df))
            width = 0.25
            
            ax1.bar(x - width, results_df['MAE'], width, label='MAE', alpha=0.8)
            ax1.bar(x, results_df['RMSE'], width, label='RMSE', alpha=0.8)
            ax1_twin = ax1.twinx()
            ax1_twin.bar(x + width, results_df['R² Score'], width, label='R² Score', color='green', alpha=0.8)
            
            ax1.set_xlabel('Models', fontweight='bold')
            ax1.set_ylabel('Error Metrics (MAE, RMSE)', fontweight='bold')
            ax1_twin.set_ylabel('R² Score', fontweight='bold', color='green')
            ax1.set_title('Regression Model Comparison', fontweight='bold', fontsize=14)
            ax1.set_xticks(x)
            ax1.set_xticklabels(results_df['Model'])
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            ax1.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            st.pyplot(fig1)
            st.download_button("📥 PNG", fig_to_bytes(fig1, "png"), "regression_comparison.png", "image/png")
            plt.close()

            # 2. Predicted vs Actual for all models
            st.markdown("#### 🎯 Predicted vs Actual - All Models")
            
            fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, (name, model) in enumerate(trained_models.items()):
                y_pred = model.predict(X_test_s)
                r2 = r2_score(y_test, y_pred)
                
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                
                axes[idx].scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
                axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                axes[idx].set_xlabel('Actual Spend ($)', fontweight='bold')
                axes[idx].set_ylabel('Predicted Spend ($)', fontweight='bold')
                axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            st.download_button("📥 Download Chart", fig_to_bytes(fig2, "png"), "pred_vs_actual_all.png", "image/png")
            plt.close()

            # 3. Residual plots
            st.markdown("#### 📉 Residual Analysis")
            
            fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, (name, model) in enumerate(trained_models.items()):
                y_pred = model.predict(X_test_s)
                residuals = y_test - y_pred
                
                axes[idx].scatter(y_pred, residuals, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
                axes[idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
                axes[idx].set_xlabel('Predicted Spend ($)', fontweight='bold')
                axes[idx].set_ylabel('Residuals ($)', fontweight='bold')
                axes[idx].set_title(f'{name} - Residuals', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            st.download_button("📥 Download Residuals", fig_to_bytes(fig3, "png"), "residual_plots.png", "image/png")
            plt.close()

            # 4. Feature importance (coefficients)
            st.markdown("#### 🔍 Feature Coefficients Comparison")
            
            coef_df = pd.DataFrame({
                "Feature": X_cols,
                "Linear": trained_models["Linear Regression"].coef_,
                "Ridge": trained_models["Ridge Regression"].coef_,
                "Lasso": trained_models["Lasso Regression"].coef_
            })
            
            coef_df["Abs_Linear"] = np.abs(coef_df["Linear"])
            coef_df = coef_df.sort_values("Abs_Linear", ascending=False).head(15)
            
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            x_pos = np.arange(len(coef_df))
            width = 0.25
            
            ax4.barh(x_pos - width, coef_df["Linear"], width, label="Linear", alpha=0.8)
            ax4.barh(x_pos, coef_df["Ridge"], width, label="Ridge", alpha=0.8)
            ax4.barh(x_pos + width, coef_df["Lasso"], width, label="Lasso", alpha=0.8)
            
            ax4.set_yticks(x_pos)
            ax4.set_yticklabels(coef_df["Feature"])
            ax4.set_xlabel('Coefficient Value', fontweight='bold')
            ax4.set_title('Top 15 Feature Coefficients Comparison', fontweight='bold', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            st.pyplot(fig4)
            st.download_button("📥 Download Coefficients", fig_to_bytes(fig4, "png"), "feature_coefficients.png", "image/png")
            plt.close()

            # Feature coefficient table
            st.dataframe(coef_df[["Feature", "Linear", "Ridge", "Lasso"]].round(4), use_container_width=True)
            download_button(coef_df, "feature_coefficients.csv", "📥 Download Coefficient Table")

            # Predictions export
            st.markdown("### 💾 Export Predictions")
            
            pred_export = pd.DataFrame({
                "Customer_ID": df.loc[X_test.index, "Customer_ID"] if "Customer_ID" in df.columns else X_test.index,
                "Actual_Spend": y_test,
                "Linear_Pred": trained_models["Linear Regression"].predict(X_test_s),
                "Ridge_Pred": trained_models["Ridge Regression"].predict(X_test_s),
                "Lasso_Pred": trained_models["Lasso Regression"].predict(X_test_s)
            })
            
            pred_export["Best_Model_Pred"] = best_model.predict(X_test_s)
            pred_export["Error"] = pred_export["Actual_Spend"] - pred_export["Best_Model_Pred"]
            
            st.dataframe(pred_export.head(10).round(2), use_container_width=True)
            download_button(pred_export, "regression_predictions.csv", "📥 Download All Predictions")

        else:
            st.warning("⚠️ No features available for regression analysis")

with tabs[5]:
    st.markdown('<h2 class="sub-header">Dynamic Pricing Strategy</h2>', unsafe_allow_html=True)
    st.markdown("**Using best regression model to implement personalized pricing**")

    if "Monthly_Luxury_Spend" not in df.columns:
        st.error("❌ Cannot implement pricing without spend data")
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

            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train_s, y_train)

            X_all_s = scaler.transform(X)
            df["Predicted_Spend"] = gb.predict(X_all_s)

            st.markdown("### 💎 Pricing Configuration")

            with st.expander("⚙️ Adjust Pricing Parameters"):
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

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Multiplier", f"{df['Price_Multiplier'].mean():.3f}x")
            col2.metric("Price Adjustment", f"{df['Price_Adj_Pct'].mean():.1f}%")
            col3.metric("Revenue Lift", f"${(df['Personalized_Price'].sum() - df['Base_Price'].sum()):,.0f}")
            col4.metric("Customers Impacted", f"{len(df):,}")

            st.markdown("### 📊 Pricing Analysis Charts")

            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                scatter = ax1.scatter(df["Base_Price"], df["Personalized_Price"],
                                     c=df["Price_Multiplier"], cmap="RdYlGn", alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
                min_p = min(df["Base_Price"].min(), df["Personalized_Price"].min())
                max_p = max(df["Base_Price"].max(), df["Personalized_Price"].max())
                ax1.plot([min_p, max_p], [min_p, max_p], "k--", linewidth=2, label='No Change')
                ax1.set_xlabel("Base Price ($)", fontweight="bold")
                ax1.set_ylabel("Personalized Price ($)", fontweight="bold")
                ax1.set_title("Base vs Personalized Pricing", fontweight="bold")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax1, label="Multiplier")
                plt.tight_layout()
                st.pyplot(fig1)
                st.download_button("📥 PNG", fig_to_bytes(fig1, "png"), "pricing_scatter.png", "image/png", key="price1")
                plt.close()
            
            with col2:
                if "Brand_Loyalty_Score" in df.columns:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    bins = pd.cut(df["Brand_Loyalty_Score"], bins=[0, 3, 6, 8, 10], labels=["Low", "Medium", "High", "Very High"])
                    mult_loy = df.groupby(bins, observed=True)["Price_Multiplier"].mean()

                    if len(mult_loy) > 0:
                        ax2.bar(range(len(mult_loy)), mult_loy.values, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"], alpha=0.8)
                        ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label='Base (1.0x)')
                        ax2.set_xticks(range(len(mult_loy)))
                        ax2.set_xticklabels(mult_loy.index)
                        ax2.set_ylabel("Avg Price Multiplier", fontweight="bold")
                        ax2.set_xlabel("Loyalty Segment", fontweight="bold")
                        ax2.set_title("Pricing by Customer Loyalty", fontweight="bold")
                        ax2.legend()
                        ax2.grid(True, alpha=0.3, axis="y")
                        plt.tight_layout()
                        st.pyplot(fig2)
                        st.download_button("📥 PNG", fig_to_bytes(fig2, "png"), "loyalty_pricing.png", "image/png", key="price2")
                        plt.close()

            st.markdown("### 📋 Digital Price Chart (Sample)")

            chart_cols = ["Customer_ID", "Monthly_Luxury_Spend", "Predicted_Spend",
                         "Base_Price", "Price_Multiplier", "Personalized_Price", "Price_Adj_Pct"]
            chart_cols = [c for c in chart_cols if c in df.columns]

            price_chart = df[chart_cols].round(2)
            st.dataframe(price_chart.head(20), use_container_width=True)

            download_button(price_chart, "complete_price_chart.csv", "📥 Download Full Price Chart")
        else:
            st.warning("⚠️ No features available for pricing model")

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray; font-size: 14px;">🚀 Complete ML Analytics Dashboard | Built with Streamlit | Powered by scikit-learn</div>',
            unsafe_allow_html=True)
