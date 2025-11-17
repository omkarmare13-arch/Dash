"""
LuxeDash
A comprehensive machine learning analytics platform 
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import requests
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LuxeDash",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ============================================================================
# THEME CONFIGURATION
# ============================================================================
THEMES = {
    'dark': {
        'primary': '#1a1a2e',
        'secondary': '#16213e',
        'accent': '#d4af37',
        'text': '#f5f5f5',
        'text_secondary': '#b0b0b0',
        'background': '#0f0f1e',
        'card': '#1f1f3a',
        'border': '#2a2a4a'
    },
    'light': {
        'primary': '#2c3e50',
        'secondary': '#34495e',
        'accent': '#d4af37',
        'text': '#2c3e50',
        'text_secondary': '#7f8c8d',
        'background': '#f8f9fa',
        'card': '#ffffff',
        'border': '#e0e0e0'
    }
}

theme = THEMES[st.session_state.theme]

# ============================================================================
# PREMIUM CSS STYLING
# ============================================================================
st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    html, body, [class*="css"] {{
        font-size: 15px;
        color: {theme['text']};
    }}
    
    /* Main Container */
    .main {{
        background-color: {theme['background']};
        padding: 2rem 1rem;
    }}
    
    /* Sticky Header */
    .sticky-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%);
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-bottom: 2px solid {theme['accent']};
    }}
    
    .logo-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    .logo-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: {theme['accent']};
        margin: 0;
        letter-spacing: 1px;
    }}
    
    .logo-tagline {{
        font-size: 0.85rem;
        color: {theme['text_secondary']};
        margin-top: 0.25rem;
        font-weight: 300;
    }}
    
    .theme-toggle {{
        cursor: pointer;
        background: rgba(255,255,255,0.1);
        border: 1px solid {theme['accent']};
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: {theme['text']};
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .theme-toggle:hover {{
        background: {theme['accent']};
        color: {theme['primary']};
        transform: translateY(-2px);
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background: {theme['card']};
        border: 1px solid {theme['border']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .kpi-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        border-color: {theme['accent']};
    }}
    
    .kpi-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {theme['accent']};
        margin: 0.5rem 0;
    }}
    
    .kpi-label {{
        font-size: 0.9rem;
        color: {theme['text_secondary']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .kpi-footnote {{
        font-size: 0.75rem;
        color: {theme['text_secondary']};
        margin-top: 0.5rem;
        font-style: italic;
    }}
    
    /* Status Chip */
    .status-chip {{
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    .chip-synthetic {{
        background: rgba(52, 152, 219, 0.2);
        color: #3498db;
        border: 1px solid #3498db;
    }}
    
    .chip-upload {{
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }}
    
    .chip-github {{
        background: rgba(155, 89, 182, 0.2);
        color: #9b59b6;
        border: 1px solid #9b59b6;
    }}
    
    /* Section Headers */
    .section-header {{
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: {theme['accent']};
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {theme['accent']};
    }}
    
    .section-subheader {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {theme['text']};
        margin: 1.5rem 0 1rem 0;
    }}
    
    .section-description {{
        font-size: 0.95rem;
        color: {theme['text_secondary']};
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: {theme['card']};
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid {theme['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 3.5rem;
        background-color: transparent;
        border-radius: 8px;
        color: {theme['text_secondary']};
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(212, 175, 55, 0.1);
        color: {theme['accent']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {theme['accent']} 0%, #c4941f 100%);
        color: {theme['primary']} !important;
        font-weight: 700;
    }}
    
    /* Cards & Containers */
    .custom-card {{
        background: {theme['card']};
        border: 1px solid {theme['border']};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Metrics */
    div[data-testid="metric-container"] {{
        background: {theme['card']};
        border: 1px solid {theme['border']};
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    div[data-testid="metric-container"]:hover {{
        border-color: {theme['accent']};
        box-shadow: 0 4px 8px rgba(212, 175, 55, 0.2);
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {theme['accent']} 0%, #c4941f 100%);
        color: {theme['primary']};
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(212, 175, 55, 0.3);
    }}
    
    /* Download Buttons */
    .stDownloadButton>button {{
        background: {theme['card']};
        color: {theme['accent']};
        border: 1px solid {theme['accent']};
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.3s ease;
    }}
    
    .stDownloadButton>button:hover {{
        background: {theme['accent']};
        color: {theme['primary']};
        transform: translateY(-2px);
    }}
    
    /* Info Boxes */
    .stAlert {{
        border-radius: 8px;
        border-left: 4px solid {theme['accent']};
        background: {theme['card']};
    }}
    
    /* Dataframes */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid {theme['border']};
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {theme['card']};
        border-right: 2px solid {theme['border']};
    }}
    
    section[data-testid="stSidebar"] > div {{
        padding-top: 2rem;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {theme['card']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        font-weight: 600;
        color: {theme['text']};
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {theme['accent']};
        background-color: rgba(212, 175, 55, 0.05);
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {theme['accent']} 0%, #c4941f 100%);
    }}
    
    /* Tooltips */
    .tooltip-icon {{
        color: {theme['accent']};
        cursor: help;
        margin-left: 0.25rem;
    }}
    
    /* Badge */
    .badge {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin: 0.2rem;
    }}
    
    .badge-success {{
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }}
    
    .badge-info {{
        background: rgba(52, 152, 219, 0.2);
        color: #3498db;
        border: 1px solid #3498db;
    }}
    
    .badge-warning {{
        background: rgba(243, 156, 18, 0.2);
        color: #f39c12;
        border: 1px solid #f39c12;
    }}
    
    .badge-premium {{
        background: rgba(212, 175, 55, 0.2);
        color: {theme['accent']};
        border: 1px solid {theme['accent']};
    }}
    
    /* Chart Caption */
    .chart-caption {{
        font-size: 0.85rem;
        color: {theme['text_secondary']};
        font-style: italic;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: rgba(212, 175, 55, 0.05);
        border-left: 3px solid {theme['accent']};
        border-radius: 4px;
    }}
    
    /* Footer */
    .footer {{
        margin-top: 4rem;
        padding: 2rem;
        text-align: center;
        border-top: 2px solid {theme['border']};
        color: {theme['text_secondary']};
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section_header(title, description=""):
    """Display a styled section header with optional description"""
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<p class="section-description">{description}</p>', unsafe_allow_html=True)

def kpi_card(icon, value, label, footnote=""):
    """Create a KPI card component"""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {f'<div class="kpi-footnote">{footnote}</div>' if footnote else ''}
    </div>
    """
    return html

def status_chip(status_type, label):
    """Create a status chip"""
    chip_class = f"chip-{status_type.lower()}"
    return f'<span class="status-chip {chip_class}">{label}</span>'

def styled_metric_row(metrics_dict):
    """Display a row of styled metrics"""
    cols = st.columns(len(metrics_dict))
    for col, (label, value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(label, value)

def download_buttons(data, filename_base, label="Download"):
    """Create download buttons for multiple formats"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    if isinstance(data, pd.DataFrame):
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                f"📥 {label} CSV",
                data.to_csv(index=False),
                f"{filename_base}_{timestamp}.csv",
                "text/csv",
                use_container_width=True
            )
    elif isinstance(data, bytes):
        st.download_button(
            f"📥 {label}",
            data,
            f"{filename_base}_{timestamp}.png",
            "image/png",
            use_container_width=True
        )

def fig_to_bytes(fig, fmt="png"):
    """Convert matplotlib figure to bytes"""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight", facecolor='white')
    buf.seek(0)
    return buf.getvalue()

def apply_plot_style():
    """Apply consistent plot styling"""
    plt.style.use('seaborn-v0_8-darkgrid' if st.session_state.theme == 'dark' else 'seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

def chart_caption(text):
    """Display a styled chart caption"""
    st.markdown(f'<div class="chart-caption">💡 <strong>How to read:</strong> {text}</div>', unsafe_allow_html=True)

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_data
def generate_synthetic_data(n=1000, seed=42):
    """Generate synthetic luxury customer data"""
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
    """Load data from uploaded CSV"""
    try:
        return pd.read_csv(file), None
    except Exception as e:
        return None, str(e)

def load_from_github(url):
    """Load data from GitHub raw URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text)), None
    except Exception as e:
        return None, str(e)

def engineer_features(df):
    """Engineer additional features from raw data"""
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
    """Get list of numeric feature columns"""
    base = ["Age", "Annual_Income", "Monthly_Luxury_Spend", "Luxury_Purchases_Per_Year",
            "Brand_Loyalty_Score", "Digital_Shopping_Comfort", "Interest_1_2hr_Service",
            "WTP_1hr_Delivery", "City_Tier"]
    return [c for c in base if c in df.columns]

# ============================================================================
# STICKY HEADER WITH THEME TOGGLE
# ============================================================================

header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown("""
    <div class="sticky-header">
        <div class="logo-container">
            <div>
                <div class="logo-title">💎 LUXE ANALYTICS</div>
                <div class="logo-tagline">Premium ML Insights for Luxury Delivery</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if st.button("🌓 Theme", key="theme_toggle", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

# ============================================================================
# SIDEBAR - DATA CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("## 🎛️ Data Controls")
    st.markdown("---")
    
    data_source = st.radio(
        "Data Source",
        ["Synthetic Data", "Upload CSV", "GitHub URL"],
        help="Choose how to load your customer data"
    )
    
    df = None
    data_loaded = False
    data_source_type = "synthetic"
    
    if data_source == "Synthetic Data":
        st.markdown("### 📊 Generation Settings")
        n = st.slider("Sample Size", 100, 2000, 1000, 100, 
                     help="Number of customer records to generate")
        seed = st.number_input("Random Seed", 0, 1000, 42, 
                              help="For reproducible results")
        
        if st.button("🎲 Generate Data", type="primary", use_container_width=True) or "main_df" not in st.session_state:
            with st.spinner("Generating synthetic data..."):
                df = generate_synthetic_data(n, seed)
                df = engineer_features(df)
                st.session_state["main_df"] = df
                st.session_state.last_refresh = datetime.now()
                data_loaded = True
                st.success(f"✓ {len(df):,} records")
                st.toast("Data generated successfully!")
        else:
            df = st.session_state.get("main_df")
            if df is not None:
                data_loaded = True
                st.success(f"✓ {len(df):,} records")
        data_source_type = "synthetic"
    
    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("📁 Upload File", type=["csv"], 
                                   help="Upload a CSV file with customer data")
        if uploaded:
            with st.spinner("Loading file..."):
                df, error = load_from_upload(uploaded)
                if df is not None:
                    df = engineer_features(df)
                    data_loaded = True
                    st.success(f"✓ {len(df):,} records")
                    st.toast("File uploaded successfully!")
                else:
                    st.error(f"Error: {error}")
        data_source_type = "upload"
    
    elif data_source == "GitHub URL":
        url = st.text_input("🔗 Raw CSV URL", placeholder="https://raw.githubusercontent.com/...",
                           help="Paste the raw GitHub URL to your CSV file")
        if st.button("📥 Load Data", type="primary", use_container_width=True):
            if url:
                with st.spinner("Fetching from GitHub..."):
                    df, error = load_from_github(url)
                    if df is not None:
                        df = engineer_features(df)
                        data_loaded = True
                        st.success(f"✓ {len(df):,} records")
                        st.toast("Data loaded from GitHub!")
                    else:
                        st.error(f"Error: {error}")
        data_source_type = "github"
    
    if data_loaded and df is not None:
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")
        
        # Column presence checklist
        with st.expander("📋 Column Check"):
            required_cols = ["Customer_ID", "Annual_Income", "Monthly_Luxury_Spend", "Age"]
            for col in required_cols:
                if col in df.columns:
                    st.markdown(f"✅ {col}")
                else:
                    st.markdown(f"❌ {col}")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not data_loaded or df is None:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">👋 Welcome to Luxe Analytics</h1>
        <p style="font-size: 1.2rem; color: #b0b0b0; margin-bottom: 2rem;">
            Your premium machine learning platform for luxury delivery insights
        </p>
        <p style="font-size: 1rem; color: #888;">
            👈 Get started by selecting a data source from the sidebar
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Classification")
        st.write("7 algorithms with comprehensive performance analysis")
    with col2:
        st.markdown("### 🔍 Clustering")
        st.write("K-Means & K-Modes with business insights")
    with col3:
        st.markdown("### 📈 Regression")
        st.write("Linear, Ridge & Lasso model comparison")
    
    st.stop()

# ============================================================================
# TOP SUMMARY - KPI CARDS
# ============================================================================

section_header("📊 Executive Summary")

# Status chip
chip_html = status_chip(data_source_type, 
                       {"synthetic": "Synthetic Data", 
                        "upload": "Uploaded File", 
                        "github": "GitHub Source"}[data_source_type])
st.markdown(chip_html, unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_html = kpi_card(
        "👥",
        f"{len(df):,}",
        "Total Customers",
        f"Last updated: {st.session_state.last_refresh.strftime('%H:%M')}"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

with col2:
    avg_income = df['Annual_Income'].mean() if 'Annual_Income' in df.columns else 0
    kpi_html = kpi_card(
        "💰",
        f"${avg_income/1000:.0f}K",
        "Avg Annual Income",
        "Per customer baseline"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

with col3:
    avg_spend = df['Monthly_Luxury_Spend'].mean() if 'Monthly_Luxury_Spend' in df.columns else 0
    kpi_html = kpi_card(
        "💎",
        f"${avg_spend:,.0f}",
        "Avg Monthly Spend",
        "Luxury purchases only"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

with col4:
    adoption = (df['Likely_to_Use'].sum()/len(df)*100) if 'Likely_to_Use' in df.columns else 0
    kpi_html = kpi_card(
        "📈",
        f"{adoption:.1f}%",
        "Service Adoption",
        "Predicted likelihood"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tabs = st.tabs([
    "📊 Overview",
    "🎯 Classification",
    "🔍 Clustering",
    "🛒 Association Rules",
    "📈 Regression",
    "💰 Pricing"
])

# ============================================================================
# TAB 0: OVERVIEW
# ============================================================================

with tabs[0]:
    section_header("Customer Persona Analysis", 
                  "Segment your customers into distinct behavioral groups using K-Means clustering. Identify high-value segments and tailor your marketing strategies.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        k = st.slider("Number of Personas (K)", 2, 8, 4, 
                     help="Choose how many customer segments to create. 3-5 personas typically work best.")
    with col2:
        st.info("💡 **Tip:** Start with 4 personas and adjust based on your business needs.")
    
    numeric = get_numeric_features(df)
    
    if len(numeric) >= 3:
        X = df[numeric].fillna(df[numeric].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with st.spinner("Creating customer personas..."):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            df["Persona"] = kmeans.fit_predict(X_scaled)
            
            # Assign meaningful names based on spending
            spend_order = df.groupby("Persona")["Monthly_Luxury_Spend"].mean().sort_values()
            names = {}
            name_options = ["💼 Budget-Conscious", "🎯 Aspirational", "💎 Affluent", "👑 Ultra-Premium"]
            
            for i, (pid, _) in enumerate(spend_order.items()):
                if i < len(name_options):
                    names[pid] = name_options[i]
                else:
                    names[pid] = f"Segment {i+1}"
            
            df["Persona_Name"] = df["Persona"].map(names)
        
        st.success(f"✓ Successfully created {k} customer personas")
        
        # Persona profiles
        st.markdown("### 📋 Persona Profiles")
        
        profile_cols = numeric + ["n_categories_interested", "n_features_valued"]
        profile_cols = [c for c in profile_cols if c in df.columns]
        
        profile = df.groupby("Persona_Name")[profile_cols].mean().round(2)
        profile["Count"] = df.groupby("Persona_Name").size()
        profile["Percentage"] = (profile["Count"] / len(df) * 100).round(1)
        
        # Reorder for better display
        display_cols = ["Count", "Percentage"] + [c for c in profile.columns if c not in ["Count", "Percentage"]]
        profile = profile[display_cols]
        
        st.dataframe(
            profile.style.background_gradient(cmap='YlOrRd', subset=['Monthly_Luxury_Spend'])
                        .background_gradient(cmap='YlGn', subset=['Brand_Loyalty_Score'])
                        .bar(subset=['Percentage'], color='#d4af37'),
            use_container_width=True
        )
        
        chart_caption("Each row represents a customer persona. Darker colors indicate higher values. The percentage shows the proportion of customers in each segment.")
        
        # Download section
        st.markdown("### 💾 Downloads")
        col1, col2 = st.columns(2)
        with col1:
            download_buttons(
                df[["Customer_ID", "Persona_Name"] + numeric],
                "persona_labels",
                "Customer Labels"
            )
        with col2:
            download_buttons(profile, "persona_profiles", "Persona Profiles")
    
    else:
        st.warning("⚠️ Need at least 3 numeric features for persona analysis")

# ============================================================================
# TAB 1: CLASSIFICATION
# ============================================================================

with tabs[1]:
    section_header("Classification Model Comparison",
                  "Predict service adoption using 7 different algorithms. Compare performance metrics and choose the best model for your business.")
    
    # Model selection
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            selected_models = st.multiselect(
                "Models to Train",
                ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "Naive Bayes", "KNN", "Gradient Boosting"],
                default=["Logistic Regression", "Random Forest", "Gradient Boosting"],
                help="Select which models to train and compare"
            )
        with col2:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 
                                 help="Percentage of data to use for testing") / 100
    
    if not selected_models:
        st.warning("Please select at least one model to train")
        st.stop()
    
    numeric = get_numeric_features(df)
    cats = [c for c in df.columns if "Interested_" in c]
    feats = [c for c in df.columns if "Feat_" in c]
    
    X_cols = numeric + cats + feats + ["n_categories_interested", "n_features_valued"]
    X_cols = [c for c in X_cols if c in df.columns and c != "Monthly_Luxury_Spend"]
    
    if "Likely_to_Use" in df.columns and len(X_cols) > 0:
        X = df[X_cols].fillna(df[X_cols].median())
        y = df["Likely_to_Use"]
        
        if len(X) > 20 and y.nunique() == 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Model definitions
            all_models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
                "SVM": SVC(kernel='rbf', probability=True, random_state=42),
                "Naive Bayes": GaussianNB(),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            models = {k: v for k, v in all_models.items() if k in selected_models}
            
            if st.button("🚀 Train Models", type="primary"):
                progress_bar = st.progress(0)
                status = st.empty()
                
                results = []
                
                for idx, (name, model) in enumerate(models.items()):
                    status.text(f"Training {name}... ({idx+1}/{len(models)})")
                    
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                    y_proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    results.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, zero_division=0),
                        "Recall": recall_score(y_test, y_pred, zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else accuracy_score(y_test, y_pred)
                    })
                    
                    progress_bar.progress((idx + 1) / len(models))
                
                status.empty()
                progress_bar.empty()
                
                st.session_state['classification_results'] = results
                st.session_state['trained_models'] = models
                st.session_state['scaler'] = scaler
                st.session_state['test_data'] = (X_test_s, y_test)
                
                st.success("✓ All models trained successfully!")
                st.toast("Models trained successfully!")
            
            # Display results if available
            if 'classification_results' in st.session_state:
                results_df = pd.DataFrame(st.session_state['classification_results'])
                results_df = results_df.sort_values("ROC-AUC", ascending=False)
                
                st.markdown("### 📊 Performance Comparison")
                
                st.dataframe(
                    results_df.style.highlight_max(
                        axis=0, 
                        subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                        color='#d4af3750'
                    ).format({
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "ROC-AUC": "{:.4f}"
                    }),
                    use_container_width=True
                )
                
                chart_caption("Higher values are better. The best model in each metric is highlighted.")
                
                # Best model highlight
                best_idx = 0
                best_name = results_df.iloc[best_idx]["Model"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Best Model", best_name)
                with col2:
                    st.metric("ROC-AUC Score", f"{results_df.iloc[best_idx]['ROC-AUC']:.4f}")
                with col3:
                    st.metric("Accuracy", f"{results_df.iloc[best_idx]['Accuracy']:.4f}")
                
                # ROC Curve
                st.markdown("### 📉 ROC Curves")
                
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(10, 7))
                
                models = st.session_state['trained_models']
                X_test_s, y_test = st.session_state['test_data']
                
                for name, model in models.items():
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_s)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        auc_score = roc_auc_score(y_test, y_proba)
                        ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={auc_score:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
                ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
                ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
                ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=13, pad=15)
                ax.legend(frameon=True, shadow=True, fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                chart_caption("The closer the curve is to the top-left corner, the better the model. AUC (Area Under Curve) summarizes overall performance.")
                
                col1, col2 = st.columns(2)
                with col1:
                    download_buttons(fig_to_bytes(fig), "roc_curves", "ROC Chart PNG")
                with col2:
                    download_buttons(results_df, "classification_results", "Results")
                
                plt.close()
        else:
            st.warning("⚠️ Insufficient data or target not binary")
    else:
        st.warning("⚠️ Target 'Likely_to_Use' missing or no features available")

# ============================================================================
# TAB 2: CLUSTERING
# ============================================================================

with tabs[2]:
    section_header("Customer Clustering Analysis",
                  "Discover hidden customer segments using unsupervised learning. Analyze patterns and characteristics of each group.")
    
    st.markdown("### 🔵 K-Means Clustering")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        k_clust = st.slider("Number of Clusters (K)", 2, 10, 5, 
                           help="Number of customer groups to create. Use the elbow method to find optimal K.")
    with col2:
        show_elbow = st.checkbox("Show Elbow Analysis", value=True)
    
    numeric = get_numeric_features(df)
    
    if len(numeric) >= 3:
        X = df[numeric].fillna(df[numeric].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if show_elbow:
            with st.spinner("Analyzing optimal K..."):
                K_range = range(2, min(11, len(df)//10))
                inertias = []
                silhouette_scores = []
                
                for k_test in K_range:
                    kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init=20)
                    kmeans_test.fit(X_scaled)
                    inertias.append(kmeans_test.inertia_)
                    silhouette_scores.append(silhouette_score(X_scaled, kmeans_test.labels_))
            
            col1, col2 = st.columns(2)
            
            with col1:
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(K_range, inertias, 'o-', linewidth=2.5, markersize=10, color='#d4af37')
                ax.axvline(x=k_clust, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Selected K={k_clust}')
                ax.set_xlabel('Number of Clusters (K)', fontweight='bold')
                ax.set_ylabel('Inertia', fontweight='bold')
                ax.set_title('Elbow Method', fontweight='bold', pad=15)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(K_range, silhouette_scores, 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
                ax.axvline(x=k_clust, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Selected K={k_clust}')
                ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Good threshold')
                ax.set_xlabel('Number of Clusters (K)', fontweight='bold')
                ax.set_ylabel('Silhouette Score', fontweight='bold')
                ax.set_title('Silhouette Analysis', fontweight='bold', pad=15)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # Perform clustering
        with st.spinner(f"Clustering with K={k_clust}..."):
            kmeans = KMeans(n_clusters=k_clust, random_state=42, n_init=20)
            df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, df["KMeans_Cluster"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters Created", k_clust)
        with col2:
            st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
        with col3:
            quality = "Excellent ✨" if silhouette_avg > 0.7 else "Good ✓" if silhouette_avg > 0.5 else "Fair"
            st.metric("Quality", quality)
        
        st.markdown("### 📊 Cluster Profiles")
        
        profile = df.groupby("KMeans_Cluster")[numeric].mean().round(2)
        profile["Count"] = df.groupby("KMeans_Cluster").size()
        profile["Percentage"] = (profile["Count"] / len(df) * 100).round(1)
        
        st.dataframe(
            profile.style.background_gradient(cmap='YlOrRd', subset=numeric)
                        .bar(subset=['Percentage'], color='#d4af37'),
            use_container_width=True
        )
        
        chart_caption("Each row represents a cluster. Higher values are shown in darker colors.")
        
        # Cluster insights
        st.markdown("### 💡 Cluster Insights")
        
        for cluster_id in range(k_clust):
            cluster_data = df[df["KMeans_Cluster"] == cluster_id]
            
            with st.expander(f"**Cluster {cluster_id}** ({len(cluster_data)} customers, {len(cluster_data)/len(df)*100:.1f}%)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Income", f"${cluster_data['Annual_Income'].mean():,.0f}" if "Annual_Income" in cluster_data.columns else "N/A")
                with col2:
                    st.metric("Monthly Spend", f"${cluster_data['Monthly_Luxury_Spend'].mean():,.0f}" if "Monthly_Luxury_Spend" in cluster_data.columns else "N/A")
                with col3:
                    st.metric("Loyalty", f"{cluster_data['Brand_Loyalty_Score'].mean():.1f}/10" if "Brand_Loyalty_Score" in cluster_data.columns else "N/A")
                
                # Top differentiators
                cluster_means = cluster_data[numeric].mean()
                overall_means = df[numeric].mean()
                diff = ((cluster_means - overall_means) / overall_means * 100).abs().sort_values(ascending=False).head(3)
                
                st.markdown("**🎯 Top 3 Differentiators:**")
                for feat, pct in diff.items():
                    direction = "↑" if cluster_means[feat] > overall_means[feat] else "↓"
                    st.markdown(f'<span class="badge badge-info">{direction} {feat}: {pct:.1f}% diff</span>', unsafe_allow_html=True)
        
        # Downloads
        st.markdown("### 💾 Downloads")
        col1, col2 = st.columns(2)
        with col1:
            download_buttons(profile, "cluster_profiles", "Cluster Profiles")
        with col2:
            cluster_export = df[["Customer_ID"] + numeric + ["KMeans_Cluster"]] if "Customer_ID" in df.columns else df[numeric + ["KMeans_Cluster"]]
            download_buttons(cluster_export, "cluster_labels", "Customer Labels")
    
    else:
        st.warning("⚠️ Need at least 3 numeric features for clustering")
    
    # K-Modes section
    st.markdown("---")
    st.markdown("### 🟠 K-Modes (Categorical Clustering)")
    
    if KMODES_AVAILABLE:
        cats = [c for c in df.columns if ("Interested_" in c or "Feat_" in c) and df[c].dtype in ["int64", "object"]]
        
        if len(cats) >= 3:
            k_modes = st.slider("Number of Modes (K)", 2, 8, 4, 
                               help="Number of categorical clusters")
            
            X_cat = df[cats].fillna(0).astype(int)
            
            try:
                with st.spinner(f"Performing K-Modes clustering..."):
                    km = KModes(n_clusters=k_modes, random_state=42, n_init=10)
                    df["KModes_Cluster"] = km.fit_predict(X_cat)
                
                st.success(f"✓ K-Modes completed with {k_modes} clusters")
                
                # Mode profiles - FIXED
                modes = pd.DataFrame(km.cluster_centroids_, columns=cats).T
                modes.columns = [f"Mode {i+1}" for i in range(k_modes)]
                
                # Add cluster sizes as a row
                sizes = df.groupby("KModes_Cluster").size()
                size_row = pd.DataFrame([sizes.values], columns=modes.columns, index=["Cluster Size"])
                modes = pd.concat([modes, size_row])
                
                st.dataframe(modes, use_container_width=True)
                chart_caption("Each column is a mode (cluster). Values show the most common category for each feature.")
                
                download_buttons(modes, "kmodes_profiles", "Mode Profiles")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.info("ℹ️ Need at least 3 categorical columns for K-Modes")
    else:
        st.info("ℹ️ K-Modes library not available. Install: `pip install kmodes`")

# ============================================================================
# TAB 3: ASSOCIATION RULES
# ============================================================================

with tabs[3]:
    section_header("Association Rule Mining",
                  "Discover which products and features customers tend to like together. Use these insights for bundling and cross-selling strategies.")
    
    if not MLXTEND_AVAILABLE:
        st.error("❌ mlxtend library required. Install: `pip install mlxtend`")
    else:
        basket_cols = [c for c in df.columns if ("Interested_" in c or "Feat_" in c) and df[c].dtype in ["int64", "float64"]]
        
        if len(basket_cols) >= 2:
            with st.expander("⚙️ Mining Configuration", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_sup = st.slider(
                        "Min Support (%)", 1, 50, 10,
                        help="Minimum frequency (%) that item combinations must appear. Lower values find more rules but may be less meaningful."
                    ) / 100
                
                with col2:
                    min_conf = st.slider(
                        "Min Confidence (%)", 10, 90, 50,
                        help="Minimum reliability (%) of the rule. Higher values mean more trustworthy rules."
                    ) / 100
                
                with col3:
                    top_n = st.slider(
                        "Top N Rules", 5, 50, 15,
                        help="Number of top rules to display, ranked by lift."
                    )
            
            if st.button("⛏️ Mine Association Rules", type="primary"):
                with st.spinner("Mining patterns..."):
                    try:
                        basket = df[basket_cols].astype(bool)
                        freq = apriori(basket, min_support=min_sup, use_colnames=True)
                        
                        if len(freq) > 0:
                            st.success(f"✓ Found {len(freq):,} frequent itemsets")
                            
                            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                            
                            if len(rules) > 0:
                                rules["antecedent_str"] = rules["antecedents"].apply(
                                    lambda x: ", ".join(list(x)).replace("Interested_", "").replace("Feat_", "").replace("_", " ")
                                )
                                rules["consequent_str"] = rules["consequents"].apply(
                                    lambda x: ", ".join(list(x)).replace("Interested_", "").replace("Feat_", "").replace("_", " ")
                                )
                                
                                rules = rules.sort_values("lift", ascending=False)
                                
                                display = rules[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].head(top_n).copy()
                                display.columns = ["If Customer Likes", "Then Likely Likes", "Support", "Confidence", "Lift"]
                                
                                st.session_state['association_rules'] = display
                                st.session_state['all_rules'] = rules
                                
                                st.toast("Rules generated successfully!")
                            else:
                                st.warning(f"⚠️ No rules found. Try lowering confidence to {max(10, min_conf*100-10):.0f}%")
                        else:
                            st.warning(f"⚠️ No frequent itemsets. Try lowering support to {max(5, min_sup*100-5):.0f}%")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Display results
            if 'association_rules' in st.session_state:
                display = st.session_state['association_rules']
                
                st.markdown(f"### 📋 Top {len(display)} Association Rules")
                
                st.dataframe(
                    display.style.background_gradient(subset=["Lift"], cmap='RdYlGn')
                                .bar(subset=['Confidence'], color='#d4af37')
                                .format({
                                    "Support": "{:.4f}",
                                    "Confidence": "{:.4f}",
                                    "Lift": "{:.2f}"
                                }),
                    use_container_width=True
                )
                
                # Metrics explanation
                with st.expander("📖 Understanding the Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Support** 📊")
                        st.write("How frequently items appear together")
                        st.write("Higher = more common")
                    with col2:
                        st.markdown("**Confidence** 🎯")
                        st.write("Likelihood of consequent given antecedent")
                        st.write("Higher = more reliable")
                    with col3:
                        st.markdown("**Lift** 🚀")
                        st.write("Strength of association")
                        st.write(">1 = positive correlation")
                
                # Key insight
                top_rule = display.iloc[0]
                st.markdown("### 💡 Strongest Association")
                st.success(f"""
                **{top_rule['If Customer Likes']}** → **{top_rule['Then Likely Likes']}**
                
                Customers interested in {top_rule['If Customer Likes']} are **{top_rule['Lift']:.1f}x more likely** to be interested in {top_rule['Then Likely Likes']}.
                
                💼 **Recommendation:** Create bundle offers or cross-sell strategies for these items.
                """)
                
                # Chart
                st.markdown("### 📊 Rule Visualization")
                
                rules = st.session_state['all_rules']
                
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(10, 6))
                
                scatter = ax.scatter(
                    rules['support'], rules['confidence'],
                    c=rules['lift'], s=150, cmap='RdYlGn', alpha=0.7,
                    edgecolors='black', linewidths=1.5
                )
                
                ax.set_xlabel('Support', fontweight='bold', fontsize=11)
                ax.set_ylabel('Confidence', fontweight='bold', fontsize=11)
                ax.set_title('Association Rules: Support vs Confidence', fontweight='bold', fontsize=13, pad=15)
                ax.grid(True, alpha=0.3)
                
                cbar = plt.colorbar(scatter, ax=ax, label='Lift')
                cbar.set_label('Lift', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                chart_caption("Each point is a rule. Position shows support/confidence; color shows lift. Top-right points with green color are the strongest rules.")
                
                col1, col2 = st.columns(2)
                with col1:
                    download_buttons(fig_to_bytes(fig), "association_viz", "Chart")
                with col2:
                    download_buttons(display, "association_rules", "Rules")
                
                plt.close()
        
        else:
            st.warning("⚠️ Need at least 2 categorical columns for association rules")

# ============================================================================
# TAB 4: REGRESSION
# ============================================================================

with tabs[4]:
    section_header("Regression Model Comparison",
                  "Predict customer spending using Linear, Ridge, and Lasso regression. Compare model performance and understand feature importance.")
    
    if "Monthly_Luxury_Spend" not in df.columns:
        st.error("❌ Target variable 'Monthly_Luxury_Spend' not found")
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
            
            with st.expander("⚙️ Regularization Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    ridge_alpha = st.slider(
                        "Ridge Alpha (L2)", 0.01, 10.0, 1.0, 0.1,
                        help="L2 regularization strength. Higher = more penalty on large coefficients."
                    )
                with col2:
                    lasso_alpha = st.slider(
                        "Lasso Alpha (L1)", 0.01, 10.0, 1.0, 0.1,
                        help="L1 regularization strength. Can zero out features (feature selection)."
                    )
            
            if st.button("🚀 Train Regression Models", type="primary"):
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(alpha=ridge_alpha),
                    "Lasso Regression": Lasso(alpha=lasso_alpha, max_iter=10000)
                }
                
                with st.spinner("Training models..."):
                    results = []
                    trained_models = {}
                    
                    for name, model in models.items():
                        model.fit(X_train_s, y_train)
                        trained_models[name] = model
                        
                        y_pred = model.predict(X_test_s)
                        
                        results.append({
                            "Model": name,
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                            "R² Score": r2_score(y_test, y_pred),
                            "Explained Variance (%)": r2_score(y_test, y_pred) * 100
                        })
                    
                    results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
                    
                    st.session_state['regression_results'] = results_df
                    st.session_state['regression_models'] = trained_models
                    st.session_state['regression_test'] = (X_test_s, y_test)
                    st.session_state['regression_scaler'] = scaler
                    st.session_state['regression_features'] = X_cols
                    
                    st.success("✓ Models trained successfully!")
                    st.toast("Regression models ready!")
            
            # Display results
            if 'regression_results' in st.session_state:
                results_df = st.session_state['regression_results']
                
                st.markdown("### 📊 Model Performance")
                
                st.dataframe(
                    results_df.style.highlight_max(subset=["R² Score"], color='#90EE90')
                                  .highlight_min(subset=["MAE", "RMSE"], color='#90EE90')
                                  .format({
                                      "MAE": "${:,.2f}",
                                      "RMSE": "${:,.2f}",
                                      "R² Score": "{:.4f}",
                                      "Explained Variance (%)": "{:.2f}%"
                                  }),
                    use_container_width=True
                )
                
                best_model_name = results_df.iloc[0]["Model"]
                best_r2 = results_df.iloc[0]["R² Score"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Best Model", best_model_name)
                with col2:
                    st.metric("R² Score", f"{best_r2:.4f}")
                with col3:
                    st.metric("RMSE", f"${results_df.iloc[0]['RMSE']:,.2f}")
                
                # Visualization
                st.markdown("### 📈 Model Comparison")
                
                trained_models = st.session_state['regression_models']
                X_test_s, y_test = st.session_state['regression_test']
                
                apply_plot_style()
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                for idx, (name, model) in enumerate(trained_models.items()):
                    y_pred = model.predict(X_test_s)
                    r2 = r2_score(y_test, y_pred)
                    
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    
                    axes[idx].scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
                    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect')
                    axes[idx].set_xlabel('Actual Spend ($)', fontweight='bold')
                    axes[idx].set_ylabel('Predicted Spend ($)', fontweight='bold')
                    axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontweight='bold', pad=10)
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                chart_caption("Points close to the red diagonal line indicate accurate predictions. R² closer to 1.0 is better.")
                
                col1, col2 = st.columns(2)
                with col1:
                    download_buttons(fig_to_bytes(fig), "regression_comparison", "Chart")
                with col2:
                    download_buttons(results_df, "regression_results", "Results")
                
                plt.close()
        
        else:
            st.warning("⚠️ No features available for regression")

# ============================================================================
# TAB 5: PRICING
# ============================================================================

with tabs[5]:
    section_header("Dynamic Pricing Strategy",
                  "Implement AI-powered personalized pricing based on customer spending capacity and loyalty. Optimize revenue while maintaining fairness.")
    
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
            
            with st.spinner("Training pricing model..."):
                gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb.fit(X_train_s, y_train)
                
                X_all_s = scaler.transform(X)
                df["Predicted_Spend"] = gb.predict(X_all_s)
                
                r2 = r2_score(y_test, gb.predict(X_test_s))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🤖 Model", "Gradient Boosting")
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            with col3:
                quality = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Fair"
                st.metric("Quality", quality)
            
            st.markdown("---")
            
            with st.expander("⚙️ Pricing Configuration", expanded=True):
                st.info("💡 Adjust these parameters to control how pricing adapts to different customer segments")
                
                col1, col2 = st.columns(2)
                with col1:
                    cap_w = st.slider(
                        "Spending Capacity Weight", 0.0, 1.0, 0.30, 0.05,
                        help="How much predicted spending influences price (higher = more impact)"
                    )
                    loy_w = st.slider(
                        "Loyalty Weight", 0.0, 1.0, 0.25, 0.05,
                        help="Reward loyal customers with better pricing (higher = more discount)"
                    )
                with col2:
                    min_mult = st.slider(
                        "Min Multiplier", 0.5, 1.0, 0.70, 0.05,
                        help="Maximum discount (0.7 = up to 30% off)"
                    )
                    max_mult = st.slider(
                        "Max Multiplier", 1.0, 2.0, 1.25, 0.05,
                        help="Maximum premium (1.25 = up to 25% premium)"
                    )
            
            # Calculate pricing
            df["Capacity_Score"] = (df["Predicted_Spend"] / df["Predicted_Spend"].median()).clip(0.5, 2.0)
            df["Loyalty_Mult"] = df["Brand_Loyalty_Score"] / 10.0 if "Brand_Loyalty_Score" in df.columns else 0.5
            df["Price_Multiplier"] = (
                (df["Capacity_Score"] - 1.0) * cap_w +
                (df["Loyalty_Mult"] - 0.5) * loy_w +
                1.0
            ).clip(min_mult, max_mult)
            
            df["Personalized_Price"] = df["Base_Price"] * df["Price_Multiplier"]
            df["Price_Adj_Pct"] = ((df["Personalized_Price"] - df["Base_Price"]) / df["Base_Price"]) * 100
            
            # Impact summary
            st.markdown("### 💎 Pricing Impact")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Multiplier", f"{df['Price_Multiplier'].mean():.3f}x")
            
            with col2:
                avg_adj = df['Price_Adj_Pct'].mean()
                st.metric("Avg Adjustment", f"{avg_adj:+.1f}%", delta=f"{avg_adj:.1f}%")
            
            with col3:
                revenue_lift = df['Personalized_Price'].sum() - df['Base_Price'].sum()
                st.metric("Revenue Lift", f"${revenue_lift:,.0f}", 
                         delta=f"+{(revenue_lift/df['Base_Price'].sum()*100):.1f}%")
            
            with col4:
                premium_customers = len(df[df['Price_Multiplier'] > 1.0])
                st.metric("Premium Pricing", f"{premium_customers:,}", 
                         help=f"{premium_customers/len(df)*100:.1f}% of customers")
            
            # Visualization
            st.markdown("### 📊 Pricing Analysis")
            
            apply_plot_style()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Price comparison scatter
            scatter = ax1.scatter(
                df["Base_Price"], df["Personalized_Price"],
                c=df["Price_Multiplier"], cmap="RdYlGn", alpha=0.7, s=80,
                edgecolors='black', linewidths=1
            )
            
            min_p = min(df["Base_Price"].min(), df["Personalized_Price"].min())
            max_p = max(df["Base_Price"].max(), df["Personalized_Price"].max())
            ax1.plot([min_p, max_p], [min_p, max_p], "k--", linewidth=2.5, label='No Change', alpha=0.7)
            ax1.set_xlabel("Base Price ($)", fontweight="bold")
            ax1.set_ylabel("Personalized Price ($)", fontweight="bold")
            ax1.set_title("Base vs Personalized Pricing", fontweight="bold", pad=15)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label="Multiplier")
            
            # Distribution
            ax2.hist(df["Price_Multiplier"], bins=30, color='#d4af37', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2.5, label='Base (1.0x)')
            ax2.axvline(x=df["Price_Multiplier"].mean(), color='green', linestyle='-', linewidth=2.5, 
                       label=f'Avg ({df["Price_Multiplier"].mean():.3f}x)')
            ax2.set_xlabel('Price Multiplier', fontweight='bold')
            ax2.set_ylabel('Number of Customers', fontweight='bold')
            ax2.set_title('Multiplier Distribution', fontweight='bold', pad=15)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            chart_caption("Left: Points above the diagonal receive premium pricing; below receive discounts. Right: Distribution shows how many customers receive each multiplier.")
            
            download_buttons(fig_to_bytes(fig), "pricing_analysis", "Chart")
            plt.close()
            
            # Digital price chart
            st.markdown("### 📋 Digital Price Chart")
            
            chart_cols = ["Customer_ID", "Monthly_Luxury_Spend", "Predicted_Spend",
                         "Base_Price", "Price_Multiplier", "Personalized_Price", "Price_Adj_Pct"]
            chart_cols = [c for c in chart_cols if c in df.columns]
            
            price_chart = df[chart_cols].round(2)
            
            st.dataframe(
                price_chart.head(20).style.background_gradient(subset=['Price_Multiplier'], cmap='RdYlGn')
                                          .bar(subset=['Price_Adj_Pct'], align='zero', color=['#d65f5f', '#5fba7d'])
                                          .format({
                                              'Monthly_Luxury_Spend': '${:,.2f}',
                                              'Predicted_Spend': '${:,.2f}',
                                              'Base_Price': '${:,.2f}',
                                              'Price_Multiplier': '{:.3f}x',
                                              'Personalized_Price': '${:,.2f}',
                                              'Price_Adj_Pct': '{:+.2f}%'
                                          }),
                use_container_width=True
            )
            
            download_buttons(price_chart, "digital_price_chart", "Full Price Chart")
            
            # Implementation guide
            with st.expander("📚 Implementation Guide"):
                st.markdown("""
                ### 🎯 Best Practices
                
                1. **Start Small**: Test with 10-20% of customers first
                2. **Monitor KPIs**: Track conversion rate, revenue, and satisfaction
                3. **Be Transparent**: Clearly communicate value to customers
                4. **Stay Compliant**: Ensure pricing follows local regulations
                5. **Iterate**: Adjust parameters based on real-world results
                
                ### ⚠️ Important Considerations
                
                - Ensure fair and non-discriminatory pricing
                - Consider ethical implications
                - Monitor for unintended biases
                - Have clear pricing policies
                - Regular model retraining (quarterly)
                """)
        
        else:
            st.warning("⚠️ No features available for pricing model")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <h3 style="color: #d4af37; font-family: 'Playfair Display', serif;">💎 LUXE ANALYTICS</h3>
    <p style="font-size: 0.9rem; margin: 1rem 0;">
        <span class="badge badge-premium">7 Classification Algorithms</span>
        <span class="badge badge-premium">K-Means & K-Modes</span>
        <span class="badge badge-premium">Association Rules</span>
        <span class="badge badge-premium">3 Regression Models</span>
        <span class="badge badge-premium">AI Pricing</span>
    </p>
    <p style="font-size: 0.85rem; color: #888; margin-top: 1.5rem;">
        Built with Streamlit • Powered by scikit-learn • Designed for Excellence
    </p>
    <p style="font-size: 0.75rem; color: #666; margin-top: 0.5rem;">
        © 2025 Luxe Analytics Platform • Premium ML Insights
    </p>
</div>
""", unsafe_allow_html=True)
