
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "cleaned" / "cleaned_customer_churn.csv"
MODEL_PATH = (
    PROJECT_DIR
    / "task4_xgboost_tuning"
    / "models"
    / "best_xgboost_model.pkl"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f5f7fc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #202b46 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 24px;
    }

    .section-title {
        color: #172554;
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e7eaf2;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(30, 41, 59, 0.04);
        min-height: 125px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #172554;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 5px;
    }

    .info-card {
        background: white;
        border: 1px solid #e7eaf2;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .info-card h4 {
        color: #172554;
        margin: 0 0 8px 0;
        font-size: 16px;
    }

    .info-card p {
        color: #64748b;
        font-size: 13px;
        margin: 0;
        line-height: 1.6;
    }

    .risk-high {
        background: #fff1f2;
        border-left: 4px solid #e11d48;
    }

    .risk-medium {
        background: #fff7ed;
        border-left: 4px solid #f59e0b;
    }

    .risk-low {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error("Unable to load the dataset or model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# GENERATE CUSTOMER RISK SCORES
# --------------------------------------------------
required_columns = {"customer_id", "churn_flag"}

if not required_columns.issubset(df.columns):
    st.error("The dataset is missing required columns.")
    st.stop()

X = df.drop(columns=["customer_id", "churn_flag"])

try:
    probabilities = model.predict_proba(X)[:, 1]
except Exception as error:
    st.error("The model could not generate predictions. Please check the model and dataset features.")
    st.exception(error)
    st.stop()

results = df.copy()
results["churn_probability"] = probabilities * 100

results["risk_level"] = pd.cut(
    results["churn_probability"],
    bins=[-0.001, 30, 60, 100.001],
    labels=["Low", "Medium", "High"]
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 📊 ChurnIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.markdown("---")

    st.markdown("### 🎯 Dashboard")
    st.write("Explore customer churn patterns, identify high-risk customers and discover retention opportunities.")

    st.markdown("---")
    st.markdown("### 📌 Risk Classification")
    st.markdown("🔴 **High:** Above 60%")
    st.markdown("🟠 **Medium:** 30%–60%")
    st.markdown("🟢 **Low:** Below 30%")

    st.markdown("---")
    st.caption("Data Science Internship Project")
    st.caption("Model: XGBoost")


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">Customer Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Monitor customer behaviour • Identify churn risk • Improve retention decisions</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
total_customers = len(results)
actual_churn_rate = results["churn_flag"].mean() * 100
high_risk_count = (results["risk_level"] == "High").sum()
average_risk = results["churn_probability"].mean()

k1, k2, k3, k4 = st.columns(4)

def metric_card(column, label, value, note):
    column.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    """, unsafe_allow_html=True)

metric_card(k1, "👥 Total Customers", f"{total_customers:,}", "Customers in dataset")
metric_card(k2, "📉 Historical Churn", f"{actual_churn_rate:.1f}%", "Actual churn rate in dataset")
metric_card(k3, "🚨 High-Risk Customers", f"{high_risk_count:,}", "Model scores above 60%")
metric_card(k4, "🎯 Average Risk Score", f"{average_risk:.1f}%", "Average model probability")

st.markdown("")

# --------------------------------------------------
# TABS
# --------------------------------------------------
overview_tab, drivers_tab, recommendations_tab = st.tabs([
    "📊  Overview",
    "🔍  Churn Drivers",
    "💡  Recommendations"
])


# ==================================================
# OVERVIEW TAB
# ==================================================
with overview_tab:

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Actual Customer Churn")
        churn_counts = (
            results["churn_flag"]
            .map({0: "Retained", 1: "Churned"})
            .value_counts()
        )
        st.bar_chart(churn_counts, color="#6366f1")

    with right:
        st.markdown("#### Predicted Customer Risk")
        risk_counts = (
            results["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )
        st.bar_chart(risk_counts, color="#8b5cf6")

    st.markdown("---")
    st.markdown(
        '<div class="section-title">🚨 Top 10 High-Risk Customers</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Customers with the highest model-estimated churn probabilities."
    )

    top_10 = results.sort_values(
        "churn_probability",
        ascending=False
    ).head(10)

    display_columns = [
        "customer_id",
        "tenure",
        "contract",
        "monthly_charge",
        "churn_probability",
        "risk_level"
    ]

    st.dataframe(
        top_10[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "customer_id": st.column_config.TextColumn("Customer ID"),
            "tenure": st.column_config.NumberColumn("Tenure (Months)"),
            "contract": st.column_config.TextColumn("Contract"),
            "monthly_charge": st.column_config.NumberColumn(
                "Monthly Charge",
                format="%.2f"
            ),
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Probability",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "risk_level": st.column_config.TextColumn("Risk Level")
        }
    )

    csv_data = top_10[display_columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Top 10 Customers",
        data=csv_data,
        file_name="top_10_high_risk_customers.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Explore Customer Records</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        min_probability = st.slider(
            "Minimum churn probability (%)",
            min_value=0,
            max_value=100,
            value=0
        )

    with f2:
        selected_risk = st.selectbox(
            "Risk category",
            ["All", "High", "Medium", "Low"]
        )

    with f3:
        contract_options = ["All"] + sorted(
            results["contract"].dropna().astype(str).unique().tolist()
        )
        selected_contract = st.selectbox(
            "Contract type",
            contract_options
        )

    filtered = results[
        results["churn_probability"] >= min_probability
    ].copy()

    if selected_risk != "All":
        filtered = filtered[
            filtered["risk_level"].astype(str) == selected_risk
        ]

    if selected_contract != "All":
        filtered = filtered[
            filtered["contract"].astype(str) == selected_contract
        ]

    st.caption(f"Showing {len(filtered):,} matching customers.")

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    full_csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Customer Data",
        data=full_csv,
        file_name="filtered_customer_churn_data.csv",
        mime="text/csv"
    )


# ==================================================
# CHURN DRIVERS TAB
# ==================================================
with drivers_tab:

    st.markdown(
        '<div class="section-title">What Influences Customer Churn?</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore which features the trained XGBoost model uses most "
        "when making its predictions."
    )

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(15).sort_values("Importance")

        st.markdown("#### ⭐ Top 15 Model Feature Importances")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("")
        ax.set_title("Features Used by the XGBoost Model")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.warning(
            "Feature importance chart could not be generated. "
            "Please check the saved model pipeline."
        )
        st.caption(str(error))

    st.markdown("---")
    st.markdown("#### 📈 Historical Churn Patterns")

    available_attributes = [
        "contract",
        "internet_service",
        "payment_method",
        "gender",
        "senior_citizen"
    ]

    selected_attribute = st.selectbox(
        "Choose a customer attribute",
        available_attributes
    )

    pattern_df = (
        results.groupby(selected_attribute, dropna=False)
        .agg(
            Customers=("churn_flag", "count"),
            Churn_Rate=("churn_flag", "mean")
        )
        .reset_index()
    )

    pattern_df["Churn Rate (%)"] = pattern_df["Churn_Rate"] * 100

    pattern_df = pattern_df.rename(
        columns={selected_attribute: "Customer Group"}
    )

    st.bar_chart(
        pattern_df.set_index("Customer Group")["Churn Rate (%)"],
        color="#6366f1"
    )

    st.dataframe(
        pattern_df[["Customer Group", "Customers", "Churn Rate (%)"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Rate (%)": st.column_config.NumberColumn(
                "Churn Rate",
                format="%.2f%%"
            )
        }
    )

    st.info(
        "Feature importance describes how the model uses features. "
        "It does not prove that a feature causes customer churn."
    )


# ==================================================
# RECOMMENDATIONS TAB
# ==================================================
with recommendations_tab:

    st.markdown(
        '<div class="section-title">Retention Strategy Center</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use the following recommendations as starting points "
        "for customer-retention planning."
    )

    high_count = (results["risk_level"] == "High").sum()
    medium_count = (results["risk_level"] == "Medium").sum()

    r1, r2 = st.columns(2)

    with r1:
        st.markdown(f"""
            <div class="info-card risk-high">
                <h4>🔴 High-Risk Customers</h4>
                <h2 style="color:#be123c;">{high_count:,}</h2>
                <p>Prioritize these customers for retention outreach.</p>
            </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
            <div class="info-card risk-medium">
                <h4>🟠 Medium-Risk Customers</h4>
                <h2 style="color:#c2410c;">{medium_count:,}</h2>
                <p>Monitor these customers and identify possible concerns.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💡 Recommended Business Actions")

    recommendations = [
        (
            "01",
            "Prioritize high-risk customers",
            "Contact customers with high predicted churn probability "
            "and understand their reasons for dissatisfaction."
        ),
        (
            "02",
            "Review month-to-month contracts",
            "Explore suitable longer-term plans and explain their "
            "benefits to customers who may be considering leaving."
        ),
        (
            "03",
            "Improve customer support",
            "Review support requests and investigate recurring "
            "technical or service-related complaints."
        ),
        (
            "04",
            "Review payment experience",
            "Check whether payment difficulties or billing confusion "
            "are contributing to customer dissatisfaction."
        ),
        (
            "05",
            "Monitor customer tenure",
            "Pay particular attention to newer customers and provide "
            "helpful onboarding and early engagement."
        )
    ]

    for number, heading, description in recommendations:
        st.markdown(f"""
            <div class="info-card">
                <h4>✨ {number} | {heading}</h4>
                <p>{description}</p>
            </div>
        """, unsafe_allow_html=True)

    st.caption(
        "These are general business suggestions based on common "
        "retention practices and dataset patterns, not guaranteed outcomes."
    )


# --------------------------------------------------
# FOOTER AND DATA LIMITATION
# --------------------------------------------------
st.markdown("---")

st.markdown("""
<div class="footer">
    <b>Customer Churn Intelligence Dashboard</b><br>
    Built with Python • Streamlit • XGBoost<br><br>
    Dataset: Public IBM Telco Customer Churn dataset.<br>
    This is a simulated telecom retention case study, not actual boutique transaction data.<br>
    Model scores on this historical dataset are demonstration scores, not validated predictions
    of future customer behaviour.
</div>
""", unsafe_allow_html=True)
