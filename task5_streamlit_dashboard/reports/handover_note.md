# Customer Churn Intelligence Dashboard
## Project Handover Document

### 1. Project Information

| Item | Details |
|---|---|
| Project | Customer Churn Intelligence Dashboard |
| Internship | Data Science Internship |
| Organization | Ezitech Software House |
| Application | Streamlit Web Dashboard |
| Machine Learning Model | Tuned XGBoost Pipeline |
| Dataset | IBM Telco Customer Churn (public dataset) |
| Main File | `task5_streamlit_dashboard/app.py` |

### 2. Project Objective

The objective of this project is to build an interactive dashboard that helps users explore customer churn patterns, identify customers with higher model-estimated churn risk, understand important model features, and review possible customer-retention strategies.

### 3. System Architecture

The application follows this workflow:

1. Load the cleaned customer churn dataset.
2. Load the saved XGBoost pipeline from Task 4.
3. Separate customer input features from customer IDs and the target column.
4. Generate churn probabilities using the model's `predict_proba()` method.
5. Assign dashboard risk categories.
6. Display results through three interactive dashboard sections.
7. Allow users to filter and download customer records.

### 4. Dashboard Modules

**Overview**
- Displays customer and churn KPIs.
- Visualizes actual churn and predicted risk distribution.
- Shows the top 10 customers with the highest estimated churn probabilities.
- Provides interactive filters and CSV downloads.

**Churn Drivers**
- Displays the top 15 model feature importances.
- Explores historical churn patterns across selected customer attributes.
- Provides customer group comparison tables.

**Recommendations**
- Summarizes high-risk and medium-risk customer counts.
- Presents practical customer-retention suggestions.

### 5. Architecture Decision Log

| Decision | Reason |
|---|---|
| Reuse the Task 4 XGBoost pipeline | Avoids retraining and maintains consistency with the previously evaluated model. |
| Use Streamlit | Provides an interactive dashboard with Python and minimal frontend complexity. |
| Use relative paths with `pathlib` | Supports project execution from different working directories and cloud deployment. |
| Use cached data and model loading | Reduces repeated loading and improves application responsiveness. |
| Use three dashboard tabs | Separates business overview, churn analysis, and recommendations for easier navigation. |
| Use probability-based risk categories | Makes model outputs easier to interpret for dashboard users. |
| Provide CSV downloads | Allows users to analyze filtered customer records outside the dashboard. |

### 6. Risk Classification

The dashboard uses these display thresholds:

- Low Risk: Probability below 30%.
- Medium Risk: Probability from 30% to 60%.
- High Risk: Probability above 60%.

These thresholds are application display choices and have not been independently validated as optimal business intervention thresholds.

### 7. How to Run the Application

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application from the project root:

```bash
python -m streamlit run task5_streamlit_dashboard/app.py
```

### 8. Deployment Information

The application is intended for deployment through Streamlit Community Cloud.

- GitHub repository: `https://github.com/HafsaSaddique/customer-churn-cleaning`
- Main file: `task5_streamlit_dashboard/app.py`
- Dependencies: `requirements.txt`
- Live application URL: https://hafsasaddique-customer-chur-task5-streamlit-dashboardapp-48yove.streamlit.app/

### 9. Limitations

- The project uses a public IBM Telco Customer Churn dataset as a simulated telecom retention case study, not actual boutique transaction data.
- Model scores on historical dataset records are demonstration scores and should not be treated as validated forecasts of future customer behaviour.
- Feature importance does not prove that a feature causes churn.
- Recommendations are general suggestions and require business validation before implementation.
- No live CRM integration or automated customer-contact functionality is included.

### 10. Future Improvements

- Integrate updated customer data from a secure business source.
- Validate prediction thresholds using business costs and retention outcomes.
- Add model monitoring and performance tracking.
- Introduce explainability tools such as SHAP.
- Add secure user authentication and role-based access if required.

### 11. Handover Status

The dashboard application and documentation have been prepared for local testing and deployment. The live deployment URL and final deployment status will be added after cloud deployment is completed.

---

Prepared as part of the Data Science Internship at Ezitech Software House.
