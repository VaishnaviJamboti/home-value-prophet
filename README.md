# home-value-prophet
# 🏠 Real Estate Investment Advisor
### Predicting Property Profitability & Future Value using Machine Learning

A complete end-to-end **machine learning application** that helps investors make data-driven real estate decisions — classifying whether a property is a **"Good Investment"** and predicting its **estimated price after 5 years**, all through an interactive Streamlit dashboard.

---

## 📌 Problem Statement

The real estate market often lacks accessible, data-driven tools for individual investors to assess long-term profitability. Most buyers rely on intuition rather than data when making high-value decisions.

This project addresses that gap by building an ML-powered advisor that:
- ✅ **Classifies** whether a property is a "Good Investment" based on growth and quality metrics
- ✅ **Predicts** the estimated property price after a 5-year period
- ✅ **Provides** an interactive platform for real-time investment analysis and visualization

---

## 🎯 Business Use Cases

| User | Benefit |
|---|---|
| 🏦 Real Estate Investors | Assess long-term returns and risk factors intelligently |
| 🏡 Individual Home Buyers | Choose high-return properties based on infrastructure and appreciation trends |
| 🏢 Property Listing Platforms | Automate investment analysis for listings with data-backed predictions |

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python | Primary language for data processing and ML |
| Framework | Streamlit | Interactive web dashboard |
| Data Handling | Pandas & NumPy | Cleaning, transformation, feature engineering |
| Machine Learning | Scikit-learn | Random Forest, Logistic Regression, Linear Regression |
| Visualization | Plotly | 20+ interactive EDA charts |
| Experiment Tracking | MLflow | Model iteration logging and management |

---

## 📂 Project Structure

```
real-estate-ml-advisor/
│
├── app.py                        # Main Streamlit application entry point
│
├── pages/
│   ├── 1_Home.py                 # Project overview & quick insights
│   ├── 2_EDA_Explorer.py         # 20 interactive EDA charts
│   ├── 3_Investment_Predictor.py # Real-time ML prediction interface
│   └── 4_Model_Performance.py    # Model metrics, ROC curve, MLflow summary
│
├── models/
│   ├── train_classifier.py       # Random Forest investment classifier
│   ├── train_regressor.py        # Random Forest price predictor
│   └── mlflow_experiments.py     # Experiment tracking & model comparison
│
├── data/
│   ├── raw_data.csv              # Original dataset (250,000 rows)
│   └── cleaned_data.csv          # Processed dataset (240,106 rows)
│
├── utils/
│   └── preprocessing.py          # Feature engineering functions
│
├── requirements.txt
└── README.md
```

---

## 🔧 Key Development Steps

### Phase 1 — Data Preprocessing
- Cleaned a raw dataset of **250,000 rows**
- Resolved critical issues: incorrect property ages, illogical floor numbers
- Final cleaned dataset: **240,106 rows** with 0 nulls and 0 duplicates

### Phase 2 — Feature Engineering
Created two powerful new features:
- **`Infrastructure_Score`** — a composite proxy for neighborhood quality (schools, hospitals, transport, security)
- **`Appreciation_Rate`** — captures historical price growth to drive investment classification

### Phase 3 — Machine Learning Pipeline
Built and trained models using **Scikit-learn** ensuring results were **free from data leakage**:

| Task | Model | Metric | Score |
|---|---|---|---|
| Investment Classification | Random Forest | Accuracy | **87.6%** |
| Investment Classification | Random Forest | ROC-AUC | **0.94** |
| Investment Classification | Logistic Regression | Accuracy | ~75% |
| Price Prediction (5-Year) | Random Forest | R² Score | **0.857** |
| Price Prediction (5-Year) | Random Forest | MAE | **₹31.47 Lakhs** |
| Price Prediction (5-Year) | Linear Regression | R² Score | ~0.72 |

### Phase 4 — Streamlit Deployment
Built a **4-page interactive web application** for data exploration and real-time forecasting.

---

## 📊 EDA Highlights (20 Charts)

**Price & Size Analysis**
- Distribution of property prices (₹20L–₹500L, near-uniform)
- Price vs Size scatter with trend line
- Outlier detection via box plots

**Location-Based Analysis**
- Avg price by state (Uttar Pradesh leads at ~₹12,358/SqFt)
- Price trends across top 5 most expensive localities
- State-wise property distribution

**Feature Relationship & Correlation**
- Correlation heatmap of all numeric features
- Nearby schools/hospitals vs Price/SqFt
- Price by furnished status and facing direction

**Investment & Ownership Insights**
- Properties by owner type distribution
- Availability status breakdown
- Infrastructure score vs investment classification

---

## 🧠 ML Model Details

### Investment Classifier (Random Forest)
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
# Target: 'Good_Investment' (1) or 'Poor_Investment' (0)
# Key features: Infrastructure_Score, Appreciation_Rate, 
#               Price_per_SqFt, BHK, City, State
```

**Top Feature Importances:**
1. Appreciation_Rate
2. Infrastructure_Score  
3. Price_per_SqFt
4. BHK Type
5. Transport_Accessibility

### Price Predictor (Random Forest Regressor)
```python
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Target: Estimated price after 5 years
# Formula applied: current_price × (1 + appreciation_rate)^5
```

---

## 🖥️ Application Pages

### 🏠 Home
- Dataset at a glance (240K rows, 42 features)
- Project pipeline overview
- State-wise property distribution chart
- Property type distribution donut chart

### 📊 EDA Explorer
- 20 interactive Plotly charts across 4 categories
- Filter by: Price Range, Location, BHK Type, Investment Status
- Correlation matrix for all numeric features

### 🔮 Investment Predictor
Input any property details and get:
- ✅ Investment classification (Good / Poor)
- 📈 Confidence score (%)
- 💰 Current estimated price
- 🔮 5-year price projection with growth chart
- 📋 Investment score breakdown (Appreciation, Infrastructure, Amenities, Height/SqFt)

### 📈 Model Performance
- Model leaderboard comparing all algorithms
- Confusion matrix for Random Forest classifier
- ROC curve (Random Forest vs Logistic Regression)
- MLflow experiment summary table

---

## 📈 Key Business Insights

| Insight | Finding |
|---|---|
| Price Distribution | Nearly uniform ₹20L–₹500L, mean ~₹255L |
| Top State by Price/SqFt | Uttar Pradesh (~₹12,358/SqFt) |
| Most Popular Property Type | 2BHK and 3BHK dominate across all major cities |
| Infrastructure Impact | High infrastructure scores are the #1 predictor of "Good Investment" |
| Amenities Effect | Parking and amenities have minimal direct effect on current price |

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/real-estate-ml-advisor.git
cd real-estate-ml-advisor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
mlflow
```

---

## 📸 Screenshots

> *(Add screenshots of your app pages here)*
> - Home Dashboard
> - EDA Explorer
> - Investment Predictor (with 5-year projection)
> - Model Performance & ROC Curve

---

## 🔮 Future Enhancements

- [ ] Add location-based map visualization (Folium/Plotly Maps)
- [ ] Integrate live property listing data via API
- [ ] Add loan EMI calculator alongside price prediction
- [ ] Build ensemble model combining RF + XGBoost for better accuracy
- [ ] Deploy on Streamlit Community Cloud

---

## 👤 Author

**Anil**
Data Analytics Intern — Labmentix
- 📧 jambotivaishnavi0@gmail.com
- 💼 linkedin.com/in/vaishnavi-jamboti-8179ab314   github.com/VaishnaviJamboti    
- 🐙 github.com/VaishnaviJamboti

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ using Python, Scikit-learn, Streamlit, Plotly, and MLflow*.
