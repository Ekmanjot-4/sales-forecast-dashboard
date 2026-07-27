# 📈 Store Sales Analysis & Time Series Forecasting

An end-to-end Python data analysis and time series decomposition project analyzing retail sales performance across multiple regions, categories, and seasons.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Time Series Analysis:** `statsmodels`
* **Evaluation Metrics:** `scikit-learn` (`mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`)

---

## 🚀 Key Features & Analysis Tasks

### Task 1: Data Preprocessing & Exploratory Data Analysis (EDA)
* **Datetime Feature Engineering:** Parsed order/ship dates to extract `Year`, `Month`, `Week Number`, `Day of Week`, `Quarter`, and `Season`.
* **Data Hygiene:** Verified null values, checked for duplicates, and transformed datatypes.
* **Category Performance:** Discovered that the **Technology** category generated the highest total revenue (~$827,455).
* **Regional Growth:** Identified the **East** region as having the most consistent upward sales trajectory over the 4-year period.
* **Logistics & Delay:** Evaluated average shipping delays across regions (overall average: ~3.96 days).
* **Seasonality:** Identified annual sales spikes, with **November** consistently showing peak performance.

### Task 2: Time Series Decomposition
* Resampled order dates to monthly sales aggregates.
* Performed **Additive Seasonal Decomposition** (`statsmodels.tsa.seasonal.seasonal_decompose`) to break down sales data into four distinct components:
  1. **Observed:** Raw aggregated monthly sales time series.
  2. **Trend:** Long-term direction and growth trajectory.
  3. **Seasonal:** Recurring annual patterns.
  4. **Residuals:** Random noise and unexplained variance.

---

## 📂 Repository Structure

```text
├── data/
│   └── train.csv                 # Raw dataset
├── notebooks/
│   └── Sales_Analysis.ipynb      # Main Google Colab / Jupyter Notebook
├── README.md                     # Project documentation
└── requirements.txt              # Project dependencies
