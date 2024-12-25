# 🏠 U.S. Residential Housing Price Prediction  

A comprehensive project focused on predicting **residential housing prices** and **Real Estate Investment Trusts (REIT)** performance using data from **Redfin**, **Zillow**, and **Alpha Vantage**. This repository features machine learning models, backtesting frameworks for trading strategies, and an interactive **Streamlit application** for visualization and stakeholder engagement.

---

## 📈 Overview  
- Predict housing prices across U.S. counties using Redfin and Zillow data.  
- Forecast REIT prices into the future using tree-based models.  
- Evaluate custom **trading strategies** with a backtesting framework.  
- Access predictions and insights via a deployed **Streamlit app**.

🔗 **[Explore the Deployed Streamlit App](https://residential-reit-prediction.streamlit.app/)**  

---

## 🛠️ Key Features  
- **Data Integration:**  
  - Aggregates housing metrics such as average sales price and total listings from **Redfin** and **Zillow**.  
  - Incorporates REIT data and market benchmarks from **Alpha Vantage**.  

- **Predictive Modeling:**  
  - Trains tree-based models (Random Forest, Adaboost, XGBoost) to forecast REIT prices.  
  - Features robust feature engineering and cross-validation for accurate predictions.  

- **Trading Strategies:**  
  - Implements a backtesting framework to compare custom REIT trading strategies with market benchmarks like SPY.  
  - Analyzes strategy performance using metrics such as returns, volatility, and Sharpe ratios.  

- **Streamlit Application:**  
  - Interactive dashboard presenting housing price trends, REIT forecasts, and trading strategy evaluations.  
  - Deployed and accessible for stakeholders to explore results in real-time.  

---

## 📂 Project Structure  
```
📁 data/  
  ├── REIT_predictions.csv  
  ├── final_data.csv

📁 notebooks/  
  ├── AVB_modeling.ipynb
  ├── EQR_modeling.ipynb
  ├── ESS_modeling.ipynb
  ├── INVH_modeling.ipynb
  ├── backtesting.ipynb  
  ├── data_preprocessing.ipynb  

📁 streamlit_app/
  ├── pages
    ├── 1_Data_Overview.py
    ├── 2_Visualizations.py
    ├── 3_Predictions.py
    ├── 4_Trading_Strategies.py   
  ├── Home.py  
```

---

## 🔧 Tools & Technologies  
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Data Sources**: Redfin, Zillow, Alpha Vantage  
- **Deployment**: Streamlit  

---

## 🚀 How to Run Locally  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/itsGarrin/us-residential-housing-price-prediction.git
   cd us-residential-housing-price-prediction
   ```

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:  
   ```bash
   streamlit run streamlit_app/Home.py
   ```

4. **Explore the Results**: Open `localhost:8501` in your browser.

---

## 📊 Results & Insights  

- **Predictions:** Accurate forecasts for REIT prices with visualization of trends.  
- **Backtesting:** Trading strategies outperformed market benchmarks in specific timeframes.  
- **Streamlit App:** A user-friendly interface for exploring predictions and strategy evaluations.  

---

Contributions, issues, and feedback are welcome!
