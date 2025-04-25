
# 🚢 Titanic Survival Prediction – A Data Science Project

This project analyzes Titanic passenger data to **predict survival** using various machine learning models and showcases a full **data science pipeline** including EDA, data cleaning, feature engineering, model training, and evaluation.

---

## 📁 Project Structure

```
.
├── data/
│   └── tested.csv                # Titanic dataset
├── notebooks/
│   └── titanic_model.ipynb       # Jupyter notebook with EDA & model trials
├── src/
│   ├── data_loader.py            # Loads dataset
│   ├── preprocessing.py          # Cleans and processes data
│   ├── train_models.py           # Trains models
│   ├── evaluate.py               # Evaluates model performance
│   └── main.py                   # Runs full pipeline
├── requirements.txt              # Required Python packages
└── README.md
```

---

## 🎯 Objective

The goal is to **explore the Titanic dataset**, understand key patterns using **Exploratory Data Analysis (EDA)**, and build models to **predict which passengers survived** the Titanic disaster.

---

## 🧠 Key Steps in the Project

1. **📊 EDA (Exploratory Data Analysis)**  
   - Visualizations of age, class, gender, and survival rates  
   - Handling missing data  
   - Insights into how features affect survival

2. **🔧 Data Preprocessing**  
   - Filling missing values  
   - Label encoding / One-hot encoding  
   - Feature selection & scaling

3. **🧪 Model Building & Evaluation**  
   - Logistic Regression, Decision Trees, Random Forest, etc.  
   - Accuracy, precision, recall, F1 score, confusion matrix

4. **💡 Insights**  
   - Women and children had higher survival chances  
   - Passengers in 1st class had better survival odds  
   - Fare and age were important features

---

## 📈 Sample EDA Plot

```python
sns.barplot(x="Sex", y="Survived", data=data)
```

---

## 🧪 How to Run the Project

1. Clone the repo or download the project files.

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the main pipeline:

```bash
python src/main.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook notebooks/titanic_model.ipynb
```

---

## 🛠 Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn (EDA & visualization)
- Scikit-learn (ML models & evaluation)
- Jupyter Notebook

---

## 🚀 Future Enhancements

- Use advanced models like XGBoost or SVM  
- Add model saving and prediction on new test data  
- Create a dashboard using Streamlit or Flask  
