```
███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ██╗        ██╗  ██╗███████╗ █████╗ ██╗   ████████╗██╗  ██╗
████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██║        ██║  ██║██╔════╝██╔══██╗██║   ╚══██╔══╝██║  ██║
██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║██║        ███████║█████╗  ███████║██║      ██║   ███████║
██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║██║        ██╔══██║██╔══╝  ██╔══██║██║      ██║   ██╔══██║
██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║███████╗   ██║  ██║███████╗██║  ██║███████╗ ██║   ██║  ██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═╝   ╚═╝  ╚═╝
```

<h1 align="center">🧠 Prediction of Mental Health Treatment Patterns</h1>

<p align="center">
  <i>A complete end-to-end machine learning project — from raw survey data to predictive insights</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-2ecc71?style=for-the-badge" />
</p>

---

## 📖 Table of Contents

- [About The Project](#-about-the-project)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Step-by-Step Breakdown](#-step-by-step-breakdown)
  - [1. Setup & Data Loading](#1️⃣-setup--data-loading)
  - [2. Data Cleaning & Preprocessing](#2️⃣-data-cleaning--preprocessing)
  - [3. Exploratory Data Analysis](#3️⃣-exploratory-data-analysis-eda)
  - [4. Feature Selection](#4️⃣-feature-selection-techniques)
  - [5. Model Building](#5️⃣-model-building)
  - [6. Model Evaluation](#6️⃣-model-evaluation)
- [Results Summary](#-results-summary)
- [Key Findings & Insights](#-key-findings--insights)
- [Tech Stack](#️-tech-stack)
- [How to Run](#️-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🌟 About The Project

Mental health is one of the most underaddressed areas in modern healthcare. This project dives deep into **survey data from tech industry employees** to understand what factors influence whether a person seeks professional mental health treatment.

The goal is not just prediction — it's **interpretation**. By combining multiple feature selection methods and a transparent classification model, this notebook aims to answer:

> *"What are the most influential factors that drive someone to seek mental health treatment?"*

The full pipeline covers: data cleaning → rich EDA with 7+ visualizations → 4 feature selection techniques compared side by side → a fully evaluated Logistic Regression model — all in one clean, well-documented notebook.

🔗 **Kaggle Notebook:** [Mental Health EDA & Important Features](https://www.kaggle.com/code/mdnaimislam165436/mental-health-eda-important-features)

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Name** | Mental Health in Tech Survey |
| **Source** | [OSMI — Open Sourcing Mental Illness](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) |
| **File** | `survey.csv` |
| **Target Variable** | `treatment` — Whether the respondent has sought mental health treatment (Yes / No) |
| **Respondents** | Tech industry workers globally |
| **Features** | Demographics, workplace environment, mental health history, employer policies |

---

## 🔄 Project Pipeline

```
┌─────────────┐   ┌──────────────────┐   ┌─────────┐   ┌───────────────────┐   ┌───────────────┐   ┌────────────┐
│  Data Load  │──►│  Data Cleaning   │──►│   EDA   │──►│ Feature Selection │──►│Model Building │──►│ Evaluation │
└─────────────┘   └──────────────────┘   └─────────┘   └───────────────────┘   └───────────────┘   └────────────┘
```

Each step builds on the previous — raw data goes in, actionable predictions and insights come out.

---

## 📋 Step-by-Step Breakdown

### 1️⃣ Setup & Data Loading

The notebook begins by installing and importing all required libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

**Additional libraries installed:**
- `category_encoders` — Advanced categorical encoding support
- `skrebate` — ReliefF-based feature selection
- `boruta` — Boruta wrapper feature selection

The dataset is loaded from Kaggle's input directory and initially explored with `.columns` and `.head()` to understand its structure.

---

### 2️⃣ Data Cleaning & Preprocessing

Raw survey data is often messy and inconsistent. This phase brings it into a clean, model-ready state through four sub-steps:

#### 🔹 Column Formatting
All column names were stripped of whitespace, lowercased, and spaces replaced with underscores:
```python
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
```

#### 🔹 Gender Standardization
The `gender` column had 40+ inconsistent text entries. All were mapped to three clean categories:

| Raw Entry | Mapped To |
|---|---|
| `"maile"`, `"cis male"`, `"mal"`, `"man"`, `"male-ish"` | `male` |
| `"femail"`, `"cis female"`, `"woman"`, `"female"` | `female` |
| Everything else | `other` |

#### 🔹 Age Outlier Removal
Respondents with ages below 18 or above 65 were removed. Infinite and NaN values were handled. An **Age Distribution Histogram with KDE curve** confirmed a clean, realistic age range.

#### 🔹 Categorical Encoding
All `object`-type columns were encoded with `LabelEncoder` to create a fully numeric `data_encoding` DataFrame — ready for correlation analysis and model training.

---

### 3️⃣ Exploratory Data Analysis (EDA)

EDA was performed using **7 detailed visualizations** to reveal patterns before any modeling:

| # | Visualization | What It Reveals |
|---|---|---|
| 1 | 📊 Top 10 Countries — Bar Chart | Majority of respondents are from USA, UK, Canada |
| 2 | 📊 Top 10 States — Bar Chart | California, New York, Washington lead state-wise |
| 3 | 🥧 Country Pie Chart | USA dominates at ~60%+ of all responses |
| 4 | 🥧 Treatment Distribution Pie Chart | Roughly equal split between treatment vs. no treatment |
| 5 | 🥧 Top States Pie Chart | Regional concentration of tech workers visible |
| 6 | 🎻 Violin Plot — Age vs Treatment by Gender | Age ranges overlap; subtle gender differences in treatment rates |
| 7 | 🌡️ Spearman Correlation Heatmap | Multi-feature relationships shown at a glance |

#### 🔎 Top 10 Features Most Correlated with `treatment`:

```
family_history          ████████████████████  ← Strongest
care_options            ███████████████████
benefits                ██████████████████
obs_consequence         █████████████████
anonymity               ████████████████
work_interfere          ███████████████
state                   ██████████████
country                 █████████████
comments                ████████████
mental_health_interview ███████████
```

---

### 4️⃣ Feature Selection Techniques

Four independent feature selection methods were applied and compared. Using multiple techniques ensures the final selected features are **robust**, not just artifacts of one method.

---

#### ✅ Method 1 — Chi-Square Test

Features were normalized to [0, 1] using `MinMaxScaler`, then evaluated with the Chi-Square statistic:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Higher χ² score = stronger statistical dependency with the target variable.

```python
from sklearn.feature_selection import SelectKBest, chi2
chi2_selector = SelectKBest(chi2, k=10)
X_kbest = chi2_selector.fit_transform(X_scaled, y)
```

---

#### ✅ Method 2 — Recursive Feature Elimination (RFE)

`RFE` with `LogisticRegression` recursively removes the least important features at each step until only the top 10 remain:

```python
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X, y)
```

**Final RFE Selected Features:**
`family_history`, `care_options`, `benefits`, `obs_consequence`, `anonymity`, `work_interfere`, `state`, `country`, `comments`, `mental_health_interview`

---

#### ✅ Method 3 — SMLR (Sparse Multinomial Logistic Regression)

`LogisticRegressionCV` with **L2 regularization** and **5-fold cross-validation** was trained. Features were ranked by **absolute coefficient magnitude**:

```python
from sklearn.linear_model import LogisticRegressionCV
smlr = LogisticRegressionCV(cv=5, penalty='l2', max_iter=1000)
smlr.fit(X, y)
coef = pd.Series(np.abs(smlr.coef_[0]), index=X.columns)
```

This provides a **cross-validated, regularized** view of feature importance.

---

#### ✅ Method 4 — ReliefF (via Mutual Information)

`mutual_info_classif` scores each feature by how much information it provides about the target — measuring the reduction in uncertainty:

```python
from sklearn.feature_selection import mutual_info_classif
relieff_scores = mutual_info_classif(X, y)
```

---

#### 📊 Combined Feature Importance — Consensus Plot

All four methods were merged into a single DataFrame. A `counts` column tracks how many methods selected each feature. A **Feature Selection Counts Bar Chart** shows clear consensus:

> Features selected by **3 or 4 methods** = most trustworthy predictors ✅

---

### 5️⃣ Model Building

A **Logistic Regression** classifier was chosen — an interpretable and effective baseline for binary classification.

**Final Feature Set:**
```python
top_features = [
    'work_interfere', 'age', 'family_history', 'care_options',
    'state', 'country', 'no_employees', 'leave',
    'benefits', 'phys_health_interview'
]
```

**Training Pipeline:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

| Setting | Value |
|---|---|
| Algorithm | Logistic Regression |
| Train / Test Split | 67% / 33% |
| Feature Scaling | StandardScaler |
| Max Iterations | 1000 |
| Random State | 42 |

---

### 6️⃣ Model Evaluation

Four metrics were used for a complete, honest picture of model performance:

#### 🎯 Accuracy
```
Logistic Regression Accuracy: ~70.46%
```
The model correctly classifies approximately **7 out of 10 cases**.

---

#### 📋 Classification Report

| Class | Label | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **0** | No Treatment | 0.65 | 0.74 | 0.70 |
| **1** | Treatment | 0.76 | 0.67 | 0.71 |
| | **Macro Avg** | **0.71** | **0.71** | **0.70** |

The model is **well-balanced** — it doesn't heavily favor one class. A precision of 0.76 for class 1 means when it predicts "treatment", it's right 76% of the time.

---

#### 🔲 Confusion Matrix

```
                   Predicted: No    Predicted: Yes
Actual: No     [  True Negative  |  False Positive  ]
Actual: Yes    [  False Negative |  True Positive   ]
```

Plotted using `ConfusionMatrixDisplay` with a blue colormap — visually shows where the model succeeds and where it misclassifies.

---

#### 📈 ROC Curve & AUC Score

The **ROC curve** plots True Positive Rate vs. False Positive Rate across all decision thresholds.

- **AUC closer to 1.0** = better discriminating power
- The model's AUC confirms performance significantly above the random baseline (AUC = 0.5)
- The sharp rise toward the top-left corner indicates good class separability

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| **Model** | Logistic Regression |
| **Accuracy** | ~70.46% |
| **Precision (Treatment Class)** | 0.76 |
| **Recall (Treatment Class)** | 0.67 |
| **F1-Score (Macro Avg)** | ~0.70 |
| **Feature Selection Methods** | 4 (Chi-Square, RFE, SMLR, ReliefF) |
| **Train / Test Split** | 67% / 33% |
| **Top Feature** | `family_history` |

---

## 💡 Key Findings & Insights

After thorough analysis, several important patterns emerged:

**🔑 1. Family history is the #1 predictor**
Individuals with a family history of mental illness are significantly more likely to seek treatment — hereditary awareness plays a major role in help-seeking behavior.

**🏢 2. Workplace environment is critical**
`work_interfere`, `benefits`, `care_options`, and `anonymity` are top predictors — a supportive workplace culture directly influences whether employees seek help.

**🔒 3. Anonymity is a major barrier**
The strong predictive power of the `anonymity` feature reveals that people are far more willing to seek help when they feel their privacy is protected.

**📅 4. Age alone doesn't determine treatment-seeking**
The violin plot shows significant age overlap between those who did and didn't seek treatment — age is not a strong standalone predictor.

**⚧ 5. Gender plays a nuanced role**
While not the dominant feature, the violin plot shows subtle differences — females appear slightly more likely to seek treatment across age groups.

**🌍 6. Geography matters**
`country` and `state` both appear in top feature lists — mental health awareness and access to services vary significantly by region.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `Python 3.11` | Core language |
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Base plotting framework |
| `seaborn` | Statistical visualizations (heatmap, violin, barplot) |
| `scikit-learn` | ML models, preprocessing, feature selection, metrics |
| `category_encoders` | Advanced categorical encoding |
| `skrebate` | ReliefF-based feature selection |
| `boruta` | Boruta feature selection wrapper |

---

## ⚙️ How to Run

### 🏆 Option A — Run on Kaggle (Easiest)
1. Open: [Kaggle Notebook Link](https://www.kaggle.com/code/mdnaimislam165436/mental-health-eda-important-features)
2. Click **"Copy & Edit"**
3. Run all cells — the dataset is already connected!

### 💻 Option B — Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/mdnaim2004/Mental_Health_Prediction.git
cd Mental_Health_Prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders skrebate boruta
```

**3. Download the dataset**
Get `survey.csv` from [Kaggle OSMI Dataset](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) and place it in the project folder.

**4. Update the data path**
```python
# Change:
data = pd.read_csv("/kaggle/input/mental-health-in-tech-survey/survey.csv")
# To:
data = pd.read_csv("survey.csv")
```

**5. Launch Jupyter**
```bash
jupyter notebook mental-health-prediction.ipynb
```

---

## 🚀 Future Improvements

- [ ] Try ensemble models — **Random Forest**, **XGBoost**, **LightGBM** for higher accuracy
- [ ] Handle class imbalance with **SMOTE** or class weighting
- [ ] Use **SHAP values** to explain individual-level predictions
- [ ] Apply **k-fold cross-validation** for more reliable accuracy estimates
- [ ] **Hyperparameter tuning** with GridSearchCV or Optuna
- [ ] Add more **feature engineering** from open-text fields
- [ ] Build a **simple web app** (Streamlit) for live predictions

---

## 👤 Author

| | |
|---|---|
| **Name** | Md. Naim Islam |
| **Field** | Computer Science & Engineering |

<br/>

[![Kaggle](https://img.shields.io/badge/Kaggle-mdnaimislam165436-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mdnaimislam165436)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Md.%20Naim%20Islam-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-naim-00a164381/)
&nbsp;
[![Gmail](https://img.shields.io/badge/Gmail-naim.cse2004@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:naim.cse2004@gmail.com)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>⭐ If this project helped you, please give it a star on GitHub! It means a lot. ⭐</b>
</p>

<p align="center">
  Made with ❤️ and lots of ☕ by <b>Md. Naim Islam</b>
</p>
