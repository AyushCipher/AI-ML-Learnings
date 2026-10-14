# 🫀 Heart Disease - Exploratory Data Analysis (EDA)

An exploratory data analysis project investigating clinical parameters associated with heart disease risk.

---

## 📌 Project Overview
The objective of this exploratory analysis is to examine the statistical distributions, risk patterns, and feature correlations in patient clinical data.

### Key Analysis Highlights
- **Target Distribution**: Assessing the balance between positive (`HeartDisease = 1`) and negative (`HeartDisease = 0`) patient classes.
- **Univariate Analysis**: Histograms, KDE plots, and box plots for Age, RestingBP, Cholesterol, and MaxHR.
- **Bivariate & Multivariate Patterns**:
  - Chest pain presentation (`ChestPainType`) vs. heart disease prevalence.
  - Exercise-induced angina and ST slope abnormalities in correlation with cardiovascular risk.
  - Pairwise correlation heatmaps highlighting inter-feature collinearity.

---

## 📁 Directory Structure
```
1. Heart Disease EDA/
├── heart.csv        # Clinical dataset (918 records, 11 features + 1 target)
├── Heart.ipynb      # Complete exploratory data analysis & visualization notebook
└── README.md        # Project documentation
```

---

## 🚀 How to Run
Launch Jupyter Notebook or VS Code / IDE:
```bash
jupyter notebook Heart.ipynb
```
The notebook directly reads `heart.csv` from the same folder.
