# 🏥 Medical Insurance Charges - Exploratory Data Analysis (EDA)

An in-depth exploratory data analysis project uncovering demographic, lifestyle, and health factors that drive individual medical insurance costs.

---

## 📌 Project Overview
Healthcare and insurance expenditure is significantly influenced by personal demographics and health behaviors. This EDA project explores:
- **Smoking Impact**: Quantifying the dramatic cost surge associated with smoker status (`smoker = yes` vs. `smoker = no`).
- **BMI & Obesity Thresholds**: Analyzing non-linear charge escalation for BMI $> 30$.
- **Age Trends**: Tracking linear progression of baseline healthcare expenses as age advances.
- **Regional & Family Size Variations**: Comparing charges across US regions (`southwest`, `southeast`, `northwest`, `northeast`) and dependents (`children`).

---

## 📁 Directory Structure
```
2. Insurance EDA/
├── insurance.csv     # Medical insurance cost dataset (1,338 records, 7 attributes)
├── insurance.ipynb   # Visualizations, statistical summaries & EDA notebook
└── README.md         # Project documentation
```

---

## 🚀 How to Run
Open and execute the notebook in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook insurance.ipynb
```
The notebook automatically loads `insurance.csv` from its local folder.
