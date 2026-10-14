# 🚨 Unsupervised Anomaly Detection Masterclass

A deep dive into statistical, tree-based, and density-based anomaly detection algorithms for detecting rare events and irregular patterns in unlabeled data.

---

## 📌 Project Overview
Anomaly and outlier detection is vital for fraud detection, cybersecurity intrusion prevention, sensor fault diagnostics, and financial risk mitigation. 

This notebook benchmarks three state-of-the-art anomaly detectors on non-linear synthetic geometries with injected outliers:
1. **Isolation Forest**: Sub-sampling, random recursive partitioning, path-length anomaly scoring (anomalies isolate with fewer splits).
2. **Local Outlier Factor (LOF)**: Measuring local density deviation of an observation with respect to its $k$-nearest neighbors.
3. **DBSCAN (Noise Identification)**: Utilizing density connectivity to isolate noise points ($label = -1$) that fail to satisfy core point criteria.

---

## 📁 Directory Structure
```
2. Anomaly Detection/
├── AnomalyDetection.ipynb          # End-to-end anomaly detection models & visualizations
└── README.md                       # Project documentation
```

---

## 🚀 How to Run
Open and run in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook AnomalyDetection.ipynb
```
