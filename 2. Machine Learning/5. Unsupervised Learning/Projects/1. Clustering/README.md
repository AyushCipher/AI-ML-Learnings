# 🌐 Unsupervised Clustering Algorithms

A comprehensive study and benchmarking of unsupervised clustering algorithms on non-linear and spherical multi-dimensional data.

---

## 📌 Project Overview
This project explores partitioning and density-based clustering techniques to uncover latent group structures in unlabeled datasets:
- **K-Means Clustering**: Lloyd's algorithm, centroid convergence, and finding the optimal $k$ using the **Elbow Method (WCSS / Inertia)** and **Silhouette Analysis**.
- **Hierarchical / Agglomerative Clustering**: Dendrogram construction, linkage criteria (Ward, Complete, Average, Single).
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Core points, border points, epsilon ($\epsilon$) neighborhood, `min_samples`, and arbitrary shape discovery.

---

## 📁 Directory Structure
```
1. Clustering/
├── Clustring.ipynb                 # Complete clustering implementations and evaluations
├── clustering_visualizations.png   # High-resolution benchmark comparison visualizations
└── README.md                       # Project documentation
```

---

## 🚀 How to Run
Open and run in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook Clustring.ipynb
```
