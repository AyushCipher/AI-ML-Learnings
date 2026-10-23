# 🧠 Deep Learning — Module 1: Artificial Neural Networks (ANN)

Welcome to the **Artificial Neural Networks (ANN)** foundational module of the Deep Learning repository. This module provides a complete, progressive theory-to-practice curriculum structured from **Basics to Advanced**.

---

## 📁 Repository Structure

```text
4. Deep Learning/
└── 1. ANN/
    ├── README.md                                                        # Module Index & Theory Guide
    ├── data/                                                            # Central Data Store
    │   ├── Bank_Customer_Churn.csv                                      # 10,000-row Bank Churn Dataset
    │   └── Iris.csv                                                     # 150-row Iris Dataset
    └── notebooks/                                                       # Curated Notebooks (Basics to Advanced)
        ├── 1. Basics - Perceptron & Linear Models.ipynb                 # 🟢 Level 1: Perceptron Foundations & Logic Gates
        ├── 2. Core - Artificial Neural Networks & Iris Classification.ipynb # 🟡 Level 2: Multilayer ANN & Iris Case Study
        ├── 3. Deep Dive - Activations, Loss Functions & Optimizers.ipynb # 🟠 Level 3: Deep Dive into Functions & Algorithms
        ├── 4. Advanced Project - Bank Customer Churn Prediction System.ipynb # 🔴 Level 4: Industrial End-to-End Project
        └── 5. Masterclass - Complete ANN End-to-End Curriculum.ipynb    # 🏆 Unified Complete Masterclass
```

---

## 🗺️ Progressive Learning Curriculum (Basics to Advanced)

### [🟢 1. Basics: Perceptron & Linear Decision Boundaries](file:///c:/Users/ayush/OneDrive/Desktop/TagPractice/Gen-AI-Learnings/4.%20Deep%20Learning/1.%20ANN/notebooks/1.%20Basics%20-%20Perceptron%20%26%20Linear%20Models.ipynb)
* **Biological Inspiration**: Mapping Dendrites, Soma, Axons, and Synapses to artificial neural components.
* **McCulloch-Pitts Neuron (1943)**: Threshold logic and binary decision units.
* **Frank Rosenblatt's Perceptron (1958)**: Linear combination $z = \mathbf{w}^T\mathbf{x} + b$ and Heaviside step activation.
* **Perceptron Learning Algorithm (PLA)**: Weight update rule $\Delta w_j = \eta(y - \hat{y})x_j$, error signals, and hyperplane rotation.
* **NumPy Implementation from Scratch**: Linearly separable AND and OR logic gates with decision boundary plots.
* **The Fatal Flaw & The First AI Winter**: Mathematical proof by Minsky & Papert (1969) demonstrating why single-layer perceptrons fail on non-linear XOR functions.

---

### [🟡 2. Core: Artificial Neural Networks & Iris Classification](file:///c:/Users/ayush/OneDrive/Desktop/TagPractice/Gen-AI-Learnings/4.%20Deep%20Learning/1.%20ANN/notebooks/2.%20Core%20-%20Artificial%20Neural%20Networks%20%26%20Iris%20Classification.ipynb)
* **From Perceptrons to Multi-Layer Perceptrons (MLP)**: Hidden layers as hierarchical feature extractors.
* **Anatomy of a Neuron & Biases**: Why bias $b$ is mandatory to prevent decision boundaries from being anchored to the origin.
* **Universal Approximation Theorem**: Mathematical expressiveness of deep networks over wide shallow networks.
* **Matrix Calculus**: Forward propagation ($Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}$) and backpropagation error flow via the chain rule.
* **NumPy 2-Layer Neural Network from Scratch**: Training an MLP on XOR and visualizing the non-linear continuous probability decision surface.
* **Comparative Case Study on Iris**: Training a linear Perceptron vs. a Multi-Layer ANN with Softmax, featuring pairplots and side-by-side confusion matrices.

---

### [🟠 3. Deep Dive: Activations, Loss Functions & Optimizers](file:///c:/Users/ayush/OneDrive/Desktop/TagPractice/Gen-AI-Learnings/4.%20Deep%20Learning/1.%20ANN/notebooks/3.%20Deep%20Dive%20-%20Activations,%20Loss%20Functions%20%26%20Optimizers.ipynb)
* **Mathematical Necessity of Non-Linearity**: Proof that multi-layer linear networks collapse into a single-layer linear model.
* **Activation Functions**: Equations, analytical derivatives, output ranges, and tradeoffs for:
  * **Step Function**, **Sigmoid**, **Tanh**, **ReLU**, **Leaky ReLU**, **ELU**, **Softmax**, **Swish/SiLU**, and **GELU**.
  * Explanations and visual remedies for the **Vanishing Gradient** and **Dying ReLU** problems.
* **Loss Functions**:
  * Regression: **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **Huber Loss**.
  * Classification: **Binary Cross-Entropy (BCE)**, **Categorical Cross-Entropy (CCE)**, and **Sparse Categorical Cross-Entropy**.
  * Mathematical proof why MSE fails when paired with Sigmoid for classification (non-convex loss surface).
* **Optimization Algorithms**:
  * Optimization challenges: Local minima, plateaus, saddle points, ravines.
  * **Batch GD**, **SGD**, **Mini-Batch GD**.
  * **Momentum**, **Nesterov Accelerated Gradient (NAG)**.
  * **AdaGrad**, **RMSprop**, **Adam**, and **AdamW** (decoupled weight decay).
  * Empirical benchmark shootout on Beale's non-convex surface with comparative convergence curves.

---

### [🔴 4. Advanced Project: Bank Customer Churn Prediction System](file:///c:/Users/ayush/OneDrive/Desktop/TagPractice/Gen-AI-Learnings/4.%20Deep%20Learning/1.%20ANN/notebooks/4.%20Advanced%20Project%20-%20Bank%20Customer%20Churn%20Prediction%20System.ipynb)
* **Business Context & Retention Economics**: Preventing customer attrition on 10,000 banking customers.
* **Exploratory Data Analysis**: Target imbalance audit (~14% churn), age vs. churn distributions, and country attrition rates.
* **Leakage-Free Preprocessing**: Categorical encoding (`OneHotEncoder`, binary `Gender`), stratified 70/15/15 train/val/test split, and `StandardScaler` fitted strictly on training partition.
* **Machine Learning Baselines**: Establishing benchmarks using Logistic Regression and Random Forest.
* **Deep ANN Architecture**:
  $$\text{Input}(11) \to \text{Dense}(64, \text{ReLU}) + \text{BatchNorm} + \text{Dropout}(0.3) \to \text{Dense}(32, \text{ReLU}) + \text{BatchNorm} + \text{Dropout}(0.2) \to \text{Dense}(16, \text{ReLU}) + \text{Dropout}(0.1) \to \text{Dense}(1, \text{Sigmoid})$$
* **Dynamic Training**: EarlyStopping and ReduceLROnPlateau callbacks.
* **Diagnostic Evaluation**: Confusion Matrix, ROC-AUC curve, and Precision-Recall (PR-AUC) curve.
* **Decision Threshold Optimization**: Tuning decision threshold ($0.50 \to 0.35$) to boost churn recall, minimizing expensive false negatives.
* **Production Inference Pipeline**: Real-time `predict_customer_churn()` API accepting raw customer profiles, returning probability scores, risk tiering, and targeted retention strategies.

---

### [🏆 5. Masterclass: Complete ANN End-to-End Curriculum](file:///c:/Users/ayush/OneDrive/Desktop/TagPractice/Gen-AI-Learnings/4.%20Deep%20Learning/1.%20ANN/notebooks/5.%20Masterclass%20-%20Complete%20ANN%20End-to-End%20Curriculum.ipynb)
* The complete, unified reference combining all four progressive modules into a single, cohesive, publication-quality interactive notebook.

---

## ⚡ Quick Reference: Activation & Loss Selection Guide

| Task / Problem Type | Output Activation | Recommended Loss Function | Primary Evaluation Metric |
| :--- | :--- | :--- | :--- |
| **Binary Classification** | Sigmoid | Binary Cross-Entropy (Log Loss) | ROC-AUC, PR-AUC, F1-Score |
| **Multi-Class Classification** | Softmax | Categorical Cross-Entropy | Multiclass F1, Accuracy |
| **Multi-Class (Integer Labels)** | Softmax | Sparse Categorical Cross-Entropy | Multiclass F1, Accuracy |
| **Regression (Standard)** | Linear / Identity | Mean Squared Error (MSE / $L_2$) | RMSE, $R^2$ Score |
| **Regression (Outlier-Robust)**| Linear / Identity | Huber Loss / Smooth $L_1$ | MAE, Huber Loss |
