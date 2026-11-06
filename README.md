# AIml-Project
Class project for heart dieases
## Resources & Experiment Documentation

### 📋 Executive Summary
This project implements and evaluates a **K-Nearest Neighbors (KNN)** pipeline to predict the presence of heart disease using the Kaggle `heart.csv` dataset (302 unique samples and 13 clinical features). Through iterative testing, we explored data preprocessing, hyperparameter optimization, bias-variance tradeoffs, data augmentation (SMOTE), and alternative scaling methods.

---

### 🛠️ Key Techniques & Methodology

1. **Data Preprocessing & Scaling**
   * **Duplicate Removal:** Identified and dropped 1 duplicate record, reducing the sample count from 303 to 302 to prevent data leakage.
   * **Feature Scaling:** Since KNN calculates distances (such as Euclidean or Manhattan) to make predictions, we applied **StandardScaler** (and compared it with **MinMaxScaler**) to guarantee that features with larger numerical ranges (like `chol` or `thalach`) do not dominate the distance metrics.
   * **Stratified Splitting:** Utilized a stratified 80-20 train-test split to preserve target class proportions (54% diseased, 46% healthy) across splits.

2. **Exploratory Data Analysis (EDA) Insights**
   * **Class Balance:** The target classes are relatively well-balanced (164 Disease cases, 138 No Disease cases).
   * **Key Predictors:** High correlation and clear boxplot shifts showed that **chest pain type (`cp`)** and **maximum heart rate (`thalach`)** are positively correlated with heart disease, whereas **ST depression (`oldpeak`)**, **number of major vessels (`ca`)**, and **thalassemia (`thal`)** are negatively correlated.

3. **Bias-Variance Tradeoff Analysis**
   * **Overfitting ($K=1$):** Yielded $100\%$ training accuracy but poor generalization to unseen test data ($75.41\%$) due to capturing high-frequency noise in the dataset.
   * **Optimal ($K=22$):** Successfully balanced bias and variance, achieving high performance on both training ($83.82\%$) and validation sets, with the two learning curves tightly converging.
   * **Underfitting ($K=29$):** Too many neighbors led to overly simple decision boundaries, reducing performance on both train and test sets.

4. **Data Augmentation via SMOTE**
   * **Oversampling:** Used Synthetic Minority Over-sampling Technique (SMOTE) to balance the training set classes to an equal 131-131 distribution. Given that our dataset was already reasonably balanced, SMOTE yielded matching performance compared to standard training ($80.33\%$ test accuracy).

5. **Hyperparameter Tuning & Search Space**
   * Explored **450 candidate configurations** via a 10-fold Stratified GridSearchCV:
     * $K$ range: $1$ to $25$
     * Weights: `['uniform', 'distance']`
     * Distance Metrics: `['euclidean', 'manhattan', 'minkowski']` (with power parameter $p \in [1, 2, 3]$)
   * **Best Parameters Found:** `{'metric': 'manhattan', 'n_neighbors': 21, 'p': 1, 'weights': 'uniform'}`.

---

### 📊 Performance Summary Matrix

| Model Variant | Scaling | Optimization Strategy | Test Accuracy | Test AUC-ROC |
|:---|:---|:---|:---|:---|
| **Baseline KNN ($K=5$)** | StandardScaler | Default Hyperparameters | **$80.33\%$** | - |
| **Tuned KNN ($K=22$)** | StandardScaler | Cross-Validation on $K$ | **$80.33\%$** | - |
| **SMOTE + KNN ($K=22$)** | StandardScaler | SMOTE Class Balancing | **$80.33\%$** | - |
| **GridSearchCV Optimized** | StandardScaler | Comprehensive Grid Search | **$77.05\%$** | **$0.9172$** |
| **GridSearchCV Optimized** | MinMaxScaler | Alternative Scale Check | **$78.69\%$** | - |

* Although the grid-search model on test data showed slightly lower test accuracy (77.05%), its excellent **AUC-ROC of 0.917** confirms that the classifier retains highly robust probabilistic separation power between classes.*
