# Heart Disease Prediction using K-Nearest Neighbors (KNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Subhadip24/AIml-Project/blob/main/Heart_Disease_KNN_Classification.ipynb)

A class project for the AI/ML course. It implements a **K-Nearest Neighbors (KNN)** classifier in Python to predict whether a patient has heart disease from 13 routine clinical measurements, using the Kaggle `heart.csv` dataset in Google Colab. The project tunes the model to improve accuracy, tests data augmentation with SMOTE, and demonstrates both overfitting and underfitting by sweeping K from 1 to 200.

---

## Team

| Name | Enrollment No. |
|:-----|:---------------|
| Subhadip Mondal | 241001001186 |
| Souradipto Majumdar | 241001001177 |
| Dishtya Thakur | 241001001204 |
| Ayush Saha | 241001001154 |

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [How the Project Meets the Brief](#how-the-project-meets-the-brief)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [How to Run](#how-to-run)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Overfitting vs. Underfitting Analysis](#overfitting-vs-underfitting-analysis)
9. [Key Findings](#key-findings)
10. [Limitations and Future Work](#limitations-and-future-work)
11. [References](#references)

---

## Problem Statement

> Implement the K-means clustering algorithm or KNN using Python, using a real-life example.
> **Dataset:** Kaggle (`heart.csv`, provided in this repository). **Platform:** Google Colab.
> You can change optimizers or activation functions, add augmentation to the data, or use any kind of fine-tuning method to increase the overall accuracy.
> Check the overfitting and underfitting concepts by analysing the overall algorithm.

**Why KNN and not K-means?** Every patient in the dataset has a known diagnosis label (`target`), so this is a **supervised classification** problem. KNN is a supervised classifier; K-means is an unsupervised clustering method that ignores labels.

**Real-life example:** a screening aid that flags patients who are likely to have heart disease, based on measurements taken in a routine check-up.

---

## How the Project Meets the Brief

| Requirement | Where it is addressed |
|:------------|:----------------------|
| KNN in Python | `KNeighborsClassifier` from scikit-learn, wrapped in `Pipeline`s (notebook sections 4–10) |
| Real-life example | Heart disease screening from clinical data |
| Kaggle `heart.csv` | Loaded in section 1; file included in this repo |
| Google Colab | Notebook opens and runs in Colab (badge above); data loads automatically |
| Change optimizers / activation functions | **Not applicable to KNN.** KNN has no training loop, optimizer or activation function. Its equivalent tuning knobs were tuned instead: K, distance metric, vote weighting and feature scaler (sections 5, 8, 9) |
| Fine-tuning to increase accuracy | 10-fold cross-validation over K = 1–200; `GridSearchCV` over 150 unique parameter combinations; StandardScaler vs. MinMaxScaler comparison |
| Data augmentation | SMOTE (Synthetic Minority Over-sampling Technique), inside the pipeline (section 7) |
| Overfitting / underfitting analysis | Train vs. validation accuracy across K, learning curves, bias-variance error plot, with data-driven diagnosis of K = 1, the CV-optimal K and the underfitting K (section 6) |
| Honest evaluation | Repeated cross-validation comparison (section 10) and final summary (section 11) |

---

## Repository Structure

```
AIml-Project/
├── Heart_Disease_KNN_Classification.ipynb   # Full pipeline: EDA, preprocessing, KNN, tuning, SMOTE, analysis
├── heart.csv                                # Kaggle heart disease dataset (303 rows, 14 columns)
└── README.md                                # This file
```

---

## Dataset

**Source:** Kaggle heart disease dataset (derived from the UCI Cleveland Heart Disease data).

| Property | Value |
|:---------|:------|
| Rows | 303 (302 after removing 1 duplicate) |
| Features | 13 (all numeric) |
| Target | `target`: 1 = heart disease, 0 = no heart disease |
| Missing values | None |
| Class balance | 164 disease (54%) / 138 no disease (46%) |

### Feature dictionary

| Feature | Description | Type / range |
|:--------|:------------|:-------------|
| `age` | Age in years | 29–77 |
| `sex` | Sex (1 = male, 0 = female) | Binary |
| `cp` | Chest pain type | Categorical, 0–3 |
| `trestbps` | Resting blood pressure (mm Hg) | 94–200 |
| `chol` | Serum cholesterol (mg/dl) | 126–564 |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Binary |
| `restecg` | Resting electrocardiogram result | Categorical, 0–2 |
| `thalach` | Maximum heart rate achieved | 71–202 |
| `exang` | Exercise-induced angina (1 = yes) | Binary |
| `oldpeak` | ST depression induced by exercise relative to rest | 0.0–6.2 |
| `slope` | Slope of the peak exercise ST segment | Categorical, 0–2 |
| `ca` | Number of major vessels coloured by fluoroscopy | 0–4 |
| `thal` | Thalassemia test result | Categorical, 0–3 |

### Strongest correlations with the target

| Positive | Negative |
|:---------|:---------|
| `cp` (+0.43) | `exang` (−0.44) |
| `thalach` (+0.42) | `oldpeak` (−0.43) |
| `slope` (+0.34) | `ca` (−0.41) |
| | `thal` (−0.34) |

Correlation signs follow this Kaggle version's target encoding, where `target = 1` means disease.

---

## How to Run

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. Select **Runtime → Run all**.

The notebook installs `imbalanced-learn` itself. It loads `heart.csv` from the current folder if present, and otherwise reads it directly from this GitHub repository, so no manual upload is needed. A full run takes a few minutes, mostly for the K sweep and repeated cross-validation.

### Option 2: Run locally

```bash
git clone https://github.com/Subhadip24/AIml-Project.git
cd AIml-Project
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook Heart_Disease_KNN_Classification.ipynb
```

### Requirements

| Library | Purpose |
|:--------|:--------|
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | KNN, pipelines, scaling, cross-validation, grid search, metrics |
| `imbalanced-learn` | SMOTE oversampling inside a pipeline |

---

## Methodology

### 1. Data cleaning
- Checked for missing values: none found.
- Removed **1 duplicate row** (303 → 302 samples) so the same patient cannot appear in both the training and test sets.

### 2. Exploratory data analysis
- Target class distribution, feature correlation heatmap and correlation with the target
- Age distribution by diagnosis
- Box plots of `age`, `trestbps`, `chol`, `thalach` and `oldpeak` by diagnosis

### 3. Preprocessing and leakage-free design
- **Stratified 80/20 train-test split** (`random_state=42`): 241 training and 61 test patients, preserving the 54/46 class ratio.
- **Feature scaling** with `StandardScaler`. KNN classifies by distance, so without scaling, large-range features such as `chol` would dominate small-range ones such as `oldpeak`.
- **Every model is a `Pipeline`** (scaler → [SMOTE] → KNN). The scaler and SMOTE are therefore fitted only on training data, including inside every cross-validation fold.
- **One fixed cross-validation scheme** (stratified 10-fold, shuffled, `random_state=42`) is used throughout, so results are comparable.

### 4. Baseline model
- KNN with the default K = 5, Euclidean distance and uniform weights.

### 5. Tuning K
- For K = 1–30 and 35–200 (step 5), recorded training accuracy, test accuracy and 10-fold cross-validation accuracy.
- Chose **K = 27**, which had the highest cross-validation accuracy (83.37%). The CV curve is flat near the top: many values of K score within one standard error of the best.

### 6. Data augmentation (SMOTE)
- SMOTE oversamples the minority class in the training data from 110 to 131, so both classes have 131 samples.
- It sits inside the pipeline, so in cross-validation it only ever sees training folds.

### 7. Grid search
- `GridSearchCV` with the same 10-fold cross-validation over **150 unique combinations**:
  - `n_neighbors`: 1–25
  - `weights`: `uniform`, `distance`
  - `metric`: `euclidean`, `manhattan`, and `minkowski` with `p = 3`
- `p` only affects the Minkowski metric (Minkowski with p = 1 or 2 is identical to Manhattan or Euclidean), so the grid avoids duplicate combinations.
- **Best parameters:** `metric='manhattan'`, `n_neighbors=19`, `weights='distance'` (CV accuracy 84.62%).

### 8. Scaler comparison
- Re-trained the grid-search winner with `MinMaxScaler` and compared it against `StandardScaler`.

### 9. Repeated cross-validation
- Compared all model variants with **repeated stratified 10-fold cross-validation** on the full dataset (10 repeats = 100 train/test splits), reporting accuracy, recall on the disease class and AUC.

---

## Results

### Test-set results (61 patients)

| Model | CV accuracy | Test accuracy | Disease recall | Test AUC | Sick patients missed (of 33) |
|:------|:-----------:|:-------------:|:--------------:|:--------:|:----------------------------:|
| K = 1 (overfit) | 77.98% | 75.41% | 0.818 | 0.748 | 6 |
| Baseline, K = 5 | 80.47% | 80.33% | 0.909 | 0.867 | 3 |
| **CV-tuned, K = 27** | 83.37% | **81.97%** | **0.970** | 0.924 | **1** |
| SMOTE + K = 27 | 81.72% | 81.97% | 0.939 | 0.914 | 2 |
| Grid search (Manhattan, K = 19, distance) | **84.62%** | 75.41% | 0.909 | **0.926** | 3 |
| K = 155 (underfit) | 75.08% | 78.69% | 1.000 | 0.846 | 0 |

Scaler comparison for the grid-search winner: StandardScaler 84.62% CV / 75.41% test; MinMaxScaler 83.37% CV / 77.05% test.

### Repeated cross-validation (100 splits)

Mean over 100 folds, ± standard deviation across the 10 repeats.

| Model | Accuracy | Disease recall | AUC |
|:------|:--------:|:--------------:|:---:|
| K = 1 (overfit) | 0.762 ± 0.010 | 0.806 ± 0.013 | 0.757 ± 0.010 |
| Baseline, K = 5 | 0.822 ± 0.005 | 0.865 ± 0.008 | 0.884 ± 0.004 |
| CV-tuned, K = 27 | 0.826 ± 0.005 | 0.932 ± 0.003 | 0.906 ± 0.006 |
| SMOTE + K = 27 | 0.833 ± 0.004 | 0.900 ± 0.010 | 0.906 ± 0.006 |
| Grid search best | 0.825 ± 0.007 | 0.901 ± 0.008 | 0.914 ± 0.006 |
| K = 155 (underfit) | 0.790 ± 0.003 | 0.963 ± 0.003 | 0.899 ± 0.004 |

### Reading the results

- **Tuning K improved the model.** Compared with the baseline, the CV-tuned K = 27 raised CV accuracy from 80.47% to 83.37% and test accuracy from 80.33% to 81.97%. It also raised disease recall from 0.91 to 0.97: it missed only 1 of 33 sick patients, against 3 for the baseline.
- **The test set is small.** Each of the 61 test patients is worth about 1.6 accuracy points, so the test gain above is one patient. The repeated-CV comparison is the more reliable guide.
- **Grid search topped CV but not the test set.** Its winner uses distance weighting, which gives 100% training accuracy by construction. It scored 84.62% on CV but only 75.41% on test. Under repeated CV it is statistically tied with the CV-tuned K, so this is mostly test-split noise rather than a worse model.
- **SMOTE had little effect.** It changed 2 of 61 test predictions and left test accuracy unchanged, because the classes were already close to balanced (54/46).
- **For screening, recall matters most.** A missed sick patient costs more than a false alarm. The CV-tuned K = 27 offers the best balance of accuracy and recall.

---

## Overfitting vs. Underfitting Analysis

The notebook chooses the three demonstration values of K from the data rather than hard-coding them. Its diagnosis is computed from the train-validation gap and from how far each score falls below the best.

| K | Train accuracy | CV accuracy | Test accuracy | Diagnosis |
|:-:|:--------------:|:-----------:|:-------------:|:----------|
| 1 | 100.00% | 77.98% | 75.41% | **Overfitting.** A 22-point train-validation gap: the model memorises every training point, including noise. |
| 27 | 83.82% | 83.37% | 81.97% | **Good fit.** High accuracy, with train and validation almost equal. |
| 155 | 78.42% | 75.08% | 78.69% | **Underfitting.** Training and validation accuracy both drop: each vote averages over most of the training set. |

**Why underfitting only appears at very large K.** Between K ≈ 5 and K ≈ 150, accuracy barely changes: this dataset tolerates very smooth decision boundaries. Underfitting sets in from about K = 155, and by K = 200 validation error reaches about 45%. The original version of this project stopped the sweep at K = 30, which was too early to show it.

**How to read the plots in section 6:**
- **Train vs. test accuracy:** the vertical distance between the curves is the variance gap. It is widest at K = 1 and nearly closes around the CV-optimal K.
- **Learning curves:** at K = 1, training accuracy stays at 100% while validation accuracy is about 20 points lower. At K = 27 the two curves converge. At K = 155 they converge at a lower level.
- **Bias-variance plot:** training error approximates bias, and the gap to validation error approximates variance. Error is high at very low K (variance) and rises sharply at very high K (bias).

---

## Key Findings

1. **KNN performs well on this problem.** The CV-tuned model (K = 27) reaches 81.97% test accuracy, 0.97 disease recall and 0.924 AUC.
2. **Both overfitting and underfitting are clearly visible.** K = 1 overfits (100% train, 75% test); K ≥ 155 underfits; K ≈ 5–150 sits in a broad good-fit region.
3. **Fine-tuning raised cross-validation accuracy** from 80.47% to 84.62%. Its gain on the 61-patient test set is within noise, as repeated cross-validation confirms.
4. **Scaling, inside a pipeline, is essential** for distance-based KNN. StandardScaler slightly beat MinMaxScaler on CV accuracy.
5. **SMOTE gave no meaningful improvement**, since the classes were already nearly balanced.
6. **Optimizer and activation-function tuning does not apply to KNN.** The equivalent hyperparameters were tuned instead.

---

## Limitations and Future Work

- **Small dataset.** 302 patients make single-split results noisy. Nested cross-validation would give an unbiased estimate of the whole tuning process.
- **Accuracy is not the right target for screening.** Future work should tune for recall on the disease class or use a cost-weighted metric, possibly with a lower decision threshold.
- **Categorical features** (`cp`, `restecg`, `slope`, `thal`) are treated as numbers. One-hot encoding them may improve distance calculations.
- **Benchmark against other models** such as logistic regression, random forest or SVM.

---

## References

- Kaggle heart disease dataset (UCI Cleveland Heart Disease data)
- scikit-learn documentation: [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- imbalanced-learn documentation: [`SMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.
