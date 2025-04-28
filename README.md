## 🧬 SVM Classification

**Problem Statement**:  
Apply multiple regression and classification algorithms, including Linear Regression, Logarithmic Regression, k-Nearest Neighbors (kNN), Naive Bayes, and SVM classification. Analyze performance across different models and evaluate classification results with precision, recall, and ROC.

**Data**:  
- Source: Titanic - Machine Learning from Disaster [`Titanic`](https://www.kaggle.com/c/titanic)
- Description: Multiple CSV files containing numerical features for predicting continuous variables (regression) or categorical labels (classification).

**Data Mining Operations**:  
- Loaded multiple datasets into Pandas dataframes.
- Applied data preprocessing: handled missing values, performed min-max normalization.
- Regression Models:
  - Conducted Simple Linear Regression and Multivariate Regression.
  - Conducted Logarithmic Regression for nonlinear relationships.
- Classification Models:
  - Applied k-Nearest Neighbors (kNN) Classification.
  - Applied Gaussian Naive Bayes Classification.
  - Conducted Support Vector Machine (SVM) Classification using different kernels (linear, polynomial, RBF).
- Evaluated SVM models by generating classification reports with precision, recall, F1 score, and plotted ROC curves.
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

**Model Outputs**:  
- Correlation matrices and heatmaps for feature relationships.
- Regression scatter plots showing fits between features and target variables.
- Confusion matrices for classification models.
- ROC curves for SVM models.
- Precision, Recall, and F1 metrics across classification algorithms.

**Limitations**:  
- Some datasets were small, limiting generalization and causing possible variance in cross-validation.
- SVM performance varied significantly depending on the chosen kernel and feature scaling.

**Were you able to effectively solve the problem?**  
Yes, each regression and classification model was successfully trained, evaluated, and insights were drawn regarding which models performed best under different conditions.
