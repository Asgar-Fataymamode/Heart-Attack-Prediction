# Predicting Heart Attack Likelihood: A Comparative Analysis of Logistic Regression, Random Forest, and XGBoost Models

## Introduction

Heart disease remains one of the foremost global health challenges, significantly contributing to morbidity and mortality rates worldwide. Among heart-related conditions, heart attacks (myocardial infarctions) stand out for their sudden onset and potentially life-threatening consequences. Early and accurate prediction of heart attack risk has far-reaching implications: clinicians could intervene sooner, tailor prevention strategies to individual patients, and ultimately improve patient outcomes. Such predictive capabilities also carry the promise of reducing hospital readmissions, streamlining resource allocation, and fostering more proactive healthcare systems.

In this project, we develop and evaluate predictive models to estimate the likelihood of a heart attack based on patient characteristics and clinical measurements. The dataset includes a variety of features such as age, sex, resting blood pressure, cholesterol levels, chest pain type, exercise-induced angina, and other indicators derived from ECG results and exercise tests. By experimenting with multiple classification algorithms—Logistic Regression as a baseline, Random Forest as a robust ensemble method, and XGBoost as a state-of-the-art gradient boosting technique—we aim to determine which approach provides the best balance of accuracy, precision, and recall.

Our ultimate goal is twofold:

1. Gain insights into which features most strongly influence heart attack risk.

2. Identify a model that can serve as a reliable tool for clinicians. While more complex models might promise higher predictive power, this analysis will show that simplicity can sometimes yield surprising advantages. The results highlight the importance of careful model selection, hyperparameter tuning, and interpretability, especially in medical contexts where decisions can have profound life-or-death implications.

## Data Description

**Data Source & Features:**  
The dataset is sourced from a publicly available heart disease analysis and prediction dataset on Kaggle. It includes 302 patient records (after removing one duplicate), each with 14 attributes.

Dataset Link: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

Key features are:

- **Continuous Features:** 
  - **age:** Age of the patient in years.
  - **trtbps:** Resting blood pressure in mmHg.
  - **chol:** Cholesterol level in mg/dL.
  - **thalachh:** Maximum heart rate achieved during exercise.
  - **oldpeak:** ST depression induced by exercise relative to rest.
  
- **Categorical Features:**
  - **sex:** 0 = female, 1 = male.
  - **cp:** Chest pain type (0 = typical angina, 1-3 = varying degrees of pain).
  - **fbs:** Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes).
  - **restecg:** Resting ECG results (0, 1, 2 representing increasing severity).
  - **exng:** Exercise-induced angina (0 = no, 1 = yes).
  - **slp:** Slope of the ST segment during exercise (0 = downsloping, 1 = flat, 2 = upsloping).
  - **caa:** Number of major vessels colored by fluoroscopy (0-4).
  - **thall:** Thalassemia test results (1-3 indicating normal, fixed defect, or reversible defect).

- **Target Variable (output):**  
  0 = Less chance of heart attack  
  1 = Higher chance of heart attack

**Initial Observations:**
- The target classes are relatively balanced, slightly favoring output=1. This balanced class distribution is beneficial for model training and evaluation.
- The dataset primarily involves middle-aged to older adults, a demographic commonly associated with increased cardiovascular risk.

## Exploratory Data Analysis (EDA)

### Univariate Analysis of Continuous Features
Histograms of age, trtbps, chol, thalachh, and oldpeak provide insight into their distributions:

- **age:** Approximately bell-shaped, with most patients aged 45–60.
- **trtbps:** Slightly above normal on average (around 130–140 mmHg), reflecting a population at potential risk.
- **chol:** Centered around 200–250 mg/dL, suggesting moderately high cholesterol levels.
- **thalachh:** Peaks near 150 bpm, indicating decent exercise capacity in many patients.
- **oldpeak:** Right-skewed, with most values near zero, but a subset experiencing significant ST depression (a sign of potential ischemia).

### Univariate Analysis of Categorical Features
Bar plots of sex, cp, fbs, restecg, exng, slp, caa, and thall show:

- **sex:** More males than females, aligning with historical trends in heart disease studies.
- **cp:** Typical angina (0) is the most common chest pain type.
- **fbs:** Most patients have normal fasting blood sugar.
- **restecg:** Normal or near-normal ECG findings are common; severe abnormalities are rare.
- **exng:** Exercise-induced angina is not prevalent in most patients.
- **slp:** Most patients have normal or upsloping ST segments, indicative of less severe pathology.
- **caa:** The majority have zero major vessels colored, indicating fewer advanced coronary lesions.
- **thall:** Values of 2 and 3 dominate; their specific interpretations depend on the coding scheme but can indicate normal or fixed defects.

### Correlations and Pairwise Relationships
A correlation heatmap reveals that no single feature perfectly predicts the target. However:

- **cp** and **thalachh** correlate positively with the target, suggesting chest pain type and higher achievable heart rates may be associated with heart disease presence.
- **exng, oldpeak, caa, thall** show negative correlations with the “healthy” category, implying their abnormal values correlate with heart disease.

Pairwise plots confirm the complexity: no two features alone distinctly separate the classes. This suggests that effective prediction demands combining multiple variables and possibly nonlinear models.

### Groupwise Comparisons by Output
When comparing distributions by class (0 vs. 1):

- **Continuous Features by Output:**  
  Slight variations appear:
  - Patients with output=1 may be slightly younger and have higher max heart rates but show lower ST depression on average. The latter is somewhat counterintuitive, potentially reflecting complex underlying factors.
- **Categorical Features by Output:**  
  Certain chest pain types and thall categories appear more frequently in the heart attack group, while exercise-induced angina patterns defy simple expectations.
- **Target Distribution:**  
  The near-balance in classes is an advantage, reducing the need for sampling techniques.

### Principal Component Analysis (PCA)
PCA on numeric features fails to yield a clear linear separation between classes, reinforcing the notion that the relationship between these features and heart disease likelihood may be intricate and not easily captured in a low-dimensional linear subspace.

**EDA Conclusion:**  
No single factor dominates in predicting heart disease. Both continuous and categorical features provide subtle hints. The complexity suggests that machine learning models must leverage the entire feature set.

## Models and Methods

**Modeling Steps:**
1. **Data Preparation:**
   - Duplicate rows removed, no missing values found.
   - Categorical features one-hot encoded.
   - Numeric features scaled to support models sensitive to scale (e.g Logistic Regression).

2. **Train-Test Split:**
   - 80/20 split stratified by target ensures balanced class distribution in both sets.

3. **Baseline Model: Logistic Regression**
   - Chosen for simplicity, interpretability, and as a baseline.
   - No hyperparameter tuning initially; used default parameters.

4. **Random Forest (Ensemble Model)**
   - Utilizes multiple decision trees for robust performance.
   - Evaluated with default parameters for a fair initial comparison.

5. **XGBoost (Gradient Boosting)**
   - Tested with default parameters.
   - Then tuned using a GridSearchCV approach to improve performance (adjusting n_estimators, max_depth, learning_rate, and subsampling parameters).

**Evaluation Metrics:**
- **Accuracy:** Percentage of correct predictions.
- **Precision and Recall:** Crucial in medical contexts. High recall for the heart attack class (1) ensures fewer missed high-risk patients, while high precision reduces false alarms.
- **F1-Score:** Harmonizes precision and recall into a single metric.
- **Confusion Matrix:** Provides granular insight into true positives, true negatives, false positives, and false negatives.

## Results and Interpretation

### Logistic Regression (Baseline)
**Performance:**
- Accuracy: ~0.836
- Precision: ~0.829
- Recall: ~0.879
- F1-Score: ~0.853

**Interpretation:**
Logistic Regression surprises by offering the best balance. It correctly identifies a high number of positive cases (heart attacks) and maintains good precision and recall. Few false negatives mean it rarely misses high-risk patients—paramount in a medical setting.

### Random Forest
**Performance:**
- Accuracy: ~0.738
- Precision: ~0.730
- Recall: ~0.818
- F1-Score: ~0.771

**Interpretation:**
Random Forest, despite its complexity, underperforms relative to Logistic Regression. While it identifies many positive cases (high recall), it misclassifies more healthy individuals as heart attack patients (higher false positives). This reduces precision and overall accuracy.

### XGBoost (Default)
**Performance:**
- Accuracy: ~0.689
- Precision: ~0.694
- Recall: ~0.758
- F1-Score: ~0.725

**Interpretation:**
Without tuning, XGBoost lags behind both Logistic Regression and Random Forest. Although it handles positive cases decently, it struggles with correct classification of non-heart attack patients, leading to a lower overall accuracy.

### XGBoost (Tuned)
After hyperparameter tuning:
**Performance:**
- Accuracy: ~0.721
- Precision: ~0.711
- Recall: ~0.818
- F1-Score: ~0.761

**Interpretation:**
Tuning improves XGBoost’s performance, making it competitive with Random Forest. It identifies positive cases well and improves precision slightly, but still cannot surpass Logistic Regression’s impressive balance.

### Confusion Matrices and Detailed Metrics
- **Logistic Regression Confusion Matrix:**  
  Few false negatives (4) and a good number of true positives (29) highlight this model’s strength in not missing those at risk.
  
- **Random Forest and XGBoost:**  
  Increased false positives and/or lower accuracy indicate these models struggle to achieve the nuanced balance that Logistic Regression provides.

- **ROC Curve Comparison**
  The ROC curves demonstrate that Logistic Regression (AUC = 0.89) outperforms Random Forest (AUC = 0.86) and XGBoost (AUC = 0.81) in terms of separability between classes. The higher AUC for Logistic Regression indicates that it more consistently distinguishes between high-risk and low-risk patients across various probability thresholds.

- **Precision-Recall Curve Comparison**
The Precision-Recall curves show that Logistic Regression again leads with an Average Precision (AP) of 0.91, followed by Random Forest (AP = 0.89) and XGBoost (AP = 0.84). Since precision-recall curves focus on the predictive performance for the positive class (heart attack risk), this confirms that Logistic Regression not only identifies most at-risk patients but also does so with fewer false alarms compared to the other models.

### Feature Importance
- The Random Forest’s top features highlight oldpeak, thal-based variables (thal_2 and thal_3), and thalachh (maximum heart rate) as highly influential in predicting heart attack likelihood. Cholesterol (chol), resting blood pressure (trtbps), and age also play significant roles. This suggests that a combination of exercise-induced measures (oldpeak, thalachh) and certain thalassemia test results is critical for the Random Forest model’s decision-making.
- XGBoost places a dominant emphasis on a single feature: thal_2. This categorical feature related to thalassemia test results outweighs other features like exercise-induced angina (exng_1) and specific chest pain categories (cp_3). This indicates that, within XGBoost’s learned structure, variations in this particular thallium-based feature strongly influence its predictions.

### Overall Model Comparison
| Model                 | Accuracy  | Precision | Recall   | F1-Score |
|-----------------------|-----------|-----------|----------|----------|
| Logistic Regression    | 0.836066 | 0.828571  | 0.878788 | 0.852941 |
| Random Forest          | 0.737705 | 0.729730  | 0.818182 | 0.771429 |
| XGBoost (Default)      | 0.688525 | 0.694444  | 0.757576 | 0.724638 |
| XGBoost (Tuned)        | 0.721311 | 0.710526  | 0.818182 | 0.760563 |

**Key Takeaways:**
- Logistic Regression excels despite the availability of more complex algorithms.
- Ensemble methods (Random Forest, XGBoost) show potential but require more extensive tuning or feature engineering to match the simpler model’s performance.
- The high recall scores across models mean most at-risk patients are identified, but Logistic Regression’s higher precision reduces unnecessary alarms.

## Conclusion and Next Steps

**Summary of Findings:**
This project demonstrates that complexity does not always guarantee superiority. The Logistic Regression model, a simple linear classifier, outperformed more advanced models in predicting heart attack risk. It balanced precision and recall effectively, minimizing missed high-risk patients without causing an excessive number of false alarms.

**Insights:**
1. **Strong Predictors:**  
   Features related to chest pain type and exercise test results (thalachh, oldpeak, and related categorical variables) show consistent relevance. While no single feature perfectly distinguishes heart attack risk, these factors collectively contribute to better predictions.
   
2. **Balanced Classes:**  
   The relatively balanced target distribution simplifies model evaluation and reduces the need for resampling techniques.
   
3. **Model Complexity vs. Performance:**  
   Logistic Regression’s success here suggests that the underlying relationships might be sufficiently linear or that the data size and feature set favor simpler models. Advanced models might require more data, sophisticated hyperparameter tuning, or additional domain-specific features to outperform the baseline.

**Practical Implications:**
- A medical practitioner could implement a Logistic Regression-based decision-support tool to identify high-risk patients. The model’s simplicity and transparency align well with clinical settings, where interpretability is crucial.
- More advanced models might still have a role if future efforts include richer data sources (e.g imaging, genetic markers, advanced biomarkers) or if data size and complexity increase significantly.

**Next Steps for Improvement:**
1. **Feature Engineering:**  
   Incorporate domain knowledge to derive new features (e.g ratios like cholesterol-to-HDL ratio, or grouping chest pain types differently).
   
2. **Inclusion of Additional Data:**  
   Integrate more patient history (smoking habits, family history, medications) or lifestyle factors to capture a fuller picture of risk.
   
3. **Advanced Hyperparameter Tuning:**  
   While some tuning was done for XGBoost, Bayesian optimization methods could uncover better parameter sets.
   
4. **Interpretability Tools:**  
   Techniques like SHAP or LIME could be employed to explain model predictions on an individual patient level, bolstering clinician trust and facilitating insight into the drivers of risk.

**Conclusion:**
In a context where patient lives are at stake, the priority lies in accuracy, reliability, and interpretability rather than complexity. This analysis shows that a well-implemented, linear baseline model can outperform more complex algorithms on a given dataset. Logistic Regression stands out as a reliable, high-performing, and transparent tool for predicting heart attack likelihood, providing a strong foundation for future enhancements and clinical integration.

---
