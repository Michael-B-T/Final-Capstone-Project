# Predicting Depression Risk from Behavioral and Lifestyle Data
Capstone Project – Machine Learning & AI Program

Michael Tucker

# Executive Summary
## Project Overview and Goals

This project explores whether patterns in everyday behavior, such as sleep, physical activity, social interaction and other lifestyle variables, can be used to predict depression risk. Because a suitable real world dataset with PHQ-9 scores was not available at this stage, a synthetic dataset was generated to imitate realistic, research supported relationships between behavioral factors and depression symptoms.

The project includes a complete exploratory data analysis, data cleaning pipeline, baseline modeling, hyperparameter tuning, and performance evaluation across multiple classification algorithms.

## Key Findings

Models trained on the synthetic dataset were able to predict depression with moderate accuracy.

Sleep quality, physical activity, screen time, and frequency of social interactions were consistently among the most influential predictors.

Logistic Regression and Random Forest performed best among baseline models, with Random Forest showing the most stable feature importance patterns.

Although the dataset is synthetic, the observed patterns mirror common findings in psychological and behavioral research.

## Conclusion

This initial phase demonstrates that behavioral features can signal elevated depression risk and that classification models can successfully learn these patterns. However, validation on real human data is essential before drawing any clinical conclusions. This work establishes the modeling framework, evaluation pipeline, and analytic foundation required for that next step.

## Problem Statement

Depression affects millions of people annually, and early identification is a key factor in improving treatment outcomes. Because behavioral changes often emerge before clinical diagnosis, a model that can estimate risk based on routine lifestyle indicators could support screening efforts or population level mental-health surveillance. I want to find a way to combine machine learning and AI with psychological assessment tools. To that end, I've done my best to create a predictive model that estimates a person's risk of depression based on features of their daily habits and behavioral patterns.



## Goal:
Build and evaluate machine learning models that can classify individuals into “higher-risk” or “lower-risk” categories for depression using behavioral/lifestyle data.

## Potential benefits:

Earlier identification of more at risk individuals

Support for mental health research

Building foundation for more sophisticated clinical screening tools

## Model Outcomes & Learning Approach

Learning type: Supervised learning

Task: Binary classification — predict whether an individual’s profile places them at elevated risk of depression

Model outputs: A probability score between 0–1 and a predicted risk class (0 = lower risk, 1 = higher risk)

Models tested:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine
- k-Nearest Neighbors (baseline)

# Data Acquisition
## Sources

Because validated PHQ-9 datasets were not available, two datasets were created and used:

depression_synth.csv is the raw synthetic data

depression_synth_clean_for_modeling.csv is the much more cleaned version used for modeling
Both files are included in the project repository.

## Features

The dataset includes commonly researched behavioral indicators associated with depression:

- Sleep duration and sleep quality

- Physical activity level

- Screen time

- Social media use

- Diet/nutrition indicators

- Frequency of in person social interactions

- Demographic elements (age, education)

## Initial Data Visualization

Distributions, correlations, and plots were created to evaluate:

- Variability and skew

- Relationships to depression scores

- Potential outliers

- Class separation

These visualizations confirmed that the synthetic data behaved realistically and contained useful signal for modeling.

# Data Preprocessing & Preparation
## Cleaning Steps

- Removed duplicated rows

- Imputed missing values with median (numeric) or mode (categorical)

- Encoded categorical variables using one-hot encoding

- Standardized numeric variables for models sensitive to scale

- Dropped non-informative identifiers

## Train/Test Split

An 80/20 split used for holdout evaluation

I also used tratified sampling ensured proportional representation of both depression risk classes

## Additional Processing

- Feature scaling for Logistic Regression, SVM, and KNN

- Verified no data leakage occurred

- Confirmed class balance was sufficient to use ROC-AUC and accuracy without correction

# Modeling Approach
## Models Selected

A range of classification algorithms were chosen to establish baselines and compare performance:

- Logistic Regression (interpretable, stable baseline)

- Random Forest Classifier (nonlinear relationships, feature importance)

- Support Vector Machine (tested with linear kernel)

- k-Nearest Neighbors (simple comparison baseline)

## Hyperparameter Tuning

Grid Search with cross validation was applied to:

- Regularization parameters (Logistic Regression)

- Tree depth, number of estimators, and split criteria (Random Forest)

- C values for SVM

- k values for KNN

Cross validation helped avoid overfitting and provided more stable performance estimates.

# Model Evaluation
## Metrics Used

Accuracy: overall proportion of correct predictions

Recall: emphasized due to the goal of identifying higher-risk individuals

Precision: monitored to ensure the model does not over-predict risk

ROC-AUC: measures ranking ability independent of threshold

## Results Summary

Logistic Regression: Balanced performance and strong interpretability

Random Forest: Best balance of predictive power and stability; most informative feature importances

SVM: Good performance but sensitive to scaling and parameters

KNN: Weakest baseline, used mainly for comparison

Random Forest emerged as the most practical model for this phase due to:

- Better representation of nonlinear relationships

- More robust behavior across cross-validation folds

- Clearer permutation importance scores

# Findings
## Key Insights

Sleep quality and social interaction frequency were the strongest predictors of depression risk. Interestingly, sleed duration was a much weaker predictor, meaning that it doesn't matter as much how long you sleep, only that it's restful. It might also be that depression causes restless sleep and that's the causation, not the other way around.

High screen time combined with low physical activity was associated with higher PHQ-9 values.

These patterns match established findings in behavioral psychology and mental-health research.

## Interpretation for Non-Technical Stakeholders

The model identifies lifestyle patterns that correlate with higher depression risk scores.

These patterns should absolutely not be used for clinical decisions without validation on real human data. The data I used for this project was synthesized, and though it looks good, testing with real data should be the next step.

The results support the idea that behavioral based risk screening models may be viable with appropriate data.

## Limitations

As I've said multiple times, the dataset is synthetic, that is by far the most important aspect to consider. Additionally, correlations may not reflect real world variance. Clinical variables such as medication status or chronic illness were not included either, so there's much more data to consider in the future.

## Next Steps & Recommendations

Acquire a real-world dataset with validated depression scores (PHQ-9 or similar).

Improve feature engineering, especially around:

- Time-series behavior

- Interaction terms

- Demographic normalization

Extend modeling:

Gradient boosting (XGBoost, LightGBM)

- Calibration curves for better probability estimates

- Incorporate explainability tools like SHAP or LIME.

Collaborate with psychology/behavioral-health experts to refine features and validate assumptions.


# Contact

Michael Tucker

Email: Mvcatucker@gmail.com

LinkedIn: https://www.linkedin.com/in/michael-b-tucker/
