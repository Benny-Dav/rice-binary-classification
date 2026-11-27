Rice Variety Binary Classification

This is a machine learning project that uses physical measurements of rice grains to classify them into two varieties: Cammeo and Osmancik.

The project explores an end-to-end supervised machine learning workflow, covering data exploration, preprocessing, model development, evaluation, error analysis, model comparison and predictions on unseen data.

Project Overview

The objective is to build a classification model that can identify the variety of a rice grain based on its physical characteristics and measurements.
This is a binary classification problem with two target classes:
-Cammeo
-Osmancik

Dataset
The project uses the Rice Cammeo and Osmancik dataset, containing 3810 observations.
The dataset contains seven numerical features and one target variable.
Feature               Description

Area                  Grain area

Perimeter             Grain perimeter

Major_Axis_Length     Length of the major axis

Minor_Axis_Length     Length of the minor axis

Eccentricity          Grain eccentricity

Convex_Area           Grain convex area

Extent                Grain extent

Class                 Rice variety (target)


Technologies and Libraries

Python 3.12
Pandas
NumPy
Scikit-learn
Matplotlib
uv 

Project Structure

rice-binary-classification/
├── data/
│   └── raw/
├── src/
│   └── rice_binary_classification/
│       ├── __init__.py
│       ├── data.py
│       ├── eda.py
│       ├── preprocessing.py
│       ├── model.py
│       ├── evaluation.py
│       ├── predictions.py
│       └── main.py
├── pyproject.toml
├── uv.lock
└── README.md

Methodology

The project followed these stages:

1. Exploratory Data Analysis

-Examined dataset structure and feature distributions.

-Investigated potential outliers.

-Analysed relationships and correlations between features.

-Explored how physical measurements differ across the two rice varieties.

2. Data Preparation

-Separated features from the target variable.

-Split the dataset into training and testing sets using an 80:20 ratio.

-Applied stratified sampling to preserve class proportions.

-Standardised numerical features using StandardScaler.

Fitted the scaler on training data and reused it to transform test data.

3. Model Development

Developed a Logistic Regression classifier as the baseline model.

The model was trained using the seven original features and evaluated on the held-out test dataset.

4. Model Evaluation

Evaluated the baseline model using:

Accuracy

Precision

Recall

F1-score

Confusion matrix

5. Model Improvement and Comparison

Two experiments were conducted:

Feature selection: Removed Convex_Area to investigate the effect of reducing redundancy between highly correlated features.

Algorithm comparison: Implemented a Decision Tree classifier to investigate whether a model capable of capturing non-linear relationships would perform better.

Neither experiment improved upon the baseline Logistic Regression model.

6. Prediction Demonstration

Implemented a prediction pipeline that accepts new rice measurements, applies the fitted scaler, and uses the trained Logistic Regression model to:

Predict the rice variety.

Generate class probabilities for Cammeo and Osmancik.

Results

The baseline Logistic Regression model achieved 91.60% accuracy on the held-out test dataset.

Classification Performance

Metric

Cammeo

Osmancik

Precision

91.72%

91.52%

Recall

88.34%

94.04%

F1-score

90.00%

92.76%

Confusion Matrix

Actual / Predicted

Cammeo

Osmancik

Cammeo

288

38

Osmancik

26

410

Model Comparison

Model

Test Accuracy

Logistic Regression (Baseline)

91.60%

Logistic Regression (Feature Selection)

91.47%

Decision Tree

88.98%

Logistic Regression performed best among the three evaluated configurations and was selected as the final model.

Key Findings

Logistic Regression achieved the highest test accuracy.

The model performed better at identifying Osmancik samples than Cammeo samples in terms of recall.

Removing Convex_Area did not produce a meaningful improvement.

The Decision Tree performed worse than the baseline model across the evaluated metrics.

Running the Project Locally

Prerequisites

Python 3.12

uv

1. Clone the repository

git clone <your-repository-url>
cd rice-binary-classification

2. Install dependencies

uv sync

3. Prepare the dataset

Download the Rice Cammeo and Osmancik dataset and place the CSV file in:

data/raw/

4. Run the project

uv run python -m rice_binary_classification.main

The program executes the classification workflow and displays model evaluation results and predictions on a new rice sample.

Limitations

The model does not achieve perfect classification and still produces incorrect predictions.

Cammeo has lower recall than Osmancik, meaning a greater proportion of actual Cammeo samples are missed.

Several input features are strongly correlated, which can affect the stability and interpretation of individual Logistic Regression coefficients.

Performance has been evaluated using a single stratified train-test split.

Conclusion

This project demonstrates the application of supervised machine learning to a binary classification problem.

Through exploratory data analysis, preprocessing, model development, evaluation, and experimentation, Logistic Regression was identified as the best-performing model among the approaches evaluated.

The final model achieved 91.60% test accuracy and successfully generated predictions for previously unseen rice measurements.
