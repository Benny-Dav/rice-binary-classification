from rice_binary_classification.data import load_data
from rice_binary_classification.evaluation import baseline_class_report, examine_misclassifications, gen_confusion_matrix
from rice_binary_classification.model import make_predictions, train_baseline_model, train_decision_tree
from rice_binary_classification.predictions import predict_rice_variety
from rice_binary_classification.preprocessing import scale_features, select_features, separate_features_target, split_data
import pandas as pd

def main():
    #1. load dataset
    df = load_data()

    #2. separate features (X) from target (y)
    X,y = separate_features_target(df)

    #3. split data into test and training set
    X_train, X_test, y_train, y_test = split_data(X,y)

    #4. scale training and test features
    X_train_scaled, X_test_scaled, baseline_scaler = scale_features(X_train,X_test)

    #5. train baseline model
    baseline_model = train_baseline_model(X_train_scaled,y_train)
    #print("MODEL : ",  model)
    #print(model.coef_)
    #print(model.intercept_)

    #6. generate predictions using test features
    predictions = make_predictions(baseline_model, X_test_scaled)
    predictions_df = pd.DataFrame({
        "Actual ":y_test,
        "Predicted ":predictions
    })

    #examine prediction probabilities
    probabilities = baseline_model.predict_proba(X_test_scaled)
    #print("MODEL CLASSES : ", model.classes_)
    #print("PREDICTION PROBABILITIES")
    #print(probabilities[:5])

    #7. generate eval - classification report on baseline model
    classification_report = baseline_class_report(y_test, predictions)
    report_df = pd.DataFrame(classification_report).transpose()
    report_df.loc["accuracy"] = {
        "precision":None,
        "recall":None,
        "f1-score":classification_report["accuracy"],
        "support":len(y_test)
    }
    #print("PERFORMANCE METRICS FOR BASELINE MODEL \n")
    #print(report_df)

    #8. confusion matrix
    matrix = gen_confusion_matrix(y_test,predictions)
    #print(matrix)

    #9. Examine misclassifications
    misclassifications = examine_misclassifications(X_test,y_test,predictions)
    #print(misclassifications)
    #print(misclassifications["Actual"].value_counts())
    #print(misclassifications.groupby("Actual").mean(numeric_only = True))

    #Model Improvement
    X_select1 = select_features(X) #run feature selection
    X_train, X_test, y_train, y_test = split_data(X_select1,y) #data splitting
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train,X_test) #feature scaling
    model = train_baseline_model(X_train_scaled,y_train) #model training
    predictions = make_predictions(model, X_test_scaled) #generate predictions
    classification_report = baseline_class_report(y_test, predictions)
    report_df = pd.DataFrame(classification_report).transpose()
    report_df.loc["accuracy"] = {
        "precision-":None,
        "recall":None,
        "f1-score":classification_report["accuracy"],
        "support":len(y_test)
    }
    #print("PERFORMANCE METRICS FOR IMPROVED MODEL (FEATURE SELECTION EXPERIMENT 1) \n")
    #print(report_df)

    #train decision tree model
    decision_tree_model = train_decision_tree(X_train,y_train)
    #generate predictions
    predictions = make_predictions(decision_tree_model,X_test)
    #eval
    classification_report = baseline_class_report(y_test, predictions)
    report_df = pd.DataFrame(classification_report).transpose()
    report_df.loc["accuracy"] = {
        "precision":None,
        "recall":None,
        "f1-score":classification_report["accuracy"],
        "support":len(y_test)
    }
    #print("PERFORMANCE METRICS FOR DECISION TREE ALGORITHM MODEL \n")
    #print(report_df)


    new_measurements = pd.DataFrame([{
    "Area": 12500,
    "Perimeter": 445,
    "Major_Axis_Length": 185,
    "Minor_Axis_Length": 85,
    "Eccentricity": 0.89,
    "Convex_Area": 12800,
    "Extent": 0.65
}])

    #make new predictions on unseen data
    new_predictions,probabilities = predict_rice_variety(baseline_model,baseline_scaler,new_measurements)
    print("PREDICTED VARIETIES \n")
    print(new_predictions[0])
    print("CLASS PROBABILITIES \n")
    print(probabilities)
if __name__ == "__main__":
    main()
