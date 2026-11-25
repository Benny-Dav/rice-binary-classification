from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd

#calculate accuracy of baseline model
def baseline_accuracy(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

#generate classification report for baseline model
def baseline_class_report(y_test, predictions):
    baseline_classification_report = classification_report(y_test, predictions,output_dict=True)
    return baseline_classification_report

#confusion matrix
def gen_confusion_matrix(y_test, predictions):
    matrix = confusion_matrix(y_test, predictions)
    return matrix

#investigate misclassified observations
def examine_misclassifications(X_test, y_test, predictions):
    misclassifications=[]
    for (_,X_value),y_value,prediction in zip(X_test.iterrows(),y_test,predictions):
        if y_value != prediction:
            misclassifications.append(list(X_value)+ [y_value,prediction])
    columns = list(X_test.columns) + ["Actual","Predicted"]

    return pd.DataFrame(misclassifications,columns=columns)