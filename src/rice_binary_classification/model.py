from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#train baseline mode
def train_baseline_model(X_train_scaled, y_train):
    model = LogisticRegression()
    model.fit(X_train_scaled,y_train)
    return model

#generate predictions
def make_predictions(model, X_test_scaled):
    predictions = model.predict(X_test_scaled)
    return predictions


#Train Decision Tree Algorithm Model
def train_decision_tree(X_train,y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    return model
