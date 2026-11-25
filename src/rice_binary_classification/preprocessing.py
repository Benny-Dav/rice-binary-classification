from sklearn.model_selection import train_test_split
from rice_binary_classification.data import load_data
from sklearn.preprocessing import StandardScaler

#Feature and Target Separation
def separate_features_target(df):
    X = df.drop(columns = ["Class"])
    y = df["Class"]
    return X,y

#Train-Test Data Split
def split_data(X,y):
    return train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#Feature Scaling Pipeline
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

#Experiment 1 - Removing redunant features 
def select_features(X):
    X_select1=X.drop(columns = ["Convex_Area"])
    return X_select1

