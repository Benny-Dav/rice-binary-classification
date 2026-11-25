from rice_binary_classification.model import make_predictions


def predict_rice_variety(model, scaler, new_measurements):
  
    #scale features
    X_scaled = scaler.transform(new_measurements)

    #make predictions
    new_predictions = model.predict(X_scaled)

    #class probabilities
    probabilities = model.predict_proba(X_scaled)
    return new_predictions,probabilities
