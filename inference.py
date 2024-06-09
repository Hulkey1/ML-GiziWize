import joblib
import os
import json

def model_fn(model_dir):
    """Load model from the directory."""
    model = joblib.load(os.path.join(model_dir, "best_rf_model.pkl"))
    return model

def input_fn(request_body, request_content_type):
    """Parse input data payload."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    """Make prediction using the model."""
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    """Format prediction output."""
    if response_content_type == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError("Unsupported content type: {}".format(response_content_type))
