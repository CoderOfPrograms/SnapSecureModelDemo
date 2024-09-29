import json
import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

def sliding_window(df, window_length, overlap):
    step = window_length - overlap
    current_start = df["timeStamp"].min()

    while current_start < df["timeStamp"].max():
        current_end = current_start + window_length
        window_df = df[
            (df["timeStamp"] >= current_start) & (df["timeStamp"] < current_end)
        ]
        yield window_df
        current_start += step

def convert_data_to_df(data):
    df = pd.json_normalize(
        data,
        record_path=None,
        meta=[
            "id",
            "timeStamp",
            ["headPosition", "x"],
            ["headPosition", "y"],
            ["headPosition", "z"],
            ["headRotation", "x"],
            ["headRotation", "y"],
            ["headRotation", "z"],
            ["leftHandPosition", "x"],
            ["leftHandPosition", "y"],
            ["leftHandPosition", "z"],
            ["leftHandRotation", "x"],
            ["leftHandRotation", "y"],
            ["leftHandRotation", "z"],
            ["rightHandPosition", "x"],
            ["rightHandPosition", "y"],
            ["rightHandPosition", "z"],
            ["rightHandRotation", "x"],
            ["rightHandRotation", "y"],
            ["rightHandRotation", "z"],
        ],
    )
    return df

def extract_features(df):
    if df.empty:
        return None
    features = []
    for col in df.columns:
        if col != 'timeStamp':
            if df[col].isna().all():
                return None
            features.extend([
                df[col].max(),
                df[col].min(),
                df[col].mean(),
                df[col].std(),
                df[col].median()
            ])
    return features

def predict_single_json(json_file_path, model_path):
    # Load the pre-trained model
    model = joblib.load(model_path)

    # Load and process the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)["headControllersMotionRecordList"]
    df = convert_data_to_df(data)
    df.sort_values(by=["timeStamp"], inplace=True)
    id = df["id"].iloc[0]
    df.drop("id", axis=1, inplace=True)

    # Extract features and make predictions
    X_test = []
    for window in sliding_window(df, window_length=0.1, overlap=0):
        features = extract_features(window)
        if features is not None:
            X_test.append(features)

    if X_test:
        predictions = model.predict(X_test)
        predictions = predictions.tolist()
        pred = max(set(predictions), key=predictions.count)
    else:
        pred = None

    y_true = [id] * len(predictions)
    y_pred = predictions

    # Calculate metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    return {
        "predicted": pred,
        "actual": id,
        "balanced_accuracy": balanced_acc,
        "weighted_f1_score": weighted_f1,
    }

# Usage example
result = predict_single_json("/content/Wave/test/VrGesturesStudy_HeadAndControllers_WaveControllerStand10_2024-07-16_11-00-37.json", "NodTripleThreat.pkl")
print(result)
