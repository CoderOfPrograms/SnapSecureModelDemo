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
        errors='ignore'  # Ignore any missing fields instead of raising an error
    )

    # Select only the columns that match the meta fields
    columns_to_include = [
        "id",
        "timeStamp",
        "headPosition.x",
        "headPosition.y",
        "headPosition.z",
        "headRotation.x",
        "headRotation.y",
        "headRotation.z",
    ]

    # Filter the DataFrame to include only the specified columns
    df = df[columns_to_include]

    return df

def extract_features(df):
    if df.empty:
        return None
    # Only select specific columns to match the training data
    selected_columns = [
        "headPosition.x", "headPosition.y", "headPosition.z",
        "headRotation.x", "headRotation.y", "headRotation.z"
    ]
    
    features = []
    for col in selected_columns:
        if col in df.columns:
            if df[col].isna().all():
                return None
            # Compute max, min, mean, std, median for each column
            features.extend([
                df[col].max(),
                df[col].min(),
                df[col].mean(),
                df[col].std(),
                df[col].median()
            ])
    
    # Ensure that the number of features matches the expected number (30)
    return features if len(features) == 30 else None

def predict_single_json(json_file_path, model_path):
    # Load the pre-trained model
    model = joblib.load(model_path)

    # Load and process the JSON file
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    
    if "headControllersMotionRecordList" in json_data:
        data = json_data["headControllersMotionRecordList"]
    elif "headHandsMotionRecordList" in json_data:
        data = json_data["headHandsMotionRecordList"]
    else:
        raise ValueError("JSON file does not contain expected keys 'headControllersMotionRecordList' or 'headHandsMotionRecordList'")

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

    y_true = [id] * len(predictions) if predictions else []
    y_pred = predictions

    # Calculate metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred) if y_true else None
    weighted_f1 = f1_score(y_true, y_pred, average="weighted") if y_true else None

    return {
        "predicted": pred,
        "actual": id,
        "balanced_accuracy": balanced_acc,
        "weighted_f1_score": weighted_f1,
    }


