import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import os

# Define the MLP class (must be identical to the one used for training)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def transform(df, scaler, columns, glucose_median):
    df = df.copy()

    df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
    df['admission_month'] = df['admission_date'].dt.month
    df['admission_month'] = df['admission_month'].fillna(df['admission_month'].median())

    df['glucose_missing'] = df['glucose_level_mgdl'].isnull().astype(int)
    df['glucose_level_mgdl'] = df['glucose_level_mgdl'].fillna(glucose_median)

    df.loc[df['blood_pressure_systolic'] < 50, 'blood_pressure_systolic'] = np.nan
    df['blood_pressure_systolic'] = df['blood_pressure_systolic'].fillna(
        df['blood_pressure_systolic'].median()
    )
    
    patient_ids = df['patient_id']
    df['high_comorbidity'] = (df['charlson_comorbidity_index'] >= 4).astype(int)

    df['length_of_stay_days'] = np.log1p(df['length_of_stay_days'])
    df['prior_admissions_1yr'] = np.log1p(df['prior_admissions_1yr'])
    df['n_medications_discharge'] = np.log1p(df['n_medications_discharge'])
    df = df.fillna(0)

    df = df.drop(columns=['patient_id', 'admission_date'])

    X = pd.get_dummies(df)
    X = X.reindex(columns=columns, fill_value=0)

    return scaler.transform(X), patient_ids

def main():
    parser = argparse.ArgumentParser(description='Run inference for 30-day readmission prediction.')
    
    # Fixed: Use flags starting with '--' to allow 'default' and 'help' to work correctly
    parser.add_argument('--input', type=str, default='../data/test.csv', 
                        help='Path to the input CSV file for prediction.')
    parser.add_argument('--output', type=str, default='../data/predictions.csv', 
                        help='Path to save the predictions CSV file.')
    
    args = parser.parse_args()

    # Using os.path.join is safer for combining directory paths and filenames
    artifact_dir = '/home/growlt367/Downloads/induction/Week 7/readmission-dl/models'
    
    print(f"Loading input data from: {args.input}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found at {args.input}")
        
    input_df = pd.read_csv(args.input)

    print("Loading preprocessing and model artifacts...")
    with open(os.path.join(artifact_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(artifact_dir, 'columns.pkl'), 'rb') as f:
        columns = pickle.load(f)
    with open(os.path.join(artifact_dir, 'glucose_median.pkl'), 'rb') as f:
        glucose_median = pickle.load(f)
    with open(os.path.join(artifact_dir, 'model_paths_for_ensemble.pkl'), 'rb') as f:
        model_paths_for_ensemble = pickle.load(f)
    with open(os.path.join(artifact_dir, 'calibrator.pkl'), 'rb') as f:
        calibrator = pickle.load(f)
    with open(os.path.join(artifact_dir, 'threshold.pkl'), 'rb') as f:
        best_t = pickle.load(f)

    print("Preprocessing input data...")
    X_transformed, patient_ids = transform(input_df, scaler, columns, glucose_median)
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)

    print("Performing ensembled predictions...")
    all_fold_predictions = []
    input_dim = len(columns)
    
    for model_path in model_paths_for_ensemble:
        # Extract just the filename (e.g., 'model_fold_1.pt') from the Colab path
        model_filename = os.path.basename(model_path)
        # Reconstruct the path using your local artifact directory
        local_model_path = os.path.join(artifact_dir, model_filename)
        
        print(f"Loading model from: {local_model_path}")
        
        model = MLP(input_dim)
        model.load_state_dict(torch.load(local_model_path, weights_only=True))
        model.eval()

        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).numpy().flatten()
            all_fold_predictions.append(probs)

    ensembled_probs = np.mean(all_fold_predictions, axis=0)

    print("Applying Platt scaling and decision threshold...")
    calibrated_probs = calibrator.predict_proba(ensembled_probs.reshape(-1, 1))[:, 1]
    predictions = (calibrated_probs > best_t).astype(int)

    output_df = pd.DataFrame({
        'patient_id': patient_ids,
        'readmitted_30d_prediction': predictions,
        'readmitted_30d_probability': calibrated_probs
    })

    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")

if __name__ == '__main__':
    main()