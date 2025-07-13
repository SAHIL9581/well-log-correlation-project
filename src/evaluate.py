# src/evaluate.py
import argparse
import mlflow
import os
import shutil
import joblib
import numpy as np

# We use the same preprocessing functions
from data_preprocessing import load_las_files_from_directory, preprocess_data
from plotting import plot_and_save_well_predictions

def evaluate(run_id, test_dir):
    """
    Loads a model from an MLflow run and evaluates it by generating prediction plots.
    """
    print(f"Evaluating model from MLflow Run ID: {run_id}")
    
    # --- 1. Load Model and Artifacts from MLflow ---
    # Load the model using its URI (Uniform Resource Identifier)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.tensorflow.load_model(model_uri)
    
    # Create a temporary directory to download artifacts into
    local_dir = "temp_downloaded_artifacts"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # Download the scaler and encoder that were saved during training
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id, "scaler.gz", local_dir)
    client.download_artifacts(run_id, "encoder.gz", local_dir)
    client.download_artifacts(run_id, "feature_names.gz", local_dir)
    
    # --- 2. Load and Preprocess Test Data using loaded artifacts ---
    df_test = load_las_files_from_directory(test_dir)

    # We use the same preprocessing function, but in 'evaluation' mode (is_train=False)
    # The artifacts_path points to our downloaded artifacts, ensuring consistency.
    X_test_df, y_test, _, encoder = preprocess_data(
        df_test, is_train=False, artifacts_path=local_dir
    )

    # --- 3. Generate and Save Plots ---
    output_dir = "outputs"
    # Use the plotting function you already have!
    plot_and_save_well_predictions(model, X_test_df, y_test, encoder, output_dir)
    
    # --- 4. Log the entire output directory back to the original MLflow run ---
    print("Logging evaluation plots back to the original MLflow run...")
    # We start the *same run* again to add more artifacts to it
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(output_dir, artifact_path="evaluation_plots")
        
    print(f"\nEvaluation complete. Plots saved locally in '{output_dir}' and logged to MLflow.")
    
    # --- 5. Clean up temporary files ---
    shutil.rmtree(local_dir)
    shutil.rmtree(output_dir) # Optional: remove local plots after logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="MLflow Run ID of the trained model to evaluate.")
    parser.add_argument("--test-data-dir", type=str, required=True, help="Path to the directory of test .las files.")
    args = parser.parse_args()
    evaluate(args.run_id, args.test_data_dir)