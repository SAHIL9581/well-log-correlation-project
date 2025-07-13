# src/train.py
import argparse
import mlflow
import mlflow.tensorflow
import numpy as np
import os
import shutil
import joblib
from sklearn.metrics import classification_report, accuracy_score

from data_preprocessing import load_las_files_from_directory, preprocess_data
from model import classification_model

def train(train_dir, test_dir, epochs, batch_size):
    # Set a name for the MLflow experiment. If it doesn't exist, it will be created.
    mlflow.set_experiment("Well_Log_Facies_Prediction")

    # Start an MLflow run. Everything inside this 'with' block will be logged.
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.set_tag("mlflow.runName", "CNN_Training_Run")
        
        # --- 1. Load and Preprocess Data ---
        print("Loading training data from .las files...")
        df_train = load_las_files_from_directory(train_dir)
        
        # Create a temporary directory for artifacts that will be logged
        artifacts_path = f"temp_artifacts_{run_id}"
        
        X_train_df, y_train, scaler, encoder = preprocess_data(
            df_train, is_train=True, artifacts_path=artifacts_path
        )
        X_train = X_train_df.drop('WELL', axis=1).to_numpy()
        X_train_reshaped = np.expand_dims(X_train, axis=2).astype('float32')
        y_train_np = y_train.to_numpy().astype('int32')

        # --- LOG PARAMETERS ---
        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_classes": len(encoder.classes_),
            "training_samples": len(X_train_reshaped),
            "input_shape": str(X_train_reshaped.shape[1:])
        }
        mlflow.log_params(params)
        print(f"Logged Parameters: {params}")

        # --- 2. Define and Train Model ---
        print("Defining and training model...")
        model = classification_model(
            input_shape=X_train_reshaped.shape[1:], 
            num_classes=params["num_classes"]
        )
        
        # Use MLflow's autologging callback for Keras to track metrics per epoch
        mlflow_callback = mlflow.keras.MlflowCallback(run)
        
        model.fit(
            X_train_reshaped, y_train_np,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_split=0.2,
            callbacks=[mlflow_callback]
        )
        
        # --- 3. Evaluate on Test Set ---
        print("Evaluating model on test set...")
        df_test = load_las_files_from_directory(test_dir)
        X_test_df, y_test, _, _ = preprocess_data(
            df_test, is_train=False, artifacts_path=artifacts_path
        )
        # Drop wells from test set that have no valid labels after filtering
        X_test_df = X_test_df[X_test_df.index.isin(y_test.index)]

        X_test = X_test_df.drop('WELL', axis=1).to_numpy()
        X_test_reshaped = np.expand_dims(X_test, axis=2).astype('float32')
        y_test_np = y_test.to_numpy().astype('int32')

        y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
        
        # --- LOG METRICS ---
        test_accuracy = accuracy_score(y_test_np, y_pred)
        mlflow.log_metric("final_test_accuracy", test_accuracy)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")

        # Log classification report as a text artifact
        report_str = classification_report(y_test_np, y_pred, target_names=encoder.classes_, zero_division=0)
        mlflow.log_text(report_str, "classification_report.txt")

        # --- 4. Log Artifacts ---
        print("Logging model and preprocessing artifacts to MLflow...")
        # The model is logged in a format that MLflow understands
        mlflow.tensorflow.log_model(model, "model")
        # Log the scaler, encoder, and feature names for use in evaluation
        mlflow.log_artifact(os.path.join(artifacts_path, 'scaler.gz'))
        mlflow.log_artifact(os.path.join(artifacts_path, 'encoder.gz'))
        mlflow.log_artifact(os.path.join(artifacts_path, 'feature_names.gz'))
        
        # Clean up the temporary artifacts directory
        shutil.rmtree(artifacts_path)
        
        print("\nTraining run finished successfully.")
        print(f"View your run in the MLflow UI: http://127.0.0.1:5000")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-dir", type=str, required=True, help="Path to directory of training .las files.")
    parser.add_argument("--test-data-dir", type=str, required=True, help="Path to directory of testing .las files.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    args = parser.parse_args()
    
    train(args.train_data_dir, args.test_data_dir, args.epochs, args.batch_size)