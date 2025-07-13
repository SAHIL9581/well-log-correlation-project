# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import lasio

def load_las_files_from_directory(directory_path):
    """
    Loads all .las files from a directory, robustly handling data issues.
    This is the definitive version that handles all known issues.
    """
    all_well_dfs = []
    print(f"Loading .las files from: {directory_path}")

    for filename in os.listdir(directory_path):
        if filename.endswith('.las'):
            file_path = os.path.join(directory_path, filename)
            try:
                las = lasio.read(file_path)
                # --- FIX 1: Convert to DataFrame immediately ---
                df = las.df()
                
                # --- FIX 2: Reset index immediately to solve duplicate depth issues ---
                # This moves the depth into a column and creates a simple integer index.
                df.reset_index(inplace=True)
                
                # --- FIX 3: Reliably identify and rename the depth column ---
                # The depth is now always the first column.
                depth_col_name = df.columns[0]
                df.rename(columns={depth_col_name: 'DEPTH_MD'}, inplace=True)
                
                # Add the well name from the LAS header
                df['WELL'] = las.well.WELL.value
                # Make columns unique for this DataFrame
                df = make_columns_unique(df)
                all_well_dfs.append(df)
            except Exception as e:
                print(f"Could not read or process {filename}: {e}")
    
    if not all_well_dfs:
        raise ValueError(f"No .las files were successfully processed in {directory_path}")

    # --- FIX 4: Concatenate all dataframes ---
    # This will now work because the indexes are simple and unique integers.
    # The `sort=False` argument is not strictly needed but is good practice.
    master_df = pd.concat(all_well_dfs, ignore_index=True, sort=False)
    print(f"Successfully loaded and combined {len(all_well_dfs)} wells.")
    return master_df

def preprocess_data(df, is_train=True, artifacts_path=None):
    """
    Preprocesses data to predict FORCE_2020_LITHOFACIES_LITHOLOGY.
    """
    print("Starting preprocessing for Lithofacies Prediction...")

    # Set the target and confidence column names
    target_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
    confidence_col = 'FORCE_2020_LITHOFACIES_CONFIDENCE'

    if target_col not in df.columns:
        raise ValueError(f"Critical Error: The target column '{target_col}' was not found.")

    # Filter by confidence to ensure high-quality labels
    if confidence_col in df.columns:
        print(f"Initial data points: {len(df)}")
        df = df[df[confidence_col] == 1].copy()
        print(f"Data points after filtering for confidence=1: {len(df)}")
    
    # Drop rows where the target label is missing
    df.dropna(subset=[target_col], inplace=True)

    # Prepare list of columns to drop
    # Start with known unnecessary columns and the new target/confidence
    cols_to_drop = [
        target_col, confidence_col, 'FORMATION', 'GROUP',
        'SGR', 'ROP', 'DTS', 'DCAL', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO', 'BS', 'DRHO', 'Z_LOC',
        'WELL', 'DEPTH_MD' # These are identifiers, not features
    ]
    
    # Find the actual WELL column (in case it was renamed for uniqueness)
    well_col = None
    for col in df.columns:
        if col.startswith('WELL'):
            well_col = col
            break
    if well_col is None:
        raise ValueError("Critical Error: The 'WELL' column was not found.")

    # Use well_col instead of 'WELL' below
    labels = df[target_col].astype(str)
    well_identifiers = df[well_col]
    
    # Create the features DataFrame by dropping all non-feature columns
    features = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Store feature names for later
    feature_names = features.columns.tolist()
    
    # Fill any remaining NaNs in the feature set
    features.fillna(-999.250, inplace=True)

    # Encode Labels and Scale Features
    if is_train:
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        os.makedirs(artifacts_path, exist_ok=True)
        joblib.dump(scaler, os.path.join(artifacts_path, 'scaler.gz'))
        joblib.dump(encoder, os.path.join(artifacts_path, 'encoder.gz'))
        # Save feature names to ensure test set has same columns
        joblib.dump(feature_names, os.path.join(artifacts_path, 'feature_names.gz'))
        
        print(f"Scaler, Encoder, and Feature List saved to {artifacts_path}")
    else:
        if not all([os.path.exists(os.path.join(artifacts_path, f)) for f in ['scaler.gz', 'encoder.gz', 'feature_names.gz']]):
            raise FileNotFoundError("Artifacts (scaler/encoder/features) not found.")
            
        scaler = joblib.load(os.path.join(artifacts_path, 'scaler.gz'))
        encoder = joblib.load(os.path.join(artifacts_path, 'encoder.gz'))
        train_feature_names = joblib.load(os.path.join(artifacts_path, 'feature_names.gz'))
        
        # Align test columns with train columns
        for col in train_feature_names:
            if col not in features.columns:
                features[col] = -999.250 # Or np.nan, then fillna
        features = features[train_feature_names] # Ensure same order and columns

        known_labels = list(encoder.classes_)
        labels_encoded = [known_labels.index(l) if l in known_labels else -1 for l in labels]
        
        features_scaled = scaler.transform(features)
        print(f"Scaler, Encoder, and Feature List loaded from {artifacts_path}")

    # Create the final DataFrame for this dataset
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names, index=labels.index)
    features_scaled_df['WELL'] = well_identifiers
    
    labels_series = pd.Series(labels_encoded, name=target_col, index=labels.index)
    
    # Filter out any rows that had unknown labels (-1)
    valid_indices = labels_series[labels_series != -1].index
    features_scaled_df = features_scaled_df.loc[valid_indices]
    labels_series = labels_series.loc[valid_indices]

    print("Preprocessing complete.")
    return features_scaled_df, labels_series, scaler, encoder

def make_columns_unique(df):
    cols = df.columns.tolist()
    counts = {}
    new_cols = []
    for col in cols:
        if col in counts:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df