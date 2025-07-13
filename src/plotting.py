# src/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score
import re
import os

def plot_and_save_well_predictions(model, X_test_df, y_test, encoder, output_dir):
    """
    Generates and saves prediction plots for each well in the test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    well_names = X_test_df['WELL'].unique()
    
    print(f"Generating prediction plots for {len(well_names)} wells...")

    for well_name in well_names:
        # Filter data for the current well
        well_mask = X_test_df['WELL'] == well_name
        X_well = X_test_df.loc[well_mask].drop('WELL', axis=1)
        y_well_true = y_test[well_mask]
        
        # Reshape for model prediction (add channel dimension)
        X_well_np = np.expand_dims(X_well.to_numpy(), axis=2).astype('float32')
        
        # Predict
        predictions = model.predict(X_well_np)
        y_well_pred = np.argmax(predictions, axis=1)

        # Calculate accuracy for this well
        acc = accuracy_score(y_well_true, y_well_pred)

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(5, 10), sharey=True)
        
        # Plot Ground Truth
        ax[0].imshow(np.repeat(y_well_true.values.reshape(-1, 1), 100, 1), vmin=0, vmax=len(encoder.classes_)-1, aspect='auto')
        ax[0].set_title('Ground Truth')
        ax[0].set_ylabel("Depth Index")
        ax[0].set_xticks([])

        # Plot Prediction
        im = ax[1].imshow(np.repeat(y_well_pred.reshape(-1, 1), 100, 1), vmin=0, vmax=len(encoder.classes_)-1, aspect='auto')
        ax[1].set_title('Prediction')
        ax[1].set_xticks([])

        # Add a shared colorbar
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="20%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks(range(len(encoder.classes_)))
        cbar.set_ticklabels(encoder.classes_)
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(f'Well: {well_name}\nAccuracy: {acc:.2f}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        safe_well_name = re.sub(r'[ /]', '_', well_name)
        save_path = os.path.join(output_dir, f'{safe_well_name}_prediction.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"  - Saved plot for well {well_name} to {save_path}")