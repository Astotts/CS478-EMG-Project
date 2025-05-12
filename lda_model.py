import asyncio
import h5py
import pandas as pd
import numpy as np
# Removed: import tensorflow as tf
from sklearn.model_selection import train_test_split
# Removed: from sklearn.preprocessing import LabelEncoder - Not used currently
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt # For visualization
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns # For nice confusion matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Added LDA
# Removed: from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm # Import tqdm
import joblib # For saving sklearn models (optional)

# Synchronous helper for blocking HDF5 read (remains the same)
def _load_emg_data(file_path):
    """Loads EMG data from an HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Ensure keys are integers for consistency if needed later
            return {int(key): np.array(f[key]) for key in f.keys()}
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading HDF5 file {file_path}: {str(e)}")
        return None

# Asynchronous data loading function (remains the same)
async def load_data_async(subject_path):
    """Loads EMG data and trial labels asynchronously."""
    loop = asyncio.get_running_loop()
    hdf5_path = f"{subject_path}/emg_data.hdf5"
    csv_path = f"{subject_path}/trials.csv"

    # Load EMG data asynchronously using executor
    emg_data_task = loop.run_in_executor(None, _load_emg_data, hdf5_path)
    # Load labels asynchronously using executor
    trials_df_task = loop.run_in_executor(None, pd.read_csv, csv_path)

    # Await both tasks concurrently
    emg_data, trials_df = await asyncio.gather(emg_data_task, trials_df_task)

    # Handle potential loading errors
    if emg_data is None or trials_df is None:
        raise FileNotFoundError(f"Data files not found or failed to load in {subject_path}")

    if not emg_data:
        print(f"Warning: No EMG data loaded from {hdf5_path}")
        return np.array([]), (np.array([]), np.array([]))

    try:
        min_length = min(data.shape[1] for data in emg_data.values() if data.ndim == 2 and data.shape[1] > 0)
    except ValueError:
        print(f"Warning: Could not determine minimum length in {subject_path}. Skipping.")
        return np.array([]), (np.array([]), np.array([]))


    X = []
    y_grasp = []
    y_position = []

    for trial_idx in range(150):
        try:
            if trial_idx not in emg_data:
                continue

            trial_info = trials_df.iloc[trial_idx]
            current_signal = emg_data[trial_idx]

            if current_signal.ndim != 2 or current_signal.shape[1] < min_length:
                print(f"Warning: Invalid signal shape {current_signal.shape} for trial {trial_idx} in {subject_path}. Skipping.")
                continue

            standardized_signal = current_signal[:, :min_length]
            X.append(standardized_signal)

            y_grasp.append(trial_info['grasp'] - 1)
            y_position.append(trial_info['target_position'] - 1)

        except IndexError:
             print(f"Error: Trial index {trial_idx} out of bounds for trials.csv in {subject_path}")
             continue
        except KeyError:
             print(f"Error: Trial key {trial_idx} not found in emg_data dict for {subject_path}")
             continue
        except Exception as e:
             print(f"Error processing trial {trial_idx} in {subject_path}: {str(e)}")
             continue

    if not X:
        print(f"Warning: No valid trials processed for {subject_path}")
        return np.array([]), (np.array([]), np.array([]))

    return np.array(X), (np.array(y_grasp), np.array(y_position))


# Async version of processing a single block (remains the same structure)
async def process_subject_block_async(args):
    """Asynchronously loads and processes data for one subject block."""
    subject, day, block = args
    path = f"C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data/participant_{subject}/participant{subject}_day{day}_block{block}"

    try:
        X, (y_grasp, y_position) = await load_data_async(path)

        if X.size == 0:
            print(f"Skipping {path} due to empty data.")
            return None

        # Perform CPU-bound processing synchronously after await
        processed_X = preprocess_emg(X)
        features = extract_features(processed_X)
        return features, y_grasp, y_position
    except FileNotFoundError as e:
        print(f"Skipping {path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None


# --- Synchronous CPU-Bound Functions (Remain Unchanged) ---

def preprocess_emg(emg_signals, fs=2000):
    """
    Process raw EMG signals: Rectification, Smoothing, Normalization.
    """
    processed_signals = []
    for signal in emg_signals:
        if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
            print("Warning: Skipping invalid signal shape in preprocess_emg.")
            continue

        rectified = np.abs(signal)
        b, a = butter(4, 20/(fs/2), 'low')
        smoothed = filtfilt(b, a, rectified, axis=1)

        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        normalized = np.divide(smoothed - mean, std + 1e-6, out=np.zeros_like(smoothed), where=std!=0)

        processed_signals.append(normalized)

    if not processed_signals:
        return np.array([])
    return np.array(processed_signals)


def extract_features(processed_signals, window_size=150, overlap=0.5):
    """
    Extract time-domain and frequency-domain features using sliding windows.
    """
    features = []
    if processed_signals.size == 0:
        return np.array(features)

    for signal in processed_signals:
        if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] < window_size:
            continue

        num_channels, signal_length = signal.shape
        step_size = int(window_size * (1 - overlap))
        if step_size <= 0: step_size = 1
        num_windows = (signal_length - window_size) // step_size + 1

        if num_windows <= 0:
            continue

        window_features = []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            window = signal[:, start:end]

            if window.shape[1] != window_size: continue

            mean_abs_val = np.mean(np.abs(window), axis=1)
            var = np.var(window, axis=1)
            rms = np.sqrt(np.mean(window**2, axis=1))
            waveform_length = np.sum(np.abs(np.diff(window, axis=1)), axis=1)
            zero_crossings = np.sum(np.diff(np.sign(window), axis=1) != 0, axis=1) / window_size

            fft_val = np.fft.fft(window, axis=1)
            fft_abs = np.abs(fft_val[:, :window_size//2])
            mean_freq = np.mean(fft_abs, axis=1)
            freq_std = np.std(fft_abs, axis=1)

            window_feature = np.concatenate([
                mean_abs_val, var, rms, waveform_length, zero_crossings,
                mean_freq, freq_std
            ])
            window_features.append(window_feature)

        if window_features:
            features.append(np.mean(window_features, axis=0))

    return np.array(features)


# --- UPDATED: Evaluation function for sklearn models ---
def evaluate_model(model, X_test, y_test, labels, task_name):
    """
    Evaluates a trained scikit-learn classifier model on test data and displays results.

    Args:
        model: The trained scikit-learn classifier (e.g., LDA).
        X_test: Test features.
        y_test: True test labels (integer format).
        labels: A list or array of string labels corresponding to the classes
                (e.g., ["Grasp 0", "Grasp 1"] or ["Pos 0", "Pos 1"]).
        task_name: A string name for the task (e.g., "Grasp", "Position")
                   used for titling outputs.
    """
    print(f"\n--- Evaluating {task_name} Model ---")

    # Evaluate the model (accuracy)
    test_acc = model.score(X_test, y_test)
    print(f"{task_name} Test Accuracy: {test_acc:.4f}")
    # Note: LDA doesn't have a direct 'loss' concept like NNs during evaluation

    # Generate predictions (direct class labels)
    y_pred_classes = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)*0.8))) # Adjust size dynamically
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f'{task_name} Task - LDA - Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Classification Report
    print(f"\n--- {task_name} Classification Report (LDA) ---")
    try:
        print(classification_report(y_test, y_pred_classes, target_names=labels))
    except ValueError as e:
        print(f"Warning generating classification report for {task_name}: {e}")
        print("Ensure 'labels' provided match the classes predicted.")
        # Fallback without target names
        print(classification_report(y_test, y_pred_classes))

# --- REMOVED Keras Model Creation Functions ---
# def create_grasp_model(...): ...
# def create_position_model(...): ...

def verify_labels(trials_file_path):
    """Verifies label ranges and distribution."""
    try:
        df = pd.read_csv(trials_file_path)
        print(f"\nVerifying labels in: {trials_file_path}")
        print("Unique grasp values:", sorted(df['grasp'].unique()))
        print("Unique position values:", sorted(df['target_position'].unique()))
        print("Label distribution (Grasp, Position):")
        print(df[['grasp', 'target_position']].value_counts().sort_index())
    except FileNotFoundError:
        print(f"Label verification skipped: {trials_file_path} not found.")
    except Exception as e:
        print(f"Error verifying labels in {trials_file_path}: {str(e)}")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Added for potential DataFrame use later if needed
import seaborn as sns # Added for potentially nicer plots

def display_lda_metrics_and_plots(lda_model, X_test_scaled, y_test, labels, task_name):
    """
    Displays LDA-specific metrics and visualizations.

    Args:
        lda_model: The fitted scikit-learn LinearDiscriminantAnalysis model.
        X_test_scaled: Scaled test features (used for transformation).
        y_test: True test labels (integer format).
        labels: A list or array of string labels corresponding to the classes.
        task_name: A string name for the task (e.g., "Grasp", "Position").
    """
    print(f"\n--- LDA Specific Analysis: {task_name} Task ---")

    n_components = lda_model.n_components_
    print(f"Number of Linear Discriminants used: {n_components}")

    # 1. Explained Variance Ratio
    if hasattr(lda_model, 'explained_variance_ratio_'):
        explained_variance = lda_model.explained_variance_ratio_
        print(f"Explained Variance Ratio by LD: {explained_variance}")

        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Linear Discriminants')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.title(f'{task_name} - Explained Variance by Linear Discriminant')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.show()
    else:
        print("Explained variance ratio not available (solver might not compute it, e.g., 'lsqr' without shrinkage).")
        explained_variance = None # Set to None if not available


    # 2. Transform data to the LDA space
    try:
        X_lda = lda_model.transform(X_test_scaled)
    except Exception as e:
        print(f"Error transforming data with LDA model: {e}")
        return # Cannot proceed with plots if transformation fails

    print(f"Shape of data after LDA transformation: {X_lda.shape}")


    # 3. Scatter Plot of Data on First Two Linear Discriminants
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_test, cmap='viridis', alpha=0.7, edgecolors='k', s=50)

        # Create legend
        handles, _ = scatter.legend_elements(prop='colors')
        plt.legend(handles, labels, title="Classes")

        plt.xlabel('Linear Discriminant 1 (LD1)')
        plt.ylabel('Linear Discriminant 2 (LD2)')

        if explained_variance is not None and len(explained_variance) >= 2:
             title = f'{task_name} - Test Data Projected onto First Two LDs\n(Explains {explained_variance[0]+explained_variance[1]:.2%} of variance)'
        else:
             title = f'{task_name} - Test Data Projected onto First Two LDs'
        plt.title(title)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.show()

    elif n_components == 1:
        print("\nOnly 1 Linear Discriminant. Plotting 1D distribution.")
        plt.figure(figsize=(10, 6))
        # Create a DataFrame for easier plotting with seaborn
        df_lda = pd.DataFrame({'LD1': X_lda[:, 0], 'Class': [labels[i] for i in y_test]})

        # Histogram or Density Plot
        sns.histplot(data=df_lda, x='LD1', hue='Class', kde=True, palette='viridis', bins=30)
        # Or use sns.kdeplot(data=df_lda, x='LD1', hue='Class', fill=True, palette='viridis')

        plt.xlabel('Linear Discriminant 1 (LD1)')
        plt.ylabel('Density / Count')

        if explained_variance is not None:
             title = f'{task_name} - Test Data Projected onto LD1\n(Explains {explained_variance[0]:.2%} of variance)'
        else:
             title = f'{task_name} - Test Data Projected onto LD1'
        plt.title(title)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.show()
    else:
         print("No components found by LDA or transformation failed. Cannot plot projected data.")

# --- How to integrate into your main script ---

# In the main_async function, after training and evaluating with evaluate_model,
# add calls to the new function:

async def main_async():
    """Main async pipeline for loading, processing, and training TWO SEPARATE LDA models."""
    tasks_to_process = []
    for subject in range(1, 9): # Subjects 1-8
        for day in [1, 2]:
            for block in [1, 2]:
                tasks_to_process.append((subject, day, block))

    all_X = []
    all_y_grasp = []
    all_y_position = []

    async_tasks = [process_subject_block_async(task_info) for task_info in tasks_to_process]

    print(f"Starting processing for {len(async_tasks)} blocks...")
    for future in tqdm(asyncio.as_completed(async_tasks), total=len(async_tasks), desc="Processing Blocks"):
        result = await future
        if result is not None:
            features, y_grasp, y_position = result
            if features.size > 0:
                all_X.append(features)
                all_y_grasp.extend(y_grasp)
                all_y_position.extend(y_position)
            else:
                print("Warning: Received empty features from a processed block.")

    if not all_X:
        print("Error: No data loaded successfully. Exiting.")
        return None, None # Return None for the two models

    print(f"\nLoaded data shapes before concatenate: {[x.shape for x in all_X[:5]]}...")

    try:
        all_X = np.concatenate(all_X, axis=0)
    except ValueError as e:
        print(f"Error concatenating features: {e}")
        for i, arr in enumerate(all_X):
            print(f"Shape of all_X[{i}]: {arr.shape}")
        return None, None # Return None for the two models

    all_y_grasp = np.array(all_y_grasp)
    all_y_position = np.array(all_y_position)

    print(f"Total samples loaded: {all_X.shape[0]}")
    print(f"Feature shape per sample: {all_X.shape[1:]}")
    print(f"Grasp labels shape: {all_y_grasp.shape}")
    print(f"Position labels shape: {all_y_position.shape}")

    print("\nOverall Label Verification (0-based):")
    print("Unique grasp labels:", sorted(np.unique(all_y_grasp)))
    print("Unique position labels:", sorted(np.unique(all_y_position)))

    # ===== TRAINING PROCESS (Synchronous, Updated for LDA Models) =====
    if all_X.shape[0] == 0:
        print("No data available for training. Exiting.")
        return None, None

    # !!!!!!!!!!!!!!!!!! FIX: ADD THIS BLOCK BACK !!!!!!!!!!!!!!!!!!
    # Split data maintaining both labels (Train/Test only for LDA)
    # Consider adjusting test_size if needed (e.g., 0.25 or 0.3)
    (X_train, X_test,
     y_grasp_train, y_grasp_test,
     y_position_train, y_position_test) = train_test_split(
        all_X, all_y_grasp, all_y_position,
        test_size=0.2,
        random_state=42,
        stratify=all_y_grasp # Stratify by grasp type (or position if preferred)
    )

    print(f"\nDataset sizes:")
    print(f"Train: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Standardize features (important for LDA)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Now X_train exists and can be used
    # Use the SAME scaler fitted on train data to transform test data
    X_test = scaler.transform(X_test) # Now X_test exists and can be used

    # Determine number of classes
    num_grasp_classes = len(np.unique(all_y_grasp))
    num_position_classes = len(np.unique(all_y_position))

    print(f"\nNumber of grasp classes: {num_grasp_classes}")
    print(f"Number of position classes: {num_position_classes}")


    # --- Create the two separate LDA models ---
    print("\nCreating LDA Models...")
    lda_grasp_model = LinearDiscriminantAnalysis(solver='svd') # Using svd as before
    lda_position_model = LinearDiscriminantAnalysis(solver='svd')

    # --- Train LDA Models ---
    print("\n--- Starting LDA Grasp Model Training ---")
    lda_grasp_model.fit(X_train, y_grasp_train)
    print("--- LDA Grasp Model Training Finished ---")

    print("\n--- Starting LDA Position Model Training ---")
    lda_position_model.fit(X_train, y_position_train)
    print("--- LDA Position Model Training Finished ---")


    # ===== Evaluation (Separate LDA Models) =====
    grasp_labels = [f"Grasp {i}" for i in range(num_grasp_classes)]
    position_labels = [f"Pos {i}" for i in range(num_position_classes)]

    # --- Standard Evaluation ---
    evaluate_model(
        model=lda_grasp_model,
        X_test=X_test, # Pass scaled X_test
        y_test=y_grasp_test,
        labels=grasp_labels,
        task_name="Grasp"
    )
    evaluate_model(
        model=lda_position_model,
        X_test=X_test, # Pass scaled X_test
        y_test=y_position_test,
        labels=position_labels,
        task_name="Position"
    )

    # --- LDA-Specific Evaluation and Plots ---
    display_lda_metrics_and_plots(
        lda_model=lda_grasp_model,
        X_test_scaled=X_test, # Pass scaled X_test
        y_test=y_grasp_test,
        labels=grasp_labels,
        task_name="Grasp"
    )
    display_lda_metrics_and_plots(
        lda_model=lda_position_model,
        X_test_scaled=X_test, # Pass scaled X_test
        y_test=y_position_test,
        labels=position_labels,
        task_name="Position"
    )

    # Return the trained LDA models
    return lda_grasp_model, lda_position_model

# Run the async main function (UPDATED return handling)
if __name__ == "__main__":
    # --- Removed GPU Configuration ---

    # Run the main async function and unpack results
    grasp_lda_model, position_lda_model = asyncio.run(main_async())

    if grasp_lda_model and position_lda_model: # Check if both models were created and returned
        print("\nAsync pipeline finished successfully.")
        print("Trained LDA Grasp Model and Position Model are available.")
        # Optional: Save the trained LDA models
        try:
            joblib.dump(grasp_lda_model, 'grasp_lda_model.pkl')
            joblib.dump(position_lda_model, 'position_lda_model.pkl')
            print("LDA models saved to grasp_lda_model.pkl and position_lda_model.pkl")
        except Exception as e:
            print(f"Error saving LDA models: {e}")
    else:
        print("\nAsync pipeline did not complete successfully or loaded no data.")

