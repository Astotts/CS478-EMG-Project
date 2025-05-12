import asyncio
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from tqdm import tqdm
import joblib

# Synchronous helper for blocking HDF5 read
def _load_emg_data(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            return {int(key): np.array(f[key]) for key in f.keys()}
    except FileNotFoundError:
        # print(f"Error: HDF5 file not found at {file_path}") # Quieter for many calls
        return None
    except Exception as e:
        print(f"Error loading HDF5 file {file_path}: {str(e)}")
        return None

# Asynchronous data loading function
async def load_data_async(subject_path):
    loop = asyncio.get_running_loop()
    hdf5_path = f"{subject_path}/emg_data.hdf5"
    csv_path = f"{subject_path}/trials.csv"

    emg_data_task = loop.run_in_executor(None, _load_emg_data, hdf5_path)
    # Wrap pd.read_csv in a try-except block for run_in_executor
    async def _read_csv_async(path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            # print(f"Error: CSV file not found at {path}") # Quieter
            return None
        except Exception as e:
            print(f"Error loading CSV file {path}: {str(e)}")
            return None
    trials_df_task = loop.run_in_executor(None, lambda p: pd.read_csv(p, on_bad_lines='skip'), csv_path)


    emg_data = await emg_data_task
    trials_df = await trials_df_task # This now correctly awaits the lambda execution


    if emg_data is None or trials_df is None:
        # print(f"Warning: Data files not found or failed to load in {subject_path}")
        raise FileNotFoundError(f"Data files not found or failed to load in {subject_path}")


    if not emg_data:
        # print(f"Warning: No EMG data loaded (empty dict) from {hdf5_path}")
        return np.array([]), (np.array([]), np.array([]))

    valid_lengths = [data.shape[1] for data in emg_data.values() if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] > 0]
    if not valid_lengths:
        # print(f"Warning: No valid EMG signals to determine min length in {subject_path}.")
        return np.array([]), (np.array([]), np.array([]))
    min_length = min(valid_lengths)

    X = []
    y_grasp = []
    y_position = []

    for trial_idx in range(150): # Max expected trials based on previous scripts
        try:
            if trial_idx not in emg_data:
                continue
            if trial_idx >= len(trials_df):
                # print(f"Warning: Trial index {trial_idx} exceeds CSV rows ({len(trials_df)}) in {subject_path}.")
                continue

            trial_info = trials_df.iloc[trial_idx]
            current_signal = emg_data[trial_idx]

            if not isinstance(current_signal, np.ndarray) or current_signal.ndim != 2 or current_signal.shape[1] < min_length:
                continue

            standardized_signal = current_signal[:, :min_length]
            X.append(standardized_signal)
            y_grasp.append(int(trial_info['grasp']) - 1)
            y_position.append(int(trial_info['target_position']) - 1)
        except (IndexError, KeyError, ValueError) as e: # Catch common errors
            # print(f"Warning: Error processing trial {trial_idx} in {subject_path}: {e}. Skipping.")
            continue
        except Exception as e: # Catch any other unexpected error for a trial
            print(f"Unexpected error processing trial {trial_idx} in {subject_path}: {str(e)}. Skipping.")
            continue


    if not X:
        # print(f"Warning: No valid trials processed into X for {subject_path}")
        return np.array([]), (np.array([]), np.array([]))

    return np.array(X), (np.array(y_grasp), np.array(y_position))

# Async version of processing a single block
async def process_subject_block_async(args):
    subject, day, block = args
    BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data" # Define or ensure it's global
    path = f"{BASE_DATA_PATH}/participant_{subject}/participant{subject}_day{day}_block{block}"

    try:
        X, (y_grasp, y_position) = await load_data_async(path)
        if X.size == 0: return None

        processed_X = preprocess_emg(X)
        if processed_X.size == 0: return None
        features = extract_features(processed_X)
        if features.size == 0: return None
        return features, y_grasp, y_position
    except FileNotFoundError:
        # print(f"Skipping {path}: File not found (handled in load_data_async).")
        return None
    except Exception as e:
        print(f"Error processing {path} in process_subject_block_async: {str(e)}")
        return None

# Synchronous CPU-Bound Functions
def preprocess_emg(emg_signals, fs=2000):
    processed_signals = []
    if not isinstance(emg_signals, np.ndarray) or emg_signals.ndim < 2: return np.array([])
    for signal in emg_signals:
        if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0: continue
        rectified = np.abs(signal)
        filter_order = 4; cutoff_freq = 20
        min_len_for_filtfilt = 3 * (filter_order + 1)
        if signal.shape[1] >= min_len_for_filtfilt:
            b, a = butter(filter_order, cutoff_freq/(fs/2), 'low')
            smoothed = filtfilt(b, a, rectified, axis=1)
        else:
            smoothed = rectified
        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        normalized = np.divide(smoothed - mean, std + 1e-9, out=np.zeros_like(smoothed), where=std > 1e-9)
        processed_signals.append(normalized)
    return np.array(processed_signals) if processed_signals else np.array([])

def extract_features(processed_signals, window_size=150, overlap=0.5):
    features_for_all_trials = []
    if not isinstance(processed_signals, np.ndarray) or processed_signals.ndim != 3: return np.array([])
    for signal_trial in processed_signals:
        if signal_trial.ndim != 2 or signal_trial.shape[0] == 0 or signal_trial.shape[1] < window_size: continue
        num_channels, signal_length = signal_trial.shape
        step_size = int(window_size * (1 - overlap)); step_size = max(1, step_size)
        num_windows = (signal_length - window_size) // step_size + 1
        if num_windows <= 0: continue
        trial_window_features_list = []
        for i in range(num_windows):
            start = i * step_size; end = start + window_size
            window = signal_trial[:, start:end]
            if window.shape[1] != window_size: continue
            mean_abs_val = np.mean(np.abs(window), axis=1)
            var = np.var(window, axis=1)
            rms = np.sqrt(np.mean(window**2, axis=1))
            if window.shape[1] > 1:
                waveform_length = np.sum(np.abs(np.diff(window, axis=1)), axis=1)
                zero_crossings = np.sum(np.diff(np.sign(window + 1e-9), axis=1) != 0, axis=1) / window_size
            else:
                waveform_length = np.zeros(num_channels); zero_crossings = np.zeros(num_channels)
            fft_val = np.fft.fft(window, axis=1); fft_abs = np.abs(fft_val[:, :window_size//2])
            if fft_abs.shape[1] == 0: mean_freq = np.zeros(num_channels); freq_std = np.zeros(num_channels)
            else: mean_freq = np.mean(fft_abs, axis=1); freq_std = np.std(fft_abs, axis=1)
            current_window_features = np.concatenate([mean_abs_val, var, rms, waveform_length, zero_crossings, mean_freq, freq_std])
            trial_window_features_list.append(current_window_features)
        if trial_window_features_list:
            features_for_all_trials.append(np.mean(trial_window_features_list, axis=0))
    return np.array(features_for_all_trials) if features_for_all_trials else np.array([])

# UPDATED Evaluation function
def evaluate_model(model, X_data, y_true, class_names, task_name, dataset_name):
    print(f"\n--- Evaluating {task_name} Model on {dataset_name} Data ---")
    if X_data.size == 0 or y_true.size == 0:
        print(f"Error: Empty data for {task_name} on {dataset_name}. Skipping evaluation.")
        return
    try:
        accuracy = model.score(X_data, y_true)
        print(f"{task_name} - {dataset_name} Accuracy: {accuracy:.4f}")
        y_pred_classes = model.predict(X_data)
    except Exception as e:
        print(f"Error during prediction/scoring for {task_name} on {dataset_name}: {e}")
        return

    unique_labels_in_data = np.unique(np.concatenate((y_true, y_pred_classes)))
    cm_indices = list(range(len(class_names)))
    if max(unique_labels_in_data, default=-1) >= len(class_names): # max needs default for empty sequence
        print(f"Warning for {task_name} ({dataset_name}): Max label in data exceeds class_names length. CM/Report might be affected.")

    try:
        cm = confusion_matrix(y_true, y_pred_classes, labels=cm_indices)
        plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names)*0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label'); plt.ylabel('Actual Label')
        plt.title(f'{task_name} ({model.__class__.__name__}) - {dataset_name} Confusion Matrix')
        plt.tight_layout(); plt.show()
    except Exception as e: print(f"Error generating CM for {task_name} on {dataset_name}: {e}")

    print(f"\n--- {task_name} - {dataset_name} Classification Report ({model.__class__.__name__}) ---")
    try:
        print(classification_report(y_true, y_pred_classes, labels=cm_indices, target_names=class_names, zero_division=0))
    except Exception as e:
        print(f"Error generating classification report for {task_name} on {dataset_name}: {e}")
        try: print(classification_report(y_true, y_pred_classes, zero_division=0)) # Fallback
        except Exception as ef: print(f"Fallback report failed: {ef}")

def display_lda_metrics_and_plots(lda_model, X_data_scaled, y_data, class_names_list, task_name, dataset_name="Test"): # Added dataset_name
    print(f"\n--- LDA Specific Analysis: {task_name} Task on {dataset_name} Data ---")
    if not hasattr(lda_model, 'n_components_'): print(f"LDA model for {task_name} not fitted."); return
    n_components = lda_model.n_components_
    print(f"Number of LDs for {task_name}: {n_components}")
    if hasattr(lda_model, 'explained_variance_ratio_'):
        explained_variance = lda_model.explained_variance_ratio_
        print(f"Explained Variance Ratio ({task_name}, {dataset_name}): {explained_variance}")
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
        plt.ylabel('Explained Variance Ratio'); plt.xlabel('Linear Discriminants'); plt.xticks(range(1, len(explained_variance) + 1))
        plt.title(f'{task_name} ({dataset_name}) - Explained Variance by LD'); plt.grid(axis='y', linestyle='--'); plt.tight_layout(); plt.show()
    else: explained_variance = None
    try: X_lda = lda_model.transform(X_data_scaled)
    except Exception as e: print(f"Error transforming data with LDA for {task_name} ({dataset_name}): {e}"); return
    print(f"Shape after LDA transformation ({task_name}, {dataset_name}): {X_lda.shape}")
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_data.astype(int), cmap='viridis', alpha=0.7, edgecolors='k', s=50)
        unique_data_labels = np.unique(y_data)
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names_list[i], markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10) for i in unique_data_labels if i < len(class_names_list)]
        plt.legend(handles=legend_handles, title="Classes"); plt.xlabel('LD1'); plt.ylabel('LD2')
        title_str = f'{task_name} ({dataset_name}) - Data Projected onto First Two LDs'
        if explained_variance and len(explained_variance) >= 2: title_str += f'\n(Explains {explained_variance[0]+explained_variance[1]:.2%} of variance)'
        plt.title(title_str); plt.grid(True, linestyle='--'); plt.tight_layout(); plt.show()
    elif n_components == 1:
        print(f"\nOnly 1 LD for {task_name} ({dataset_name}). Plotting 1D distribution.")
        plt.figure(figsize=(10, 6))
        y_test_named = [class_names_list[i] if i < len(class_names_list) else f"Class {i}" for i in y_data.astype(int)]
        df_lda = pd.DataFrame({'LD1': X_lda[:, 0], 'Class': y_test_named})
        sns.histplot(data=df_lda, x='LD1', hue='Class', kde=True, palette='viridis', bins=30)
        plt.xlabel('LD1'); plt.ylabel('Density / Count')
        title_str = f'{task_name} ({dataset_name}) - Data Projected onto LD1'
        if explained_variance and len(explained_variance) >= 1: title_str += f'\n(Explains {explained_variance[0]:.2%} of variance)'
        plt.title(title_str); plt.grid(True, linestyle='--'); plt.tight_layout(); plt.show()

async def main_async():
    BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data"
    tasks_to_process = []
    for subject in range(1, 9): # Subjects 1-8
        for day in [1, 2]:
            for block in [1, 2]:
                tasks_to_process.append((subject, day, block, BASE_DATA_PATH)) # Pass base path if needed by loader

    all_X_list, all_y_grasp_list, all_y_position_list = [], [], []
    async_tasks = [process_subject_block_async(task_info[:3]) for task_info in tasks_to_process] # Assuming process_subject_block_async uses global BASE_DATA_PATH or has it hardcoded

    print(f"Starting processing for {len(async_tasks)} blocks...")
    for future in tqdm(asyncio.as_completed(async_tasks), total=len(async_tasks), desc="Processing Blocks"):
        result = await future
        if result:
            features, y_grasp, y_position = result
            if features.size > 0 and features.ndim == 2:
                all_X_list.append(features)
                all_y_grasp_list.append(y_grasp)
                all_y_position_list.append(y_position)

    if not all_X_list: print("Error: No data loaded. Exiting."); return None, None
    try:
        all_X = np.concatenate(all_X_list, axis=0)
        all_y_grasp = np.concatenate(all_y_grasp_list, axis=0)
        all_y_position = np.concatenate(all_y_position_list, axis=0)
    except ValueError as e: print(f"Error concatenating data: {e}"); return None, None

    print(f"\nTotal samples: {all_X.shape[0]}")
    if all_X.shape[0] == 0 or all_y_grasp.shape[0] != all_X.shape[0] or all_y_position.shape[0] != all_X.shape[0]:
        print("Error: Data shape mismatch or no data. Exiting."); return None, None

    # --- Data Splitting (Train, Validation, Test) ---
    # Stratify by grasp for consistency. Ensure enough samples for multi-class stratification.
    # First split: 80% for train+validation, 20% for test
    try:
        (X_train_val, X_test,
        y_grasp_train_val, y_grasp_test,
        y_position_train_val, y_position_test) = train_test_split(
            all_X, all_y_grasp, all_y_position,
            test_size=0.2, random_state=42, stratify=all_y_grasp
        )
        # Second split: (from 80%) 75% for train, 25% for validation -> 60% train, 20% val of total
        (X_train, X_val,
        y_grasp_train, y_grasp_val,
        y_position_train, y_position_val) = train_test_split(
            X_train_val, y_grasp_train_val, y_position_train_val,
            test_size=0.25, random_state=42, stratify=y_grasp_train_val
        )
    except ValueError as e: # Handles cases like too few samples for stratification
        print(f"Error during data splitting (possibly too few samples per class for stratification): {e}")
        print("Consider using a simpler train/test split or acquiring more data.")
        return None, None


    print(f"\nDataset sizes: Train={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    num_grasp_classes = len(np.unique(all_y_grasp)) # Based on all data to define labels consistently
    num_position_classes = len(np.unique(all_y_position))
    grasp_class_names = [f"Grasp {i}" for i in range(num_grasp_classes)]
    position_class_names = [f"Pos {i}" for i in range(num_position_classes)]

    lda_grasp_model, lda_position_model = None, None # Initialize

    # --- Grasp Model ---
    if num_grasp_classes > 1 and X_train_scaled.shape[0] > num_grasp_classes:
        print("\n--- Training LDA Grasp Model ---")
        lda_grasp_model = LinearDiscriminantAnalysis()
        lda_grasp_model.fit(X_train_scaled, y_grasp_train)
        print("--- LDA Grasp Model Training Finished ---")
        evaluate_model(lda_grasp_model, X_train_scaled, y_grasp_train, grasp_class_names, "Grasp", "Training")
        evaluate_model(lda_grasp_model, X_val_scaled, y_grasp_val, grasp_class_names, "Grasp", "Validation")
        evaluate_model(lda_grasp_model, X_test_scaled, y_grasp_test, grasp_class_names, "Grasp", "Test")
        display_lda_metrics_and_plots(lda_grasp_model, X_test_scaled, y_grasp_test, grasp_class_names, "Grasp", "Test")
    else: print("\nSkipping Grasp Model: Not enough classes or samples.")

    # --- Position Model ---
    if num_position_classes > 1 and X_train_scaled.shape[0] > num_position_classes:
        print("\n--- Training LDA Position Model ---")
        lda_position_model = LinearDiscriminantAnalysis()
        lda_position_model.fit(X_train_scaled, y_position_train)
        print("--- LDA Position Model Training Finished ---")
        evaluate_model(lda_position_model, X_train_scaled, y_position_train, position_class_names, "Position", "Training")
        evaluate_model(lda_position_model, X_val_scaled, y_position_val, position_class_names, "Position", "Validation")
        evaluate_model(lda_position_model, X_test_scaled, y_position_test, position_class_names, "Position", "Test")
        display_lda_metrics_and_plots(lda_position_model, X_test_scaled, y_position_test, position_class_names, "Position", "Test")
    else: print("\nSkipping Position Model: Not enough classes or samples.")

    return lda_grasp_model, lda_position_model

if __name__ == "__main__":
    # Ensure BASE_DATA_PATH is correctly defined if not hardcoded within functions that need it.
    # It's used in process_subject_block_async if not passed directly.
    # For this script, process_subject_block_async has it hardcoded.
    
    results = asyncio.run(main_async()) # Get tuple of models

    if results is not None:
        grasp_lda_model, position_lda_model = results
        print("\nAsync pipeline finished.")
        if grasp_lda_model:
            print("Trained LDA Grasp Model is available.")
            try: joblib.dump(grasp_lda_model, 'grasp_lda_model.pkl'); print("Grasp LDA model saved.")
            except Exception as e: print(f"Error saving Grasp LDA model: {e}")
        if position_lda_model:
            print("Trained LDA Position Model is available.")
            try: joblib.dump(position_lda_model, 'position_lda_model.pkl'); print("Position LDA model saved.")
            except Exception as e: print(f"Error saving Position LDA model: {e}")
    else:
        print("\nAsync pipeline did not complete successfully or encountered critical errors.")