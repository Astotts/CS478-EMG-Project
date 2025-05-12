import asyncio
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm.asyncio import tqdm as async_tqdm # Use tqdm's async version for gather
from tqdm import tqdm as sync_tqdm # Standard tqdm for sequential loops
from collections import Counter
# Import for class weighting (though not used for grasp model in this version, kept for completeness if needed later)
from sklearn.utils.class_weight import compute_class_weight

# --- Global Configuration ---
BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data" # <-- IMPORTANT: SET YOUR PATH!
RANDOM_STATE = 42 # For reproducibility

# --- Helper Functions ---

def _load_emg_data(file_path):
    """Loads EMG data from an HDF5 file (Synchronous)."""
    try:
        with h5py.File(file_path, 'r') as f:
            return {int(key): np.array(f[key]) for key in f.keys()}
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading HDF5 file {file_path}: {str(e)}")
        return None

async def load_raw_data_async(subject_path):
    """Loads RAW EMG data and trial GRASP labels asynchronously."""
    loop = asyncio.get_running_loop()
    hdf5_path = f"{subject_path}/emg_data.hdf5"
    csv_path = f"{subject_path}/trials.csv"

    emg_data = None
    trials_df = None
    error_occurred = False

    emg_data_task = loop.run_in_executor(None, _load_emg_data, hdf5_path)
    try:
        trials_df_task = loop.run_in_executor(None, pd.read_csv, csv_path)
        emg_data, trials_df = await asyncio.gather(emg_data_task, trials_df_task)
    except FileNotFoundError:
        # print(f"Error: CSV file not found at {csv_path}") # Suppressed for cleaner logs
        error_occurred = True
        if not emg_data_task.done():
            await emg_data_task
        emg_data = None
        trials_df = None
    except Exception as e:
        print(f"Error during async loading gather in {subject_path}: {str(e)}")
        error_occurred = True
        if emg_data_task and not emg_data_task.done(): await emg_data_task

    if emg_data is None or trials_df is None or error_occurred:
        return None

    if not isinstance(emg_data, dict) or not emg_data:
        # print(f"Warning: No valid EMG data loaded from {hdf5_path}")
        return None
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        # print(f"Warning: Empty or invalid trials DataFrame loaded from {csv_path}")
        return None
    if 'grasp' not in trials_df.columns: # MODIFIED: Only check for 'grasp'
        print(f"Warning: Missing required column 'grasp' in {csv_path}")
        return None

    X_raw = {}
    y_grasp_dict = {}
    valid_trial_indices = []
    num_csv_rows = len(trials_df)

    for trial_idx in range(150):
        try:
            if trial_idx not in emg_data:
                continue
            if trial_idx >= num_csv_rows:
                # print(f"Warning: Trial index {trial_idx} >= num_csv_rows {num_csv_rows} in {subject_path}. Skipping.")
                continue

            current_signal = emg_data[trial_idx]
            trial_info = trials_df.iloc[trial_idx]

            if not isinstance(current_signal, np.ndarray) or current_signal.ndim != 2 or current_signal.shape[0] == 0 or current_signal.shape[1] == 0:
                # print(f"Warning: Invalid signal shape/type for trial {trial_idx} in {subject_path}. Skipping.")
                continue

            X_raw[trial_idx] = current_signal
            y_grasp_dict[trial_idx] = int(trial_info['grasp'] - 1) # Ensure int, 0-based
            valid_trial_indices.append(trial_idx)

        except IndexError:
            # print(f"Error: Index {trial_idx} out of bounds for trials_df (len={num_csv_rows}) in {subject_path}")
            continue
        except KeyError:
            # print(f"Error: Column 'grasp' missing for trial {trial_idx} access in {subject_path}")
            continue
        except Exception as e:
            print(f"Error processing trial {trial_idx} during loading prep in {subject_path}: {str(e)}")
            continue

    if not X_raw:
        # print(f"Warning: No valid raw trials prepared for {subject_path}")
        return None

    # MODIFIED: Return only grasp related data
    return X_raw, y_grasp_dict, sorted(valid_trial_indices)


# --- Preprocessing and Feature Extraction (Unchanged from your version) ---
def preprocess_emg(emg_signals_list, fs=2000):
    processed_signals = []
    original_indices = []
    for i, signal in enumerate(emg_signals_list):
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
            continue
        rectified = np.abs(signal)
        cutoff_freq = 20; filter_order = 4; nyquist_freq = fs / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        smoothed = rectified
        if 0 < normalized_cutoff < 1:
            try:
                b, a = butter(filter_order, normalized_cutoff, 'low')
                min_len_filtfilt = 3 * max(len(a), len(b))
                if rectified.shape[1] >= min_len_filtfilt:
                    smoothed = filtfilt(b, a, rectified, axis=1)
            except ValueError: # Simplified error handling for brevity
                pass # Keep smoothed as rectified if filter fails
        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        std_mask = std > 1e-9
        normalized = np.zeros_like(smoothed)
        np.divide(smoothed - mean, std + 1e-9, out=normalized, where=std_mask)
        processed_signals.append(normalized)
        original_indices.append(i)
    return processed_signals, original_indices

def extract_features(processed_signals_list, window_size=150, overlap=0.5):
    features = []
    processed_indices_map = []
    num_channels = 0
    if processed_signals_list:
        for sig in processed_signals_list:
            if sig.ndim == 2 and sig.shape[0] > 0:
                num_channels = sig.shape[0]
                break
        if num_channels == 0:
            return np.array(features), []

    step_size = int(window_size * (1 - overlap))
    if step_size <= 0: step_size = 1

    for idx, signal in enumerate(processed_signals_list):
        if signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_size:
            continue
        signal_length = signal.shape[1]
        num_windows = (signal_length - window_size) // step_size + 1
        if num_windows <= 0: continue

        window_features_for_trial = []
        valid_window_count = 0
        for i in range(num_windows):
            start = i * step_size; end = start + window_size
            if end > signal_length: continue
            window = signal[:, start:end]
            if window.shape[1] != window_size: continue
            try:
                mean_abs_val = np.mean(np.abs(window), axis=1)
                var = np.var(window, axis=1)
                rms = np.sqrt(np.mean(window**2, axis=1))
                diff_sig = np.diff(window, axis=1)
                waveform_length = np.sum(np.abs(diff_sig), axis=1) if diff_sig.size > 0 else np.zeros(num_channels)
                zero_crossings = np.sum(np.diff(np.sign(window + 1e-9), axis=1) != 0, axis=1) / window_size if window.shape[1] > 1 else np.zeros(num_channels)
                fft_val = np.fft.fft(window, axis=1)
                fft_abs = np.abs(fft_val[:, :window_size//2])
                if fft_abs.shape[1] == 0:
                    mean_freq = np.zeros(num_channels); freq_std = np.zeros(num_channels)
                else:
                    mean_freq = np.mean(fft_abs, axis=1); freq_std = np.std(fft_abs, axis=1)
                window_feature = np.concatenate([
                    mean_abs_val, var, rms, waveform_length, zero_crossings,
                    mean_freq, freq_std])
                if np.any(np.isnan(window_feature)) or np.any(np.isinf(window_feature)): continue
                window_features_for_trial.append(window_feature)
                valid_window_count += 1
            except Exception: continue
        if valid_window_count > 0:
            trial_feature_vector = np.mean(window_features_for_trial, axis=0)
            if not (np.any(np.isnan(trial_feature_vector)) or np.any(np.isinf(trial_feature_vector))):
                features.append(trial_feature_vector)
                processed_indices_map.append(idx)
    return np.array(features), processed_indices_map

# --- Evaluation Function ---
# MODIFIED: Changed task_name to set_name (e.g., "Training", "Validation", "Test")
def evaluate_model_on_set(model, X_data, y_data, labels, set_name, model_name="Grasp"):
    """Evaluates the model on a specific dataset (train, val, or test)."""
    print(f"\n--- Evaluating {model_name} Model on {set_name} Set ---")
    try:
        # For training set, loss and accuracy are already known from history if needed
        # For val/test, model.evaluate can be used.
        # We will directly use predict for consistency in getting y_pred for reports.
        y_pred_probs = model.predict(X_data, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Calculate accuracy for the set
        accuracy = np.mean(y_pred_classes == y_data)
        print(f"{model_name} {set_name} Accuracy: {accuracy:.4f}")

        # If loss is needed and not from history (e.g. for test/val directly)
        if set_name.lower() != "training": # Avoid re-calculating training loss if not needed
            try:
                loss, _ = model.evaluate(X_data, y_data, verbose=0)
                print(f"{model_name} {set_name} Loss: {loss:.4f}")
            except Exception as e:
                 print(f"Error calculating loss for {set_name} set: {e}")


    except Exception as e:
        print(f"Error during model.predict on {set_name} set: {e}")
        return

    if y_data.size == 0 or y_pred_classes.size == 0:
        print(f"Empty data for plots/report on {set_name} set.")
        return

    # Confusion Matrix
    try:
        all_possible_labels_indices = list(range(len(labels)))
        cm = confusion_matrix(y_data, y_pred_classes, labels=all_possible_labels_indices)
        plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)*0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.title(f'{model_name} Task - {set_name} Set - Confusion Matrix')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating Confusion Matrix for {set_name} set: {e}")

    # Classification Report
    print(f"\n--- {model_name} {set_name} Set Classification Report ---")
    try:
        all_possible_labels_indices = list(range(len(labels)))
        print(classification_report(y_data, y_pred_classes, labels=all_possible_labels_indices,
                                      target_names=labels, zero_division=0))
    except Exception as e:
        print(f"Error generating classification report for {set_name} set: {e}")


# --- Model Creation Function (Unchanged, but position model function removed) ---
def create_grasp_model(input_shape, num_grasps):
    """Creates the Grasp prediction model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_grasps, activation='softmax', name='grasp_output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Pipeline ---
async def main_async():
    """Main pipeline: Load -> Find Global Min Length -> Process -> Split -> Train -> Evaluate."""

    tasks_to_process = []
    for subject in range(1, 9): # Subjects 1-8
        for day in [1, 2]:
            for block in [1, 2]:
                tasks_to_process.append((subject, day, block))

    print(f"--- Phase 1: Starting Data Loading ({len(tasks_to_process)} blocks) ---")
    loading_tasks = []
    for subject, day, block in tasks_to_process:
        path = f"{BASE_DATA_PATH}/participant_{subject}/participant{subject}_day{day}_block{block}"
        loading_tasks.append(load_raw_data_async(path))

    block_results_raw = await async_tqdm.gather(*loading_tasks, desc="Loading Blocks")
    block_results_raw = [res for res in block_results_raw if res is not None]

    if not block_results_raw:
        print("Error: No data loaded successfully from any block. Exiting.")
        return None # MODIFIED: Return only one None as we only have one model

    print(f"--- Phase 1 Finished: Successfully loaded raw data from {len(block_results_raw)} blocks ---")

    print("\n--- Phase 2a: Finding Global Minimum Signal Length ---")
    all_valid_lengths = []
    # MODIFIED: Unpack only grasp related data
    for X_raw_block, _, valid_indices in block_results_raw:
        for idx in valid_indices:
            if idx in X_raw_block and X_raw_block[idx].ndim == 2:
                all_valid_lengths.append(X_raw_block[idx].shape[1])

    if not all_valid_lengths:
        print("Error: Could not find any valid signals to determine minimum length. Exiting.")
        return None

    global_min_length = min(all_valid_lengths)
    print(f"Global minimum signal length found: {global_min_length}")

    print(f"\n--- Phase 2b: Processing Data from {len(block_results_raw)} blocks ---")
    all_features_list = []
    all_y_grasp_processed_list = []

    for block_data in sync_tqdm(block_results_raw, desc="Processing Blocks"):
        # MODIFIED: Unpack only grasp related data
        X_raw_block, y_grasp_dict, valid_indices = block_data

        signals_to_preprocess = []
        labels_grasp_truncated = []
        original_indices_truncated = []

        for i, trial_idx in enumerate(valid_indices):
            signal = X_raw_block[trial_idx]
            if signal.shape[1] >= global_min_length:
                signals_to_preprocess.append(signal[:, :global_min_length])
                labels_grasp_truncated.append(y_grasp_dict[trial_idx])
                original_indices_truncated.append(i)

        if not signals_to_preprocess: continue

        processed_signals, indices_after_preprocess = preprocess_emg(signals_to_preprocess)
        if not processed_signals: continue

        labels_grasp_preprocessed = np.array(labels_grasp_truncated)[indices_after_preprocess]

        features_block, indices_after_features = extract_features(processed_signals)
        if features_block.size == 0: continue

        final_y_grasp = labels_grasp_preprocessed[indices_after_features]

        if features_block.shape[0] == final_y_grasp.shape[0]:
            all_features_list.append(features_block)
            all_y_grasp_processed_list.append(final_y_grasp)

    print(f"--- Phase 2 Finished: Processing complete ---")

    if not all_features_list:
        print("Error: No features extracted successfully from any block after processing. Exiting.")
        return None

    print("\n--- Phase 3: Final Data Prep, Training & Evaluation ---")
    try:
        all_X = np.concatenate(all_features_list, axis=0)
        all_y_grasp = np.concatenate(all_y_grasp_processed_list, axis=0)
    except ValueError as e:
        print(f"Error concatenating final data: {e}")
        return None

    print(f"Total usable samples for grasp model: {all_X.shape[0]}")
    if all_X.shape[0] == 0:
        print("Error: No samples available for training.")
        return None
    print(f"Feature shape: {all_X.shape[1:]}")
    print("Grasp Label Distribution:", Counter(all_y_grasp))

    # --- Data Split (60% Train, 20% Validation, 20% Test) ---
    print("\nSplitting data into Train (60%), Validation (20%), Test (20%) sets (Stratify by Grasp)...")
    try:
        # First split: 80% for train+validation, 20% for test
        X_temp, X_test, y_grasp_temp, y_grasp_test = train_test_split(
            all_X, all_y_grasp,
            test_size=0.2, # 20% for test
            random_state=RANDOM_STATE,
            stratify=all_y_grasp
        )
        # Second split: From the 80%, split into 75% for train (60% of total) and 25% for validation (20% of total)
        # test_size for this split will be 0.25 (to get 20% validation from the 80% temp set)
        X_train, X_val, y_grasp_train, y_grasp_val = train_test_split(
            X_temp, y_grasp_temp,
            test_size=0.25, # 0.25 * 0.80 = 0.20 (20% of total for validation)
            random_state=RANDOM_STATE,
            stratify=y_grasp_temp # Stratify on the temporary grasp labels
        )
        print(f"Split Sizes: Train={len(X_train)} ({len(X_train)/len(all_X)*100:.1f}%), "
              f"Val={len(X_val)} ({len(X_val)/len(all_X)*100:.1f}%), "
              f"Test={len(X_test)} ({len(X_test)/len(all_X)*100:.1f}%)")
    except ValueError as e:
        print(f"Error splitting data: {e}")
        return None

    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Class Weights for Grasp Model (Optional, but good practice if imbalanced) ---
    # print("Calculating class weights for Grasp model...")
    # unique_grasp_classes = np.unique(y_grasp_train)
    # try:
    #     grasp_class_weights = compute_class_weight(
    #         class_weight='balanced',
    #         classes=unique_grasp_classes,
    #         y=y_grasp_train
    #     )
    #     grasp_class_weights_dict = dict(zip(unique_grasp_classes, grasp_class_weights))
    #     print("Grasp Class Weights:", grasp_class_weights_dict)
    # except Exception as e:
    #     print(f"Error computing grasp class weights: {e}. Proceeding without weights.")
    #     grasp_class_weights_dict = None # Fallback
    grasp_class_weights_dict = None # Set to None if not using class weights for grasp

    num_grasp_classes = len(np.unique(all_y_grasp))
    input_feature_shape = (X_train_scaled.shape[1],)

    print("\nCreating grasp model...")
    grasp_model = create_grasp_model(input_feature_shape, num_grasp_classes)
    print("--- Grasp Model Summary ---"); grasp_model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print("\n--- Starting Grasp Model Training ---")
    grasp_history = grasp_model.fit(
        X_train_scaled, y_grasp_train, validation_data=(X_val_scaled, y_grasp_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=1,
        class_weight=grasp_class_weights_dict # Apply class weights here if calculated
    )

    # --- Evaluation on Train, Validation, and Test Sets ---
    grasp_labels_names = [f"Grasp {i}" for i in range(num_grasp_classes)]

    # Evaluate on Training set
    evaluate_model_on_set(grasp_model, X_train_scaled, y_grasp_train, grasp_labels_names, "Training", "Grasp")

    # Evaluate on Validation set
    evaluate_model_on_set(grasp_model, X_val_scaled, y_grasp_val, grasp_labels_names, "Validation", "Grasp")

    # Evaluate on Test set
    evaluate_model_on_set(grasp_model, X_test_scaled, y_grasp_test, grasp_labels_names, "Test", "Grasp")

    return grasp_model, grasp_history


# --- Execution ---
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPUs,", len(tf.config.experimental.list_logical_devices('GPU')), "Logical GPUs")
        except RuntimeError as e: print(e)

    results = asyncio.run(main_async())

    if results:
        # MODIFIED: Unpack only grasp model and history
        grasp_m, grasp_h = results
        print("\nAsync pipeline finished successfully for Grasp Model.")
    else:
        print("\nAsync pipeline did not complete successfully.")