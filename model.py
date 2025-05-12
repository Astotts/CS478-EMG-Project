import asyncio
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Removed LabelEncoder (not used)
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm.asyncio import tqdm as async_tqdm # Use tqdm's async version for gather
from tqdm import tqdm as sync_tqdm # Standard tqdm for sequential loops
from collections import Counter
# Import for class weighting
from sklearn.utils.class_weight import compute_class_weight

# --- Global Configuration ---
BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data" # <-- IMPORTANT: SET YOUR PATH!
RANDOM_STATE = 42 # For reproducibility

# --- Helper Functions ---

def _load_emg_data(file_path):
    """Loads EMG data from an HDF5 file (Synchronous)."""
    # Keep this function as is
    try:
        with h5py.File(file_path, 'r') as f:
            return {int(key): np.array(f[key]) for key in f.keys()}
    except FileNotFoundError:
        # print(f"Error: HDF5 file not found at {file_path}") # Suppress for cleaner logs, handle None return
        return None
    except Exception as e:
        print(f"Error loading HDF5 file {file_path}: {str(e)}")
        return None

# Modified: Removed min_length calculation, returns raw data
async def load_raw_data_async(subject_path):
    """Loads RAW EMG data and trial labels asynchronously."""
    loop = asyncio.get_running_loop()
    hdf5_path = f"{subject_path}/emg_data.hdf5"
    csv_path = f"{subject_path}/trials.csv"

    emg_data = None
    trials_df = None
    error_occurred = False

    # Use loop.run_in_executor for potentially blocking I/O
    emg_data_task = loop.run_in_executor(None, _load_emg_data, hdf5_path)
    # Load labels using pandas within executor
    try:
        # Need to wrap pandas read_csv as it can raise errors directly
        trials_df_task = loop.run_in_executor(None, pd.read_csv, csv_path)
        emg_data, trials_df = await asyncio.gather(emg_data_task, trials_df_task)
    except FileNotFoundError as e:
            # Specifically catch FileNotFoundError from pd.read_csv if it happens before gather completes
            print(f"Error: CSV file not found at {csv_path}")
            error_occurred = True
            # If HDF5 load is still running, await it to avoid pending tasks, but ignore result if CSV failed
            if not emg_data_task.done():
                await emg_data_task # Await completion
            emg_data = None # Ensure emg_data is None if CSV fails
            trials_df = None
    except Exception as e:
            print(f"Error during async loading gather in {subject_path}: {str(e)}")
            error_occurred = True
            # Await any pending task
            if emg_data_task and not emg_data_task.done(): await emg_data_task
            # trials_df_task might not exist if pd.read_csv failed before assignment
            # We set both to None below anyway.


    # Handle potential loading errors after gather/exceptions
    if emg_data is None or trials_df is None or error_occurred:
       # print(f"Warning: Data files not found or failed to load in {subject_path}")
       return None # Return None to indicate failure

    # --- Basic Validation ---
    if not isinstance(emg_data, dict) or not emg_data:
        print(f"Warning: No valid EMG data loaded from {hdf5_path}")
        return None
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
         print(f"Warning: Empty or invalid trials DataFrame loaded from {csv_path}")
         return None
    if 'grasp' not in trials_df.columns or 'target_position' not in trials_df.columns:
         print(f"Warning: Missing required columns 'grasp' or 'target_position' in {csv_path}")
         return None

    # --- Prepare Output (No Truncation Here) ---
    X_raw = {}
    y_grasp_dict = {}
    y_position_dict = {}
    valid_trial_indices = [] # Keep track of trials that are valid in both EMG and CSV

    num_csv_rows = len(trials_df)

    for trial_idx in range(150): # Iterate through expected trial indices
        try:
            if trial_idx not in emg_data:
                 # print(f"Debug: Trial {trial_idx} not in emg_data for {subject_path}")
                 continue # Skip if EMG data missing

            if trial_idx >= num_csv_rows:
                 print(f"Warning: Trial index {trial_idx} >= num_csv_rows {num_csv_rows} in {subject_path}. Skipping.")
                 continue # Skip if no corresponding row in CSV

            current_signal = emg_data[trial_idx]
            trial_info = trials_df.iloc[trial_idx] # Get label info

            # Basic check for signal validity (Type and dimension)
            if not isinstance(current_signal, np.ndarray) or current_signal.ndim != 2 or current_signal.shape[0] == 0 or current_signal.shape[1] == 0:
                print(f"Warning: Invalid signal shape/type {getattr(current_signal, 'shape', type(current_signal))} for trial {trial_idx} in {subject_path}. Skipping.")
                continue

            # If all checks pass, store the raw signal and labels using trial_idx as key
            X_raw[trial_idx] = current_signal
            y_grasp_dict[trial_idx] = int(trial_info['grasp'] - 1) # Ensure int, 0-based
            y_position_dict[trial_idx] = int(trial_info['target_position'] - 1) # Ensure int, 0-based
            valid_trial_indices.append(trial_idx)

        except IndexError:
             print(f"Error: Index {trial_idx} out of bounds for trials_df (len={num_csv_rows}) in {subject_path}")
             continue
        except KeyError:
             print(f"Error: Column 'grasp' or 'target_position' missing for trial {trial_idx} access in {subject_path}")
             continue # Should be caught by earlier check, but good practice
        except Exception as e:
            print(f"Error processing trial {trial_idx} during loading prep in {subject_path}: {str(e)}")
            continue # Skip this trial

    if not X_raw: # If no valid trials were processed for this block
        print(f"Warning: No valid raw trials prepared for {subject_path}")
        return None

    # Return dicts keyed by original trial index
    return X_raw, y_grasp_dict, y_position_dict, sorted(valid_trial_indices)


# --- Preprocessing and Feature Extraction (Keep as previously corrected) ---
# --- Added robustness checks in previous versions ---
def preprocess_emg(emg_signals_list, fs=2000):
    """Processes a list of raw EMG signals."""
    processed_signals = []
    original_indices = [] # Keep track of index in original list
    for i, signal in enumerate(emg_signals_list):
        # Basic validation
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
            # print(f"Warning: Skipping invalid signal shape/type {signal.shape if isinstance(signal, np.ndarray) else type(signal)} at index {i} in preprocess_emg.")
            continue

        # --- Rectification & Filtering ---
        rectified = np.abs(signal)
        cutoff_freq = 20; filter_order = 4; nyquist_freq = fs / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        smoothed = rectified # Default if filtering fails
        if 0 < normalized_cutoff < 1:
            try:
                b, a = butter(filter_order, normalized_cutoff, 'low')
                min_len_filtfilt = 3 * max(len(a), len(b))
                if rectified.shape[1] >= min_len_filtfilt:
                    smoothed = filtfilt(b, a, rectified, axis=1)
                # else: print(f"Warning: Signal length {rectified.shape[1]} too short for filtfilt at index {i}.")
            except ValueError as e:
                print(f"Warning: Filter error (cutoff={normalized_cutoff}) at index {i}: {e}.")
        # else: print(f"Warning: Invalid norm cutoff {normalized_cutoff} at index {i}.")

        # --- Normalization ---
        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        # Check for zero or very small std dev across the signal per channel
        # If std dev is zero for a channel, output zeros for that channel.
        std_mask = std > 1e-9
        normalized = np.zeros_like(smoothed)
        # Only divide where std is non-zero
        np.divide(smoothed - mean, std + 1e-9, out=normalized, where=std_mask)


        processed_signals.append(normalized)
        original_indices.append(i) # Store the original index of the signal processed

    return processed_signals, original_indices # Return list of processed signals and their original indices

def extract_features(processed_signals_list, window_size=150, overlap=0.5):
    """Extracts features from a list of preprocessed signals."""
    features = []
    processed_indices_map = [] # Store index from input list corresponding to each feature vector
    num_channels = 0
    if processed_signals_list:
         # Determine expected number of channels from the first valid signal
         for sig in processed_signals_list:
              if sig.ndim == 2 and sig.shape[0] > 0:
                  num_channels = sig.shape[0]
                  break
         if num_channels == 0:
              print("Warning: Could not determine number of channels in extract_features.")
              return np.array(features), []


    step_size = int(window_size * (1 - overlap))
    if step_size <= 0: step_size = 1

    for idx, signal in enumerate(processed_signals_list):
        # Validation
        if signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_size:
            # print(f"Warning: Skipping signal index {idx} with shape {signal.shape} in extract_features.")
            continue

        signal_length = signal.shape[1]
        num_windows = (signal_length - window_size) // step_size + 1
        if num_windows <= 0: continue

        # --- Feature extraction per window ---
        window_features_for_trial = []
        valid_window_count = 0
        for i in range(num_windows):
            start = i * step_size; end = start + window_size
            if end > signal_length: continue # Should not happen with floor division, but safety check
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
            except Exception: continue # Skip window on error

        if valid_window_count > 0:
            trial_feature_vector = np.mean(window_features_for_trial, axis=0)
            if not (np.any(np.isnan(trial_feature_vector)) or np.any(np.isinf(trial_feature_vector))):
                features.append(trial_feature_vector)
                processed_indices_map.append(idx) # Store index from input list

    return np.array(features), processed_indices_map # Return features and indices relative to input list

# --- Evaluation and Model Creation Functions (Keep as previously corrected) ---
def evaluate_model(model, X_test, y_test, labels, task_name):
    """Evaluates a single trained model."""
    # Keep this function as previously corrected
    print(f"\n--- Evaluating {task_name} Model ---")
    try:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{task_name} Test Accuracy: {test_acc:.4f}")
        print(f"{task_name} Test Loss: {test_loss:.4f}")
    except Exception as e: print(f"Error during model.evaluate: {e}"); return
    try:
        y_pred_probs = model.predict(X_test); y_pred_classes = np.argmax(y_pred_probs, axis=1)
    except Exception as e: print(f"Error during model.predict: {e}"); return
    if y_test.size == 0 or y_pred_classes.size == 0: print("Empty data for plots/report."); return

    # Confusion Matrix
    try:
        all_possible_labels_indices = list(range(len(labels)))
        cm = confusion_matrix(y_test, y_pred_classes, labels=all_possible_labels_indices)
        plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)*0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label'); plt.ylabel('Actual Label')
        plt.title(f'{task_name} Task - Confusion Matrix'); plt.tight_layout(); plt.show()
    except Exception as e: print(f"Error generating Confusion Matrix: {e}")

    # Classification Report
    print(f"\n--- {task_name} Classification Report ---")
    try:
        all_possible_labels_indices = list(range(len(labels)))
        print(classification_report(y_test, y_pred_classes, labels=all_possible_labels_indices,
                                    target_names=labels, zero_division=0))
    except Exception as e: print(f"Error generating classification report: {e}")

def create_grasp_model(input_shape, num_grasps):
    """Creates the Grasp prediction model."""
    # Keep this function as is
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

def create_position_model(input_shape, num_positions):
    """Creates the Position prediction model."""
    # Keep this function as is
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_positions, activation='softmax', name='position_output')(x)
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

    # --- Phase 1: Asynchronous Loading ---
    print(f"--- Phase 1: Starting Data Loading ({len(tasks_to_process)} blocks) ---")
    loading_tasks = []
    for subject, day, block in tasks_to_process:
        path = f"{BASE_DATA_PATH}/participant_{subject}/participant{subject}_day{day}_block{block}"
        loading_tasks.append(load_raw_data_async(path))

    # Use tqdm.asyncio.gather for progress bar with gather
    block_results_raw = await async_tqdm.gather(*loading_tasks, desc="Loading Blocks")
    # Filter out None results from failed loads
    block_results_raw = [res for res in block_results_raw if res is not None]

    if not block_results_raw:
        print("Error: No data loaded successfully from any block. Exiting.")
        return None, None, None, None

    print(f"--- Phase 1 Finished: Successfully loaded raw data from {len(block_results_raw)} blocks ---")

    # --- Phase 2a: Find Global Minimum Signal Length ---
    print("\n--- Phase 2a: Finding Global Minimum Signal Length ---")
    all_valid_lengths = []
    for X_raw_block, _, _, valid_indices in block_results_raw:
         for idx in valid_indices:
              if idx in X_raw_block and X_raw_block[idx].ndim == 2:
                   all_valid_lengths.append(X_raw_block[idx].shape[1])

    if not all_valid_lengths:
        print("Error: Could not find any valid signals to determine minimum length. Exiting.")
        return None, None, None, None

    global_min_length = min(all_valid_lengths)
    print(f"Global minimum signal length found: {global_min_length}")

    # --- Phase 2b: Sequential Processing (Truncate, Preprocess, Feature Extract) ---
    print(f"\n--- Phase 2b: Processing Data from {len(block_results_raw)} blocks ---")
    all_features_list = []
    all_y_grasp_processed_list = []
    all_y_position_processed_list = []

    # Use standard tqdm for the sequential loop
    for block_data in sync_tqdm(block_results_raw, desc="Processing Blocks"):
        X_raw_block, y_grasp_dict, y_position_dict, valid_indices = block_data

        # 1. Truncate signals & prepare lists for preprocessing
        signals_to_preprocess = []
        labels_grasp_truncated = []
        labels_pos_truncated = []
        original_indices_truncated = [] # Indices from the block's valid_indices

        for i, trial_idx in enumerate(valid_indices):
            signal = X_raw_block[trial_idx]
            if signal.shape[1] >= global_min_length:
                signals_to_preprocess.append(signal[:, :global_min_length])
                labels_grasp_truncated.append(y_grasp_dict[trial_idx])
                labels_pos_truncated.append(y_position_dict[trial_idx])
                original_indices_truncated.append(i) # Store index relative to valid_indices
            # else: print(f"Debug: Trial {trial_idx} shorter ({signal.shape[1]}) than global min {global_min_length}. Skipping.")

        if not signals_to_preprocess: continue # Skip block if no signals long enough

        # 2. Preprocess truncated signals
        processed_signals, indices_after_preprocess = preprocess_emg(signals_to_preprocess)
        if not processed_signals: continue # Skip if preprocessing failed

        # Select labels corresponding to successfully preprocessed signals
        labels_grasp_preprocessed = np.array(labels_grasp_truncated)[indices_after_preprocess]
        labels_pos_preprocessed = np.array(labels_pos_truncated)[indices_after_preprocess]

        # 3. Extract Features
        features_block, indices_after_features = extract_features(processed_signals)
        if features_block.size == 0: continue # Skip if feature extraction failed

        # Select labels corresponding to successfully feature-extracted signals
        final_y_grasp = labels_grasp_preprocessed[indices_after_features]
        final_y_position = labels_pos_preprocessed[indices_after_features]

        # Final check for consistency
        if features_block.shape[0] == final_y_grasp.shape[0] == final_y_position.shape[0]:
            all_features_list.append(features_block)
            all_y_grasp_processed_list.append(final_y_grasp)
            all_y_position_processed_list.append(final_y_position)
        # else: print("Warning: Mismatch after final label selection. Skipping block result.")


    print(f"--- Phase 2 Finished: Processing complete ---")

    if not all_features_list:
        print("Error: No features extracted successfully from any block after processing. Exiting.")
        return None, None, None, None

    # --- Phase 3: Concatenate, Split, Train, Evaluate ---
    print("\n--- Phase 3: Final Data Prep, Training & Evaluation ---")
    try:
        all_X = np.concatenate(all_features_list, axis=0)
        all_y_grasp = np.concatenate(all_y_grasp_processed_list, axis=0)
        all_y_position = np.concatenate(all_y_position_processed_list, axis=0)
    except ValueError as e: print(f"Error concatenating final data: {e}"); return None, None, None, None

    print(f"Total usable samples: {all_X.shape[0]}")
    if all_X.shape[0] == 0: print("Error: No samples available for training."); return None, None, None, None
    print(f"Feature shape: {all_X.shape[1:]}")
    print("Grasp Label Distribution:", Counter(all_y_grasp))
    print("Position Label Distribution:", Counter(all_y_position))

    # --- Single Data Split ---
    print("\nSplitting data into Train/Validation/Test sets (Stratify by Grasp)...")
    try:
        (X_train, X_test,
         y_grasp_train, y_grasp_test,
         y_position_train, y_position_test) = train_test_split(
            all_X, all_y_grasp, all_y_position,
            test_size=0.2, random_state=RANDOM_STATE, stratify=all_y_grasp
        )
        (X_train, X_val,
         y_grasp_train, y_grasp_val,
         y_position_train, y_position_val) = train_test_split(
            X_train, y_grasp_train, y_position_train,
            test_size=0.25, random_state=RANDOM_STATE, stratify=y_grasp_train
        )
        print(f"Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    except ValueError as e: print(f"Error splitting data: {e}"); return None, None, None, None

    # --- Standardization ---
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Class Weights for Position Model ---
    print("Calculating class weights for Position model...")
    unique_pos_classes = np.unique(y_position_train)
    try:
        pos_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_pos_classes,
            y=y_position_train
        )
        # Create dict mapping class index to weight
        position_class_weights_dict = dict(zip(unique_pos_classes, pos_class_weights))
        print("Position Class Weights:", position_class_weights_dict)
    except Exception as e:
         print(f"Error computing class weights: {e}. Proceeding without weights.")
         position_class_weights_dict = None # Fallback


    # --- Model Creation ---
    num_grasp_classes = len(np.unique(all_y_grasp))
    num_position_classes = len(np.unique(all_y_position))
    input_feature_shape = (X_train_scaled.shape[1],)

    print("\nCreating models...")
    grasp_model = create_grasp_model(input_feature_shape, num_grasp_classes)
    position_model = create_position_model(input_feature_shape, num_position_classes)
    print("--- Grasp Model Summary ---"); grasp_model.summary()
    print("\n--- Position Model Summary ---"); position_model.summary()

    # --- Training ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print("\n--- Starting Grasp Model Training ---")
    grasp_history = grasp_model.fit(
        X_train_scaled, y_grasp_train, validation_data=(X_val_scaled, y_grasp_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=1
    )

    print("\n--- Starting Position Model Training ---")
    position_history = position_model.fit(
        X_train_scaled, y_position_train, validation_data=(X_val_scaled, y_position_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=1,
        class_weight=position_class_weights_dict # Apply class weights here
    )

    # --- Evaluation ---
    grasp_labels = [f"Grasp {i}" for i in range(num_grasp_classes)]
    position_labels = [f"Pos {i}" for i in range(num_position_classes)]

    evaluate_model(grasp_model, X_test_scaled, y_grasp_test, grasp_labels, "Grasp")
    evaluate_model(position_model, X_test_scaled, y_position_test, position_labels, "Position")

    return grasp_model, grasp_history, position_model, position_history


# --- Execution ---
if __name__ == "__main__":
    # Set random seeds
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPUs,", len(tf.config.experimental.list_logical_devices('GPU')), "Logical GPUs")
        except RuntimeError as e: print(e)

    # Run main async function
    # Note: asyncio.run is used for simplicity here. For complex scenarios, manage the event loop explicitly.
    results = asyncio.run(main_async())

    if results:
        grasp_m, grasp_h, pos_m, pos_h = results
        print("\nAsync pipeline finished successfully.")
        # Optional saving etc.
    else:
        print("\nAsync pipeline did not complete successfully.")