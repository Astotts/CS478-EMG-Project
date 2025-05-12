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
# Define feature extraction window size globally if needed for checks
FEATURE_WINDOW_SIZE = 150

# --- Helper Functions ---

def _load_emg_data(file_path):
    """Loads EMG data from an HDF5 file (Synchronous)."""
    # Keep this function as is
    try:
        with h5py.File(file_path, 'r') as f:
            # Ensure keys are integers for consistency
            data = {int(key): np.array(f[key]) for key in f.keys()}
            # Basic validation of loaded data types/dims
            valid_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] > 0 and value.shape[1] > 0:
                    valid_data[key] = value
                # else:
                    # print(f"Debug: Invalid data format for key {key} in {file_path}, shape={getattr(value, 'shape', type(value))}")
            return valid_data
    except FileNotFoundError:
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
        # Await both concurrently
        emg_data, trials_df = await asyncio.gather(emg_data_task, trials_df_task)
    except FileNotFoundError as e:
            print(f"Error: CSV file not found at {csv_path}")
            error_occurred = True
            # Ensure HDF5 task is awaited if CSV failed during gather
            if not emg_data_task.done(): await emg_data_task
            emg_data, trials_df = None, None # Ensure None return
    except Exception as e:
            print(f"Error during async loading gather in {subject_path}: {str(e)}")
            error_occurred = True
            # Await any pending task safely
            if emg_data_task and not emg_data_task.done(): await emg_data_task
            emg_data, trials_df = None, None # Ensure None return


    # Handle potential loading errors after gather/exceptions
    if emg_data is None or trials_df is None or error_occurred:
       # print(f"Warning: Data files not found or failed to load in {subject_path}")
       return None # Return None to indicate failure

    # --- Basic Validation ---
    if not isinstance(emg_data, dict) or not emg_data:
        # _load_emg_data now returns empty dict if no valid signals found, handle this
        print(f"Warning: No valid EMG data loaded/found from {hdf5_path}")
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
            # Check 1: Trial exists in successfully loaded *and validated* EMG data
            if trial_idx not in emg_data:
                # print(f"Debug: Trial {trial_idx} not in validated emg_data for {subject_path}")
                continue

            # Check 2: Corresponding row exists in CSV
            if trial_idx >= num_csv_rows:
                # print(f"Warning: Trial index {trial_idx} >= num_csv_rows {num_csv_rows} in {subject_path}. Skipping.")
                continue

            # --- Get data and labels ---
            current_signal = emg_data[trial_idx] # Already validated in _load_emg_data
            trial_info = trials_df.iloc[trial_idx]

            # Check 3: Label columns have non-null values (optional but good)
            if pd.isna(trial_info['grasp']) or pd.isna(trial_info['target_position']):
                 print(f"Warning: NaN label found for trial {trial_idx} in {csv_path}. Skipping.")
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
             # Should be caught by column check above, but belt-and-suspenders
             print(f"Error: Column 'grasp' or 'target_position' missing for trial {trial_idx} access in {subject_path}")
             continue
        except Exception as e:
             print(f"Error processing trial {trial_idx} during loading prep in {subject_path}: {str(e)}")
             continue # Skip this trial

    if not X_raw: # If no valid trials were processed for this block
        # print(f"Warning: No valid raw trials prepared for {subject_path}")
        return None

    # Return dicts keyed by original trial index and list of valid indices found
    return X_raw, y_grasp_dict, y_position_dict, sorted(valid_trial_indices)


# --- Preprocessing and Feature Extraction (Keep as previously corrected) ---
def preprocess_emg(emg_signals_list, fs=2000):
    """Processes a list of raw EMG signals."""
    processed_signals = []
    original_indices = [] # Keep track of index in original list
    for i, signal in enumerate(emg_signals_list):
        # Basic validation (should be less necessary if upstream is good, but safe)
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
                min_len_filtfilt = 3 * max(len(a), len(b)) # Default for filtfilt
                if rectified.shape[1] >= min_len_filtfilt:
                    smoothed = filtfilt(b, a, rectified, axis=1)
                # else: print(f"Warning: Signal length {rectified.shape[1]} too short for filtfilt at index {i}.") # Can happen if global_min_length is small
            except ValueError as e:
                print(f"Warning: Filter error (cutoff={normalized_cutoff}) at index {i}: {e}.")
        # else: print(f"Warning: Invalid norm cutoff {normalized_cutoff} at index {i}.")

        # --- Normalization ---
        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        std_mask = std > 1e-9 # Avoid division by zero/very small std
        normalized = np.zeros_like(smoothed)
        # Only divide where std is non-zero, leaves zeros otherwise
        np.divide(smoothed - mean, std, out=normalized, where=std_mask) # Add epsilon in denominator? No, where handles it.

        processed_signals.append(normalized)
        original_indices.append(i) # Store the original index of the signal processed

    return processed_signals, original_indices # Return list of processed signals and their original indices

def extract_features(processed_signals_list, window_size=FEATURE_WINDOW_SIZE, overlap=0.5):
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
        # Check if signal length is sufficient for *at least one* window
        if signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_size:
            # print(f"Warning: Skipping signal index {idx} with shape {signal.shape} in extract_features (invalid shape or too short for a window).")
            continue

        signal_length = signal.shape[1]
        num_windows = (signal_length - window_size) // step_size + 1
        # This check is now redundant due to the check above, but harmless
        # if num_windows <= 0: continue

        # --- Feature extraction per window ---
        window_features_for_trial = []
        valid_window_count = 0
        for i in range(num_windows):
            start = i * step_size; end = start + window_size
            # Safety check for floating point issues, should not happen with //
            if end > signal_length: end = signal_length
            window = signal[:, start:end]
            # Ensure window has the exact size (important for FFT etc.)
            if window.shape[1] != window_size: continue

            try:
                # Calculate features (ensure robustness)
                mean_abs_val = np.mean(np.abs(window), axis=1)
                var = np.var(window, axis=1)
                rms = np.sqrt(np.mean(window**2, axis=1))

                # Handle diff for single-point windows edge case (though window_size should prevent this)
                if window.shape[1] > 1:
                    diff_sig = np.diff(window, axis=1)
                    waveform_length = np.sum(np.abs(diff_sig), axis=1)
                    # Add epsilon to sign to avoid sign(0)=0 issues if signal is flat zero
                    zero_crossings = np.sum(np.diff(np.sign(window + 1e-9), axis=1) != 0, axis=1) / window_size
                else:
                    waveform_length = np.zeros(num_channels)
                    zero_crossings = np.zeros(num_channels)

                # Handle FFT edge cases
                if window_size > 0:
                    fft_val = np.fft.fft(window, axis=1)
                    # Use only positive frequencies (excluding DC potentially)
                    fft_abs = np.abs(fft_val[:, 1:window_size//2 + 1]) # Start from 1 to exclude DC? Or keep DC? Keep DC for now.
                    fft_abs = np.abs(fft_val[:, :window_size//2]) # Original was likely fine

                    if fft_abs.shape[1] > 0: # Ensure there are frequency components
                         mean_freq = np.mean(fft_abs, axis=1)
                         freq_std = np.std(fft_abs, axis=1)
                    else:
                         mean_freq = np.zeros(num_channels); freq_std = np.zeros(num_channels)
                else:
                     mean_freq = np.zeros(num_channels); freq_std = np.zeros(num_channels)


                window_feature = np.concatenate([
                    mean_abs_val, var, rms, waveform_length, zero_crossings,
                    mean_freq, freq_std])

                # Check for NaN/Inf *after* concatenation for the window
                if np.any(np.isnan(window_feature)) or np.any(np.isinf(window_feature)):
                    # print(f"Debug: NaN/Inf found in window {i} for signal index {idx}. Skipping window.")
                    continue # Skip this window

                window_features_for_trial.append(window_feature)
                valid_window_count += 1

            except Exception as e:
                 # print(f"Error calculating features for window {i}, signal {idx}: {e}")
                 continue # Skip window on error

        # Only proceed if at least one window yielded valid features
        if valid_window_count > 0:
            # Average features across valid windows for the trial
            trial_feature_vector = np.mean(window_features_for_trial, axis=0)

            # Final check for NaN/Inf in the averaged vector
            if not (np.any(np.isnan(trial_feature_vector)) or np.any(np.isinf(trial_feature_vector))):
                features.append(trial_feature_vector)
                processed_indices_map.append(idx) # Store index from input list (processed_signals_list)
            # else:
                # print(f"Debug: NaN/Inf found in final averaged features for signal index {idx}. Skipping trial.")
        # else:
            # print(f"Debug: No valid windows found for signal index {idx}. Skipping trial.")


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
        # Ensure labels cover all possible classes 0..N-1
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
        # Ensure target_names matches the potential range of labels
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
    """Main pipeline: Load -> Find Global Min Length -> Process (Pad/Truncate) -> Split -> Train -> Evaluate."""

    tasks_to_process = []
    for subject in range(1, 9): # Subjects 1-8
        for day in [1, 2]:
            for block in [1, 2]:
                tasks_to_process.append((subject, day, block))

    # --- Phase 1: Asynchronous Loading ---
    print(f"--- Phase 1: Starting Data Loading ({len(tasks_to_process)} blocks) ---")
    loading_tasks = []
    for subject, day, block in tasks_to_process:
        path = os.path.join(BASE_DATA_PATH, f"participant_{subject}", f"participant{subject}_day{day}_block{block}")
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
    # This still determines the target length for padding/truncation
    print("\n--- Phase 2a: Finding Global Minimum Signal Length ---")
    all_valid_lengths = []
    num_loaded_signals = 0
    for X_raw_block, _, _, valid_indices in block_results_raw:
          for idx in valid_indices:
               # Check index exists in dict and signal is valid numpy array
               if idx in X_raw_block and isinstance(X_raw_block[idx], np.ndarray) and X_raw_block[idx].ndim == 2:
                   num_loaded_signals += 1
                   all_valid_lengths.append(X_raw_block[idx].shape[1])

    if not all_valid_lengths:
        print("Error: Could not find any valid signals across all blocks to determine minimum length. Exiting.")
        return None, None, None, None

    global_min_length = min(all_valid_lengths)
    max_length_found = max(all_valid_lengths) # Good to know the range
    print(f"Found {len(all_valid_lengths)} valid signals (out of {num_loaded_signals} initially loaded pairs).")
    print(f"Signal Length Stats: Min={global_min_length}, Max={max_length_found}, Avg={np.mean(all_valid_lengths):.2f}")

    # --- Sanity Check: Ensure global_min_length is viable for feature extraction ---
    if global_min_length < FEATURE_WINDOW_SIZE:
         print(f"\nCRITICAL WARNING: Global minimum length ({global_min_length}) is less than feature window size ({FEATURE_WINDOW_SIZE}).")
         print(f"Padding will occur, but feature extraction requires at least {FEATURE_WINDOW_SIZE} samples.")
         print(f"Consider adjusting FEATURE_WINDOW_SIZE or investigating the short signals.")
         # Decide whether to proceed or exit
         # Option 1: Exit
         # return None, None, None, None
         # Option 2: Proceed with caution (feature extraction will skip signals padded to less than window_size)
         pass


    # --- Phase 2b: Sequential Processing (Pad/Truncate, Preprocess, Feature Extract) ---
    print(f"\n--- Phase 2b: Processing Data from {len(block_results_raw)} blocks (Target Length: {global_min_length}) ---")
    all_features_list = []
    all_y_grasp_processed_list = []
    all_y_position_processed_list = []
    total_signals_processed_count = 0
    total_signals_skipped_padding_count = 0
    total_signals_failed_prep_feat_count = 0


    # Use standard tqdm for the sequential loop
    for block_idx, block_data in enumerate(sync_tqdm(block_results_raw, desc="Processing Blocks")):
        X_raw_block, y_grasp_dict, y_position_dict, valid_indices = block_data

        # --- Determine number of channels from the first valid signal in the block ---
        num_channels_block = 0
        first_valid_signal_key = -1
        for trial_idx_check in valid_indices:
             if trial_idx_check in X_raw_block and isinstance(X_raw_block[trial_idx_check], np.ndarray) and X_raw_block[trial_idx_check].ndim == 2:
                num_channels_block = X_raw_block[trial_idx_check].shape[0]
                first_valid_signal_key = trial_idx_check
                if num_channels_block > 0: break # Found channel count
        if num_channels_block == 0:
            # print(f"Warning: Could not determine channel count for block {block_idx}. Skipping block.")
            continue # Skip block if no valid signals to get channel count


        # 1. Prepare signals for preprocessing (Truncate OR Pad)
        signals_to_preprocess = []
        labels_grasp_prepared = []
        labels_pos_prepared = []

        for i, trial_idx in enumerate(valid_indices):
            # Retrieve signal, assume it exists if in valid_indices and X_raw_block
            signal = X_raw_block[trial_idx]

            # Validate shape consistency within the block
            if signal.ndim != 2 or signal.shape[0] != num_channels_block:
                 # print(f"Debug Block {block_idx}, Trial {trial_idx}: Inconsistent channel count ({signal.shape[0]} vs {num_channels_block}). Skipping.")
                 total_signals_skipped_padding_count += 1
                 continue

            current_length = signal.shape[1]
            processed_signal = None

            if current_length == global_min_length:
                processed_signal = signal # No change needed
            elif current_length > global_min_length:
                # Truncate if longer
                processed_signal = signal[:, :global_min_length]
            elif current_length > 0: # Only pad if there's *some* data
                # Pad if shorter (but not empty)
                pad_width = global_min_length - current_length
                padding = ((0, 0), (0, pad_width)) # Pad channels axis=0 none, time axis=1 at the end
                try:
                    processed_signal = np.pad(signal, pad_width=padding, mode='constant', constant_values=0)
                except Exception as e:
                     print(f"Error padding Block {block_idx}, Trial {trial_idx} (len {current_length}): {e}. Skipping.")
                     total_signals_skipped_padding_count += 1
                     continue
            else:
                # Skip if signal is empty (length 0)
                # print(f"Debug Block {block_idx}, Trial {trial_idx}: Skipping due to zero length.")
                total_signals_skipped_padding_count += 1
                continue

            # Check final shape consistency after padding/truncating
            if processed_signal is not None and processed_signal.shape == (num_channels_block, global_min_length):
                signals_to_preprocess.append(processed_signal)
                labels_grasp_prepared.append(y_grasp_dict[trial_idx])
                labels_pos_prepared.append(y_position_dict[trial_idx])
            else:
                 # print(f"Warning: Unexpected signal shape {getattr(processed_signal, 'shape', 'None')} after processing Block {block_idx}, Trial {trial_idx}. Expected {(num_channels_block, global_min_length)}. Skipping.")
                 total_signals_skipped_padding_count += 1


        if not signals_to_preprocess: continue # Skip block if no signals prepared

        # 2. Preprocess prepared signals
        processed_signals, indices_after_preprocess = preprocess_emg(signals_to_preprocess)
        if not processed_signals:
             total_signals_failed_prep_feat_count += len(signals_to_preprocess)
             continue

        # Select labels corresponding to successfully preprocessed signals
        labels_grasp_preprocessed = np.array(labels_grasp_prepared)[indices_after_preprocess]
        labels_pos_preprocessed = np.array(labels_pos_prepared)[indices_after_preprocess]

        # Count signals lost during preprocessing
        lost_in_prep = len(signals_to_preprocess) - len(processed_signals)
        total_signals_failed_prep_feat_count += lost_in_prep


        # 3. Extract Features
        features_block, indices_after_features = extract_features(processed_signals)
        if features_block.size == 0:
            total_signals_failed_prep_feat_count += len(processed_signals) # All remaining failed here
            continue

        # Select labels corresponding to successfully feature-extracted signals
        final_y_grasp = labels_grasp_preprocessed[indices_after_features]
        final_y_position = labels_pos_preprocessed[indices_after_features]

        # Count signals lost during feature extraction
        lost_in_feat = len(processed_signals) - features_block.shape[0]
        total_signals_failed_prep_feat_count += lost_in_feat


        # Final check for consistency before appending
        if features_block.shape[0] == final_y_grasp.shape[0] == final_y_position.shape[0]:
            all_features_list.append(features_block)
            all_y_grasp_processed_list.append(final_y_grasp)
            all_y_position_processed_list.append(final_y_position)
            total_signals_processed_count += features_block.shape[0] # Add count of successfully processed signals
        else:
            print(f"CRITICAL WARNING: Mismatch after final label selection in block {block_idx}. This should not happen!")
            # Count these as lost too
            total_signals_failed_prep_feat_count += features_block.shape[0]



    print(f"--- Phase 2 Finished: Processing complete ---")
    print(f"Total signals successfully processed (features extracted): {total_signals_processed_count}")
    print(f"Total signals skipped before/during padding: {total_signals_skipped_padding_count}")
    print(f"Total signals lost during preprocessing/feature extraction: {total_signals_failed_prep_feat_count}")


    if not all_features_list:
        print("Error: No features extracted successfully from any block after processing. Exiting.")
        return None, None, None, None

    # --- Phase 3: Concatenate, Split, Train, Evaluate ---
    print("\n--- Phase 3: Final Data Prep, Training & Evaluation ---")
    try:
        # Concatenate features and labels from all blocks
        all_X = np.concatenate(all_features_list, axis=0)
        all_y_grasp = np.concatenate(all_y_grasp_processed_list, axis=0)
        all_y_position = np.concatenate(all_y_position_processed_list, axis=0)
    except ValueError as e:
        print(f"Error concatenating final data: {e}")
        # Add debug info about list contents
        print("List lengths:", len(all_features_list), len(all_y_grasp_processed_list), len(all_y_position_processed_list))
        for i in range(min(5, len(all_features_list))): # Print shapes of first few elements
             print(f" Elem {i} shapes: F={all_features_list[i].shape}, G={all_y_grasp_processed_list[i].shape}, P={all_y_position_processed_list[i].shape}")
        return None, None, None, None

    # *** This count should now match total_signals_processed_count ***
    print(f"Total usable samples (after concat): {all_X.shape[0]}")
    if all_X.shape[0] == 0: print("Error: No samples available for training."); return None, None, None, None

    print(f"Feature shape: {all_X.shape[1:]}")
    # Display final distribution going into train/test split
    print("Final Grasp Label Distribution:", Counter(all_y_grasp))
    print("Final Position Label Distribution:", Counter(all_y_position))

    # --- Single Data Split ---
    # Check if stratification is possible
    min_samples_per_grasp_class = min(Counter(all_y_grasp).values()) if all_y_grasp.size > 0 else 0
    # Need at least 2 samples per class for stratification in train/test and then train/val
    can_stratify = min_samples_per_grasp_class >= 2

    print(f"\nSplitting data into Train/Validation/Test sets...")
    stratify_option = all_y_grasp if can_stratify else None
    if not can_stratify and all_y_grasp.size > 0:
         print("Warning: Cannot stratify by grasp due to insufficient samples in some classes. Splitting without stratification.")

    try:
        # Split into Train (60%) / Validation (20%) / Test (20%)
        (X_train_val, X_test,
         y_grasp_train_val, y_grasp_test,
         y_position_train_val, y_position_test) = train_test_split(
            all_X, all_y_grasp, all_y_position,
            test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_option
        )

        # Further split Train_Val into Train and Validation
        # Adjust test_size here to get 20% validation from the original total (0.25 * 0.8 = 0.2)
        stratify_option_val = y_grasp_train_val if can_stratify else None
        if not can_stratify and y_grasp_train_val.size > 0:
             print("Warning: Cannot stratify validation split.")

        (X_train, X_val,
         y_grasp_train, y_grasp_val,
         y_position_train, y_position_val) = train_test_split(
            X_train_val, y_grasp_train_val, y_position_train_val,
            test_size=0.25, random_state=RANDOM_STATE, stratify=stratify_option_val # 0.25 * 0.8 = 0.2
        )

        print(f"Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print("Test Set Grasp Distribution:", Counter(y_grasp_test))
        print("Test Set Position Distribution:", Counter(y_position_test))

    except ValueError as e:
        print(f"Error splitting data: {e}")
        print("Check class distributions and total sample size.")
        print("Grasp Distribution:", Counter(all_y_grasp))
        print("Position Distribution:", Counter(all_y_position))
        return None, None, None, None


    # --- Standardization ---
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Class Weights for Position Model ---
    print("Calculating class weights for Position model (based on Training set)...")
    unique_pos_classes_train = np.unique(y_position_train)
    position_class_weights_dict = None # Default
    if len(unique_pos_classes_train) > 1: # Ensure there's more than one class to weight
         try:
             pos_class_weights = compute_class_weight(
                 class_weight='balanced',
                 classes=unique_pos_classes_train,
                 y=y_position_train
             )
             # Create dict mapping class index to weight
             position_class_weights_dict = dict(zip(unique_pos_classes_train, pos_class_weights))
             print("Position Class Weights:", {k: f"{v:.2f}" for k,v in position_class_weights_dict.items()})
         except Exception as e:
             print(f"Error computing class weights: {e}. Proceeding without weights.")
    else:
         print("Skipping class weight calculation: Only one position class found in training data.")


    # --- Model Creation ---
    # Determine number of classes from the *overall* dataset to ensure output layer is correct size
    num_grasp_classes = len(np.unique(all_y_grasp))
    num_position_classes = len(np.unique(all_y_position))
    input_feature_shape = (X_train_scaled.shape[1],)

    print(f"\nCreating models (Grasps={num_grasp_classes}, Positions={num_position_classes})...")
    grasp_model = create_grasp_model(input_feature_shape, num_grasp_classes)
    position_model = create_position_model(input_feature_shape, num_position_classes)
    # print("--- Grasp Model Summary ---"); grasp_model.summary()
    # print("\n--- Position Model Summary ---"); position_model.summary()

    # --- Training ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print("\n--- Starting Grasp Model Training ---")
    # Check if training data is available
    if X_train_scaled.size == 0 or y_grasp_train.size == 0:
         print("Skipping Grasp training: No training data.")
         grasp_history = None
    else:
         grasp_history = grasp_model.fit(
             X_train_scaled, y_grasp_train, validation_data=(X_val_scaled, y_grasp_val),
             epochs=100, batch_size=64, callbacks=callbacks, verbose=1
         )

    print("\n--- Starting Position Model Training ---")
    # Check if training data is available
    if X_train_scaled.size == 0 or y_position_train.size == 0:
         print("Skipping Position training: No training data.")
         position_history = None
    else:
         position_history = position_model.fit(
             X_train_scaled, y_position_train, validation_data=(X_val_scaled, y_position_val),
             epochs=100, batch_size=64, callbacks=callbacks, verbose=1,
             class_weight=position_class_weights_dict # Apply class weights here
         )

    # --- Evaluation ---
    # Use all possible classes found in the original dataset for labels
    grasp_labels = [f"Grasp {i}" for i in range(num_grasp_classes)]
    position_labels = [f"Pos {i}" for i in range(num_position_classes)]

    # Check if test data available before evaluating
    if X_test_scaled.size > 0 and y_grasp_test.size > 0:
         evaluate_model(grasp_model, X_test_scaled, y_grasp_test, grasp_labels, "Grasp")
    else:
         print("\nSkipping Grasp evaluation: No test data.")

    if X_test_scaled.size > 0 and y_position_test.size > 0:
         evaluate_model(position_model, X_test_scaled, y_position_test, position_labels, "Position")
    else:
         print("\nSkipping Position evaluation: No test data.")


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
    results = asyncio.run(main_async())

    if results:
        grasp_m, grasp_h, pos_m, pos_h = results
        print("\nAsync pipeline finished successfully.")
        # Optional saving etc.
        # if grasp_m: grasp_m.save("grasp_model.keras")
        # if pos_m: pos_m.save("position_model.keras")
    else:
        print("\nAsync pipeline did not complete successfully or encountered errors.")