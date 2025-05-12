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
import re # For parsing parameters file
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm as sync_tqdm
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# --- Global Configuration ---
BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data" # <-- SET YOUR PATH!
RANDOM_STATE = 42
# Time-based windowing parameters
WINDOW_DURATION_S = 0.150 # seconds (e.g., 150ms)
OVERLAP_RATIO = 0.5

# --- Helper Functions ---

def _load_generic_hdf5_data(file_path):
    """Loads generic HDF5 data keyed by integer trial index."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Convert keys to int for consistency
            data = {int(key): np.array(f[key]) for key in f.keys()}
            # Basic validation of content
            for key, value in data.items():
                if not isinstance(value, np.ndarray):
                    print(f"Warning: Non-numpy array found for key {key} in {file_path}")
                    return None # Or handle differently
                if value.ndim == 0: # Skip scalar datasets if they exist
                     print(f"Warning: Scalar dataset found for key {key} in {file_path}. Skipping this key.")
                     del data[key] # Remove problematic scalar entry

            return data if data else None # Return None if empty after filtering
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading generic HDF5 file {file_path}: {str(e)}")
        return None

def _parse_recording_parameters(file_path):
    """Parses key parameters from the recording_parameters.txt file."""
    params = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_').replace('(sec)', '').strip('_')
                    value = value.strip()
                    try:
                        # Try converting to float, fallback to string
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error parsing parameters file {file_path}: {e}")
        return None
    # Basic check for essential parameter
    if 'trial_length_time' not in params:
        print(f"Warning: 'Trial length time (sec)' not found in {file_path}")
        return None
    return params

async def load_multimodal_data_async(subject_path):
    """Loads RAW EMG, Finger, Glove, Labels, and Parameters asynchronously."""
    loop = asyncio.get_running_loop()
    emg_path = f"{subject_path}/emg_data.hdf5"
    finger_path = f"{subject_path}/finger_data.hdf5"
    glove_path = f"{subject_path}/glove_data.hdf5"
    csv_path = f"{subject_path}/trials.csv"
    params_path = f"{subject_path}/recording_parameters.txt"

    # Create tasks for all loading operations
    emg_task = loop.run_in_executor(None, _load_generic_hdf5_data, emg_path)
    finger_task = loop.run_in_executor(None, _load_generic_hdf5_data, finger_path)
    glove_task = loop.run_in_executor(None, _load_generic_hdf5_data, glove_path)
    params_task = loop.run_in_executor(None, _parse_recording_parameters, params_path)
    try:
        # Wrap pandas read_csv in executor as well
        trials_df_task = loop.run_in_executor(None, pd.read_csv, csv_path)
        # Gather all results
        results = await asyncio.gather(
            emg_task, finger_task, glove_task, params_task, trials_df_task,
            return_exceptions=True # Allow specific tasks to fail without stopping others
        )
    except Exception as e:
        # This catch might be less likely if return_exceptions=True
        print(f"Critical error during asyncio.gather for {subject_path}: {e}")
        return None

    # Unpack results and check for errors/None
    emg_data, finger_data, glove_data, params, trials_df = results

    # Check for exceptions returned by gather
    if isinstance(emg_data, Exception): print(f"EMG Load Error ({subject_path}): {emg_data}"); emg_data = None
    if isinstance(finger_data, Exception): print(f"Finger Load Error ({subject_path}): {finger_data}"); finger_data = None
    if isinstance(glove_data, Exception): print(f"Glove Load Error ({subject_path}): {glove_data}"); glove_data = None
    if isinstance(params, Exception): print(f"Params Load Error ({subject_path}): {params}"); params = None
    if isinstance(trials_df, Exception): print(f"CSV Load Error ({subject_path}): {trials_df}"); trials_df = None

    # --- Essential Data Check ---
    # Require EMG, Labels, and Parameters for basic processing
    if emg_data is None or trials_df is None or params is None:
        # print(f"Warning: Missing essential data (EMG/CSV/Params) for {subject_path}. Skipping block.")
        return None
    # Finger and Glove are optional, but we need to handle their absence downstream
    if finger_data is None:
        print(f"Warning: Finger data missing or failed to load for {subject_path}.")
    if glove_data is None:
         print(f"Warning: Glove data missing or failed to load for {subject_path}.")

    # --- Basic Validation of loaded data ---
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty: return None # Invalid labels CSV
    if 'grasp' not in trials_df.columns or 'target_position' not in trials_df.columns: return None # Missing label cols
    if not isinstance(emg_data, dict) or not emg_data: return None # Invalid EMG data

    # --- Prepare Output Dictionaries ---
    X_emg_raw = {}
    X_finger_raw = {}
    X_glove_raw = {}
    y_grasp_dict = {}
    y_position_dict = {}
    valid_trial_indices = []
    num_csv_rows = len(trials_df)

    # Find common trial indices present in EMG AND CSV
    # Optional: Also require Finger/Glove if needed, currently treat as optional
    potential_indices = set(emg_data.keys()) & set(range(num_csv_rows))
    # Add checks for optional data only if it loaded successfully
    if finger_data: potential_indices &= set(finger_data.keys())
    if glove_data: potential_indices &= set(glove_data.keys())

    for trial_idx in sorted(list(potential_indices)): # Iterate through common, valid indices
        try:
            # Basic validity check (non-empty 2D array) for EMG is crucial
            emg_signal = emg_data[trial_idx]
            if not isinstance(emg_signal, np.ndarray) or emg_signal.ndim != 2 or emg_signal.shape[0] == 0 or emg_signal.shape[1] == 0:
                 print(f"Warning: Invalid EMG signal (idx {trial_idx}) in {subject_path}.")
                 continue

            # Validity checks for optional Finger data
            finger_signal = None
            if finger_data and trial_idx in finger_data:
                 finger_signal_raw = finger_data[trial_idx]
                 if isinstance(finger_signal_raw, np.ndarray) and finger_signal_raw.ndim >= 1 and finger_signal_raw.size > 0:
                       # Ensure 2D for consistency (channels, time) - reshape if 1D
                       finger_signal = finger_signal_raw.reshape(-1, finger_signal_raw.shape[-1]) if finger_signal_raw.ndim == 1 else finger_signal_raw
                       if finger_signal.ndim != 2: finger_signal = None # Invalid shape after reshape
                 # else: print(f"Debug: Invalid Finger signal (idx {trial_idx})")

            # Validity checks for optional Glove data
            glove_signal = None
            if glove_data and trial_idx in glove_data:
                 glove_signal_raw = glove_data[trial_idx]
                 if isinstance(glove_signal_raw, np.ndarray) and glove_signal_raw.ndim >= 1 and glove_signal_raw.size > 0:
                       glove_signal = glove_signal_raw.reshape(-1, glove_signal_raw.shape[-1]) if glove_signal_raw.ndim == 1 else glove_signal_raw
                       if glove_signal.ndim != 2: glove_signal = None # Invalid shape after reshape
                 # else: print(f"Debug: Invalid Glove signal (idx {trial_idx})")


            # Store data if EMG is valid (Finger/Glove are optional)
            trial_info = trials_df.iloc[trial_idx]
            X_emg_raw[trial_idx] = emg_signal
            if finger_signal is not None: X_finger_raw[trial_idx] = finger_signal
            if glove_signal is not None: X_glove_raw[trial_idx] = glove_signal
            y_grasp_dict[trial_idx] = int(trial_info['grasp'] - 1)
            y_position_dict[trial_idx] = int(trial_info['target_position'] - 1)
            valid_trial_indices.append(trial_idx)

        except Exception as e:
            print(f"Error processing trial {trial_idx} during multi-modal prep in {subject_path}: {str(e)}")
            continue

    if not valid_trial_indices:
        # print(f"Warning: No valid trials after checking all modalities in {subject_path}")
        return None

    # Return dicts keyed by original trial index that are valid across required modalities
    return (X_emg_raw, X_finger_raw, X_glove_raw), (y_grasp_dict, y_position_dict), params, sorted(valid_trial_indices)

# --- Preprocessing ---
def preprocess_emg(emg_signals_list, fs=2000):
    # Keep as is (robust version from previous)
    processed_signals = []; original_indices = []
    for i, signal in enumerate(emg_signals_list):
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0: continue
        rectified = np.abs(signal); cutoff_freq=20; filter_order=4; nyquist_freq=fs/2; normalized_cutoff=cutoff_freq/nyquist_freq; smoothed=rectified
        if 0 < normalized_cutoff < 1:
            try:
                b, a = butter(filter_order, normalized_cutoff, 'low')
                if rectified.shape[1] >= 3 * max(len(a), len(b)): smoothed = filtfilt(b, a, rectified, axis=1)
            except ValueError: pass
        mean=smoothed.mean(axis=1, keepdims=True); std=smoothed.std(axis=1, keepdims=True); std_mask=std > 1e-9; normalized=np.zeros_like(smoothed)
        np.divide(smoothed - mean, std + 1e-9, out=normalized, where=std_mask); processed_signals.append(normalized); original_indices.append(i)
    return processed_signals, original_indices

def preprocess_generic(signal_list):
    """Generic preprocessing (scaling) for non-EMG signals."""
    processed_signals = []
    original_indices = []
    # Simple approach: scale each signal individually (local scaling)
    # Alternatively, collect all signals and scale globally later (like EMG features)
    # Let's do local scaling for simplicity here. Global scaling is better before ML model.
    # --> Correction: Let's just return the list, scaling will happen GLOBALLY on features later.
    for i, signal in enumerate(signal_list):
         if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
              # print(f"Warning: Skipping invalid generic signal at index {i}")
              continue
         processed_signals.append(signal) # No preprocessing, just pass through valid signals
         original_indices.append(i)
    return processed_signals, original_indices

# --- Feature Extraction ---
def extract_emg_features(processed_signals_list, window_samples, step_samples):
    """Extracts EMG features using specified window/step in samples."""
    features = []; processed_indices_map = []; num_channels = 0
    if not processed_signals_list: return np.array(features), []
    for sig in processed_signals_list: # Find num_channels
        if sig.ndim == 2 and sig.shape[0] > 0: num_channels = sig.shape[0]; break
    if num_channels == 0: return np.array(features), []

    for idx, signal in enumerate(processed_signals_list):
        if signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_samples: continue
        signal_length = signal.shape[1]; num_windows = (signal_length - window_samples) // step_samples + 1
        if num_windows <= 0: continue
        window_features_for_trial = []; valid_window_count = 0
        for i in range(num_windows):
            start = i * step_samples; end = start + window_samples; window = signal[:, start:end]
            if window.shape[1] != window_samples: continue
            try:
                mav=np.mean(np.abs(window),axis=1); var=np.var(window,axis=1); rms=np.sqrt(np.mean(window**2,axis=1))
                diff=np.diff(window,axis=1); wl=np.sum(np.abs(diff),axis=1) if diff.size>0 else np.zeros(num_channels)
                zc=np.sum(np.diff(np.sign(window+1e-9),axis=1)!=0,axis=1)/window_samples if window.shape[1]>1 else np.zeros(num_channels)
                fft=np.fft.fft(window,axis=1); fft_abs=np.abs(fft[:,:window_samples//2])
                if fft_abs.shape[1]==0: mf=np.zeros(num_channels); fs=np.zeros(num_channels)
                else: mf=np.mean(fft_abs,axis=1); fs=np.std(fft_abs,axis=1)
                wf=np.concatenate([mav,var,rms,wl,zc,mf,fs])
                if np.any(np.isnan(wf)) or np.any(np.isinf(wf)): continue
                window_features_for_trial.append(wf); valid_window_count += 1
            except Exception: continue
        if valid_window_count > 0:
            tfv = np.mean(window_features_for_trial, axis=0) # Average features over windows
            if not (np.any(np.isnan(tfv)) or np.any(np.isinf(tfv))):
                features.append(tfv); processed_indices_map.append(idx)
    return np.array(features), processed_indices_map

def extract_generic_features(processed_signals_list, window_samples, step_samples):
    """Extracts simple statistical features for generic signals."""
    features = []; processed_indices_map = []; num_channels = 0
    if not processed_signals_list: return np.array(features), []
    for sig in processed_signals_list: # Find num_channels
        if sig.ndim == 2 and sig.shape[0] > 0: num_channels = sig.shape[0]; break
    if num_channels == 0: return np.array(features), []

    for idx, signal in enumerate(processed_signals_list):
        if signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_samples: continue
        signal_length = signal.shape[1]; num_windows = (signal_length - window_samples) // step_samples + 1
        if num_windows <= 0: continue
        window_features_for_trial = []; valid_window_count = 0
        for i in range(num_windows):
            start = i * step_samples; end = start + window_samples; window = signal[:, start:end]
            if window.shape[1] != window_samples: continue
            try:
                f_mean = np.mean(window, axis=1)
                f_std = np.std(window, axis=1)
                f_min = np.min(window, axis=1)
                f_max = np.max(window, axis=1)
                # Add range = max - min
                f_range = f_max - f_min
                # Maybe add velocity (mean abs diff)?
                # diff_sig = np.diff(window, axis=1)
                # f_mavd = np.mean(np.abs(diff_sig), axis=1) if diff_sig.size > 0 else np.zeros(num_channels)

                wf = np.concatenate([f_mean, f_std, f_min, f_max, f_range]) # Add other features if desired
                if np.any(np.isnan(wf)) or np.any(np.isinf(wf)): continue
                window_features_for_trial.append(wf); valid_window_count += 1
            except Exception: continue
        if valid_window_count > 0:
            tfv = np.mean(window_features_for_trial, axis=0) # Average features over windows
            if not (np.any(np.isnan(tfv)) or np.any(np.isinf(tfv))):
                features.append(tfv); processed_indices_map.append(idx)
    return np.array(features), processed_indices_map

# --- Evaluation and Model Creation Functions ---
def evaluate_model(model, X_test, y_test, labels, task_name):
    # Keep as is
    print(f"\n--- Evaluating {task_name} Model ---")
    try: test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0); print(f"{task_name} Test Accuracy: {test_acc:.4f}\n{task_name} Test Loss: {test_loss:.4f}")
    except Exception as e: print(f"Error evaluate: {e}"); return
    try: y_pred_probs = model.predict(X_test); y_pred_classes = np.argmax(y_pred_probs, axis=1)
    except Exception as e: print(f"Error predict: {e}"); return
    if y_test.size == 0: print("Empty y_test"); return
    try: cm = confusion_matrix(y_test, y_pred_classes, labels=list(range(len(labels)))); plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)*0.8))); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels); plt.xlabel('Predicted Label'); plt.ylabel('Actual Label'); plt.title(f'{task_name} Task - Confusion Matrix'); plt.tight_layout(); plt.show()
    except Exception as e: print(f"Error CM: {e}")
    print(f"\n--- {task_name} Classification Report ---")
    try: print(classification_report(y_test, y_pred_classes, labels=list(range(len(labels))), target_names=labels, zero_division=0))
    except Exception as e: print(f"Error Report: {e}")

def create_grasp_model(input_shape, num_grasps):
    # Keep as is, input_shape will be updated automatically
    inputs=tf.keras.Input(shape=input_shape); x=tf.keras.layers.Dense(128,activation='relu')(inputs); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.Dropout(0.3)(x); x=tf.keras.layers.Dense(256,activation='relu')(x); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.Dropout(0.4)(x); x=tf.keras.layers.Dense(64,activation='relu')(x); x=tf.keras.layers.Dropout(0.3)(x); outputs=tf.keras.layers.Dense(num_grasps,activation='softmax',name='grasp_output')(x); model=tf.keras.Model(inputs=inputs,outputs=outputs); model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']); return model

def create_position_model(input_shape, num_positions):
     # Keep as is, input_shape will be updated automatically
    inputs=tf.keras.Input(shape=input_shape); x=tf.keras.layers.Dense(128,activation='relu')(inputs); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.Dropout(0.3)(x); x=tf.keras.layers.Dense(256,activation='relu')(x); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.Dropout(0.4)(x); x=tf.keras.layers.Dense(64,activation='relu')(x); x=tf.keras.layers.Dropout(0.3)(x); outputs=tf.keras.layers.Dense(num_positions,activation='softmax',name='position_output')(x); model=tf.keras.Model(inputs=inputs,outputs=outputs); model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']); return model

# --- Main Pipeline ---
async def main_async():
    """Main pipeline: Load Multi-Modal -> Estimate Rates -> Sync Time -> Process -> Split -> Train -> Evaluate."""
    tasks_to_process = [(s, d, b) for s in range(1, 9) for d in [1, 2] for b in [1, 2]]

    # --- Phase 1: Asynchronous Loading ---
    print(f"--- Phase 1: Loading Multi-Modal Data ({len(tasks_to_process)} blocks) ---")
    loading_tasks = [load_multimodal_data_async(f"{BASE_DATA_PATH}/participant_{s}/participant{s}_day{d}_block{b}") for s, d, b in tasks_to_process]
    block_results_raw = await async_tqdm.gather(*loading_tasks, desc="Loading Blocks")
    block_results_raw = [res for res in block_results_raw if res is not None]
    if not block_results_raw: print("Error: No data loaded."); return None
    print(f"--- Phase 1 Finished: Loaded raw data from {len(block_results_raw)} blocks ---")

    # --- Phase 2a: Estimate Sampling Rates & Min Duration ---
    print("\n--- Phase 2a: Estimating Sampling Rates & Minimum Duration ---")
    trial_durations = {} # { modality: [durations...] }
    trial_lengths_samples = {} # { modality: {trial_idx_global: num_samples}}
    global_trial_idx = 0
    assumed_trial_length_s = None # Get from first valid params

    for i, block_data in enumerate(block_results_raw):
        (X_emg, X_finger, X_glove), _, params, valid_indices = block_data
        if not assumed_trial_length_s and params and 'trial_length_time' in params:
             assumed_trial_length_s = params['trial_length_time']
             print(f"Using assumed trial length: {assumed_trial_length_s}s (from block {i})")
        if not assumed_trial_length_s: continue # Skip block if no trial length known yet

        for idx in valid_indices:
            current_global_idx = (i, idx) # Unique ID for trial (block_index, trial_index_in_block)
            # EMG
            if idx in X_emg and X_emg[idx].ndim == 2 and X_emg[idx].shape[1] > 0:
                rate = X_emg[idx].shape[1] / assumed_trial_length_s
                trial_durations.setdefault('emg', []).append(assumed_trial_length_s) # Assuming all cover full duration
                trial_lengths_samples.setdefault('emg', {})[current_global_idx] = X_emg[idx].shape[1]
            # Finger
            if idx in X_finger and X_finger[idx].ndim == 2 and X_finger[idx].shape[1] > 0:
                 rate = X_finger[idx].shape[1] / assumed_trial_length_s
                 trial_durations.setdefault('finger', []).append(assumed_trial_length_s)
                 trial_lengths_samples.setdefault('finger', {})[current_global_idx] = X_finger[idx].shape[1]
            # Glove
            if idx in X_glove and X_glove[idx].ndim == 2 and X_glove[idx].shape[1] > 0:
                rate = X_glove[idx].shape[1] / assumed_trial_length_s
                trial_durations.setdefault('glove', []).append(assumed_trial_length_s)
                trial_lengths_samples.setdefault('glove', {})[current_global_idx] = X_glove[idx].shape[1]

    if not assumed_trial_length_s or not trial_durations: print("Error: Cannot determine trial length or no valid trials."); return None

    # Estimate average sampling rates
    fs_estimates = {}
    print("Estimated Sampling Rates (Hz):")
    for modality, lengths_dict in trial_lengths_samples.items():
         if lengths_dict:
              avg_samples = np.mean(list(lengths_dict.values()))
              fs_estimates[modality] = avg_samples / assumed_trial_length_s
              print(f"  {modality.capitalize()}: {fs_estimates[modality]:.2f}")
         else:
             print(f"  {modality.capitalize()}: No samples found.")

    # Set default EMG rate if estimate is unreliable or missing (based on previous code)
    fs_emg = fs_estimates.get('emg', 2000.0)
    if fs_estimates.get('emg') is None: print("Warning: EMG sampling rate not estimated, using default 2000.0 Hz")
    fs_finger = fs_estimates.get('finger')
    fs_glove = fs_estimates.get('glove')

    # Determine Minimum Duration (using assumed full duration for now)
    # A more robust way would be min(num_samples / fs) if fs was certain
    min_duration_s = assumed_trial_length_s # Simplification: assume all valid trials cover the target duration
    print(f"Using minimum synchronized duration: {min_duration_s}s")

    # Calculate target samples per modality for truncation
    target_samples = {}
    if fs_emg: target_samples['emg'] = int(min_duration_s * fs_emg)
    if fs_finger: target_samples['finger'] = int(min_duration_s * fs_finger)
    if fs_glove: target_samples['glove'] = int(min_duration_s * fs_glove)
    print("Target samples per modality after truncation:", target_samples)

    # --- Phase 2b: Sequential Processing ---
    print(f"\n--- Phase 2b: Processing Multi-Modal Data ---")
    all_combined_features = []
    all_final_y_grasp = []
    all_final_y_position = []

    for block_data in sync_tqdm(block_results_raw, desc="Processing Blocks"):
        (X_emg_raw, X_finger_raw, X_glove_raw), (y_grasp_dict, y_pos_dict), _, valid_indices = block_data

        block_features_emg = {}; block_features_finger = {}; block_features_glove = {}
        processed_indices_in_block = set() # Track indices surviving all steps for this block

        # 1. Prepare lists for each modality, applying truncation
        list_emg = []; list_finger = []; list_glove = []
        list_grasp = []; list_pos = []; list_orig_indices = []

        for trial_idx in valid_indices:
            valid_emg = idx_emg = False; valid_finger = idx_finger = False; valid_glove = idx_glove = False
            sig_emg = sig_finger = sig_glove = None

            # Check EMG
            if 'emg' in target_samples and trial_idx in X_emg_raw and X_emg_raw[trial_idx].shape[1] >= target_samples['emg']:
                sig_emg = X_emg_raw[trial_idx][:, :target_samples['emg']]; valid_emg = True
            # Check Finger (Optional)
            if fs_finger and 'finger' in target_samples and trial_idx in X_finger_raw and X_finger_raw[trial_idx].shape[1] >= target_samples['finger']:
                 sig_finger = X_finger_raw[trial_idx][:, :target_samples['finger']]; valid_finger = True
            # Check Glove (Optional)
            if fs_glove and 'glove' in target_samples and trial_idx in X_glove_raw and X_glove_raw[trial_idx].shape[1] >= target_samples['glove']:
                 sig_glove = X_glove_raw[trial_idx][:, :target_samples['glove']]; valid_glove = True

            # Add trial only if EMG is valid (Finger/Glove are optional but included if valid)
            if valid_emg:
                 list_emg.append(sig_emg)
                 list_finger.append(sig_finger if valid_finger else None) # Append None if optional data invalid/missing
                 list_glove.append(sig_glove if valid_glove else None)
                 list_grasp.append(y_grasp_dict[trial_idx])
                 list_pos.append(y_pos_dict[trial_idx])
                 list_orig_indices.append(trial_idx) # Keep track of original trial index

        if not list_emg: continue # Skip block if no valid EMG trials

        # 2. Preprocess each modality
        emg_processed, idx_emg = preprocess_emg(list_emg, fs=fs_emg)
        # For generic, pass through only valid (non-None) signals
        finger_signals_to_process = [s for i, s in enumerate(list_finger) if i in idx_emg and s is not None]
        glove_signals_to_process = [s for i, s in enumerate(list_glove) if i in idx_emg and s is not None]
        finger_processed, idx_f_rel = preprocess_generic(finger_signals_to_process) # Indices relative to non-None list
        glove_processed, idx_g_rel = preprocess_generic(glove_signals_to_process)

        # Map relative finger/glove indices back to original emg indices
        map_idx_emg_to_finger = {orig_idx: i for i, orig_idx in enumerate(idx_emg) if list_finger[orig_idx] is not None}
        map_idx_emg_to_glove = {orig_idx: i for i, orig_idx in enumerate(idx_emg) if list_glove[orig_idx] is not None}
        idx_f = [k for k, v in map_idx_emg_to_finger.items() if v in idx_f_rel] # Indices from idx_emg that have valid finger data
        idx_g = [k for k, v in map_idx_emg_to_glove.items() if v in idx_g_rel] # Indices from idx_emg that have valid glove data


        # 3. Extract Features (using time windows)
        # Calculate window/step samples per modality
        window_samples = {}; step_samples = {}
        if fs_emg: window_samples['emg'] = int(WINDOW_DURATION_S * fs_emg); step_samples['emg'] = int(window_samples['emg'] * (1.0 - OVERLAP_RATIO))
        if fs_finger: window_samples['finger'] = int(WINDOW_DURATION_S * fs_finger); step_samples['finger'] = int(window_samples['finger'] * (1.0 - OVERLAP_RATIO))
        if fs_glove: window_samples['glove'] = int(WINDOW_DURATION_S * fs_glove); step_samples['glove'] = int(window_samples['glove'] * (1.0 - OVERLAP_RATIO))

        # Ensure step size is at least 1
        for mod in step_samples: step_samples[mod] = max(1, step_samples[mod])

        features_emg, idx_feat_emg_rel = extract_emg_features(emg_processed, window_samples['emg'], step_samples['emg'])
        final_indices_emg = np.array(idx_emg)[idx_feat_emg_rel] # Indices relative to original list_emg

        features_finger, idx_feat_finger_rel = extract_generic_features(finger_processed, window_samples.get('finger', 0), step_samples.get('finger', 1)) if fs_finger else (np.array([]), [])
        final_indices_finger = np.array(idx_f)[idx_feat_finger_rel] # Indices relative to original list_emg

        features_glove, idx_feat_glove_rel = extract_generic_features(glove_processed, window_samples.get('glove', 0), step_samples.get('glove', 1)) if fs_glove else (np.array([]), [])
        final_indices_glove = np.array(idx_g)[idx_feat_glove_rel] # Indices relative to original list_emg

        # 4. Align Features and Labels
        # Find common indices that survived all processing steps FOR THAT MODALITY
        common_indices = set(final_indices_emg) # Start with EMG indices
        # Convert optional feature arrays to dictionaries for easier lookup
        features_finger_dict = {idx: feat for idx, feat in zip(final_indices_finger, features_finger)}
        features_glove_dict = {idx: feat for idx, feat in zip(final_indices_glove, features_glove)}
        features_emg_dict = {idx: feat for idx, feat in zip(final_indices_emg, features_emg)}

        block_combined_features = []
        block_y_grasp = []
        block_y_position = []

        list_orig_indices_arr = np.array(list_orig_indices) # Original trial indices for the lists fed into preprocessing

        for final_idx in sorted(list(common_indices)): # final_idx is index relative to list_emg etc.
            feat_emg = features_emg_dict.get(final_idx)
            # Get optional features, use zeros if missing (needs defined dimensionality)
            # --> Simpler: only combine if *all* desired modalities are present?
            # --> Let's require EMG, and optionally include finger/glove if available

            current_features = [feat_emg]
            if final_idx in features_finger_dict: current_features.append(features_finger_dict[final_idx])
            # else: Pad with zeros? For now, only include if present. Might lead to variable feature length if not careful.
            if final_idx in features_glove_dict: current_features.append(features_glove_dict[final_idx])
            # else: Pad?

            # Check if feature lists are non-empty before concatenating
            if not all(f is not None for f in current_features): continue # Skip if EMG somehow failed

            try:
                # Ensure consistent feature lengths before concatenating if padding isn't used
                # This logic assumes features_finger/glove_dict only contain valid features
                 combined_f = np.concatenate(current_features)

                 block_combined_features.append(combined_f)
                 # Get corresponding labels using the index relative to list_emg/list_grasp/list_pos
                 block_y_grasp.append(list_grasp[final_idx])
                 block_y_position.append(list_pos[final_idx])
            except ValueError as e:
                 print(f"Warning: Error concatenating features for index {final_idx} - likely shape mismatch: {e}")
                 # Print shapes for debugging
                 print("Shapes:", [f.shape for f in current_features if f is not None])
                 continue


        # Append results for the block if any trials were successful
        if block_combined_features:
            all_combined_features.append(np.array(block_combined_features)) # Stack features for the block
            all_final_y_grasp.append(np.array(block_y_grasp))
            all_final_y_position.append(np.array(block_y_position))

    print(f"--- Phase 2 Finished: Multi-modal processing complete ---")

    if not all_combined_features: print("Error: No combined features generated."); return None

    # --- Phase 3: Final Data Prep, Training & Evaluation ---
    print("\n--- Phase 3: Final Data Prep, Training & Evaluation ---")
    try:
        # Concatenate features and labels from all blocks
        all_X_combined = np.concatenate(all_combined_features, axis=0)
        all_y_grasp = np.concatenate(all_final_y_grasp, axis=0)
        all_y_position = np.concatenate(all_final_y_position, axis=0)
    except ValueError as e: print(f"Error concatenating final data: {e}"); return None

    print(f"Total usable samples: {all_X_combined.shape[0]}")
    if all_X_combined.shape[0] == 0: print("Error: No samples available for training."); return None
    print(f"Combined Feature shape: {all_X_combined.shape[1:]}") # Check combined feature dimension
    print("Final Grasp Label Distribution:", Counter(all_y_grasp))
    print("Final Position Label Distribution:", Counter(all_y_position))

    # --- Single Data Split (Using the full combined features) ---
    print("\nSplitting data into Train/Validation/Test sets (Stratify by Grasp)...")
    try:
        (X_train, X_test, y_grasp_train, y_grasp_test, y_pos_train, y_pos_test) = train_test_split(
            all_X_combined, all_y_grasp, all_y_position, test_size=0.2, random_state=RANDOM_STATE, stratify=all_y_grasp)
        (X_train, X_val, y_grasp_train, y_grasp_val, y_pos_train, y_pos_val) = train_test_split(
            X_train, y_grasp_train, y_pos_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_grasp_train)
        print(f"Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    except ValueError as e: print(f"Error splitting data: {e}"); return None

    # --- Standardization (On the combined features) ---
    print("Standardizing combined features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Class Weights for Position Model ---
    print("Calculating class weights for Position model...")
    unique_pos_classes = np.unique(y_pos_train)
    position_class_weights_dict = None
    try:
        pos_weights = compute_class_weight('balanced', classes=unique_pos_classes, y=y_pos_train)
        position_class_weights_dict = dict(zip(unique_pos_classes, pos_weights))
        print("Position Class Weights:", {k: f"{v:.2f}" for k,v in position_class_weights_dict.items()})
    except Exception as e: print(f"Warning: Error computing class weights: {e}.")

    # --- Model Creation ---
    num_grasp_classes = len(np.unique(all_y_grasp))
    num_position_classes = len(np.unique(all_y_position))
    # IMPORTANT: Update input shape based on combined features
    input_feature_shape = (X_train_scaled.shape[1],)
    print(f"\nInput shape for models: {input_feature_shape}")

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
    grasp_history = grasp_model.fit(X_train_scaled, y_grasp_train, validation_data=(X_val_scaled, y_grasp_val), epochs=100, batch_size=64, callbacks=callbacks, verbose=1)
    print("\n--- Starting Position Model Training ---")
    position_history = position_model.fit(X_train_scaled, y_pos_train, validation_data=(X_val_scaled, y_pos_val), epochs=100, batch_size=64, callbacks=callbacks, verbose=1, class_weight=position_class_weights_dict)

    # --- Evaluation ---
    grasp_labels = [f"Grasp {i}" for i in range(num_grasp_classes)]
    position_labels = [f"Pos {i}" for i in range(num_position_classes)]
    evaluate_model(grasp_model, X_test_scaled, y_grasp_test, grasp_labels, "Grasp (Multi-Modal)")
    evaluate_model(position_model, X_test_scaled, y_pos_test, position_labels, "Position (Multi-Modal)")

    return grasp_model, grasp_history, position_model, position_history


# --- Execution ---
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    # GPU setup... (unchanged)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus),"Physical GPUs,", len(tf.config.experimental.list_logical_devices('GPU')), "Logical GPUs")
        except RuntimeError as e: print(e)

    results = asyncio.run(main_async())
    if results: print("\nAsync pipeline finished successfully.")
    else: print("\nAsync pipeline did not complete successfully.")