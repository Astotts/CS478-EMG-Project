import asyncio
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # Use layers alias
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import re
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm as sync_tqdm
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Ensure scikeras is installed: pip install scikeras
try:
    from scikeras.wrappers import KerasClassifier
except ImportError:
    print("Please install scikeras: pip install scikeras")
    exit()


# --- Global Configuration ---
BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data" # <-- SET YOUR PATH!
RANDOM_STATE = 42
WINDOW_DURATION_S = 0.150
OVERLAP_RATIO = 0.5
# --- Hyperparameters moved to grid search or defaults in create_model ---
# D_MODEL = 64
# NUM_HEADS = 4
# NUM_LAYERS = 2
# DFF = 128
# DROPOUT_RATE = 0.2

# --- Helper Functions ---
# _load_generic_hdf5_data, _parse_recording_parameters (Unchanged)
def _load_generic_hdf5_data(file_path):
    try:
        with h5py.File(file_path, 'r') as f: data = {int(k): np.array(v) for k,v in f.items()}; return data if data else None
    except FileNotFoundError: return None
    except Exception as e: print(f"Err load HDF5 {file_path}: {e}"); return None

def _parse_recording_parameters(file_path):
    params = {}; trial_len_key = 'trial_length_time'
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1); key = key.strip().lower().replace(' ', '_').replace('(sec)', '').strip('_'); value = value.strip()
                    try: params[key] = float(value)
                    except ValueError: params[key] = value
    except FileNotFoundError: return None
    except Exception as e: print(f"Err parse params {file_path}: {e}"); return None
    if trial_len_key not in params: return None
    return params

# load_multimodal_data_async (Unchanged)
async def load_multimodal_data_async(subject_path):
    loop = asyncio.get_running_loop(); emg_path=f"{subject_path}/emg_data.hdf5"; finger_path=f"{subject_path}/finger_data.hdf5"; glove_path=f"{subject_path}/glove_data.hdf5"; csv_path=f"{subject_path}/trials.csv"; params_path=f"{subject_path}/recording_parameters.txt"
    emg_task=loop.run_in_executor(None,_load_generic_hdf5_data,emg_path); finger_task=loop.run_in_executor(None,_load_generic_hdf5_data,finger_path); glove_task=loop.run_in_executor(None,_load_generic_hdf5_data,glove_path); params_task=loop.run_in_executor(None,_parse_recording_parameters,params_path)
    results = None; trials_df = None
    try: trials_df_task=loop.run_in_executor(None,pd.read_csv,csv_path); results=await asyncio.gather(emg_task,finger_task,glove_task,params_task,trials_df_task,return_exceptions=True)
    except Exception as e: print(f"Crit err gather {subject_path}: {e}"); return None
    if results is None: return None
    emg_data, finger_data, glove_data, params, trials_df = results
    if isinstance(emg_data, Exception): emg_data=None
    if isinstance(finger_data, Exception): finger_data=None
    if isinstance(glove_data, Exception): glove_data=None
    if isinstance(params, Exception): params=None
    if isinstance(trials_df, Exception): trials_df=None
    if emg_data is None or trials_df is None or params is None: return None
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty or 'grasp' not in trials_df.columns or 'target_position' not in trials_df.columns: return None
    if not isinstance(emg_data, dict) or not emg_data: return None

    X_emg_raw={}; X_finger_raw={}; X_glove_raw={}; y_grasp_dict={}; y_position_dict={}; valid_trial_indices=[]
    potential_indices=set(emg_data.keys())&set(range(len(trials_df)))
    if finger_data: potential_indices&=set(finger_data.keys())
    if glove_data: potential_indices&=set(glove_data.keys())

    for trial_idx in sorted(list(potential_indices)):
        try:
            emg_signal=emg_data[trial_idx]
            if not isinstance(emg_signal,np.ndarray)or emg_signal.ndim!=2 or emg_signal.shape[0]==0 or emg_signal.shape[1]==0: continue
            finger_signal=None; glove_signal=None
            if finger_data and trial_idx in finger_data: fsr=finger_data[trial_idx]; finger_signal=fsr.reshape(-1,fsr.shape[-1]) if isinstance(fsr,np.ndarray)and fsr.ndim>=1 and fsr.size>0 and fsr.ndim==1 else fsr if isinstance(fsr,np.ndarray)and fsr.ndim==2 and fsr.size>0 else None
            if glove_data and trial_idx in glove_data: gsr=glove_data[trial_idx]; glove_signal=gsr.reshape(-1,gsr.shape[-1]) if isinstance(gsr,np.ndarray)and gsr.ndim>=1 and gsr.size>0 and gsr.ndim==1 else gsr if isinstance(gsr,np.ndarray)and gsr.ndim==2 and gsr.size>0 else None
            trial_info=trials_df.iloc[trial_idx]; X_emg_raw[trial_idx]=emg_signal
            if finger_signal is not None and finger_signal.ndim==2: X_finger_raw[trial_idx]=finger_signal
            if glove_signal is not None and glove_signal.ndim==2: X_glove_raw[trial_idx]=glove_signal
            y_grasp_dict[trial_idx]=int(trial_info['grasp']-1); y_position_dict[trial_idx]=int(trial_info['target_position']-1); valid_trial_indices.append(trial_idx)
        except Exception as e: print(f"Warn proc trial {trial_idx} in {subject_path}: {e}"); continue
    if not valid_trial_indices: return None
    return (X_emg_raw,X_finger_raw,X_glove_raw),(y_grasp_dict,y_position_dict),params,sorted(valid_trial_indices)

# preprocess_emg (Unchanged)
def preprocess_emg(emg_signals_list, fs=2000):
    processed_signals = []; original_indices = []
    for i, signal in enumerate(emg_signals_list):
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0: continue
        rectified=np.abs(signal); cutoff_freq=20; filter_order=4; nyquist_freq=fs/2; normalized_cutoff=cutoff_freq/nyquist_freq; smoothed=rectified
        if 0<normalized_cutoff<1:
            try: b,a=butter(filter_order,normalized_cutoff,'low'); min_len=3*max(len(a),len(b)); smoothed = filtfilt(b,a,rectified,axis=1) if rectified.shape[1]>=min_len else rectified
            except ValueError: pass
        mean=smoothed.mean(axis=1,keepdims=True); std=smoothed.std(axis=1,keepdims=True); std_mask=std>1e-9; normalized=np.zeros_like(smoothed)
        np.divide(smoothed-mean,std+1e-9,out=normalized,where=std_mask); processed_signals.append(normalized); original_indices.append(i)
    return processed_signals, original_indices

# preprocess_generic (Unchanged)
def preprocess_generic(signal_list):
    processed_signals = []; original_indices = []
    for i, signal in enumerate(signal_list):
         if isinstance(signal, np.ndarray) and signal.ndim == 2 and signal.shape[0] > 0 and signal.shape[1] > 0:
             processed_signals.append(signal); original_indices.append(i)
    return processed_signals, original_indices

# extract_features_sequence (Unchanged)
def extract_features_sequence(processed_signals_list, window_samples, step_samples, feature_type='emg'):
    """
    Extracts features as a sequence of vectors (one per window). NO LONGER AVERAGES.
    feature_type: 'emg' or 'generic'
    Returns: List of feature_sequences (each is np.array [num_windows, num_features]),
             List of original indices processed.
    """
    all_feature_sequences = []
    processed_indices_map = []
    num_channels = 0
    if not processed_signals_list: return all_feature_sequences, processed_indices_map
    for sig in processed_signals_list:
        if sig.ndim == 2 and sig.shape[0] > 0: num_channels = sig.shape[0]; break
    if num_channels == 0: return all_feature_sequences, processed_indices_map

    for idx, signal in enumerate(processed_signals_list):
        if signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_samples: continue
        signal_length = signal.shape[1]; num_windows = (signal_length - window_samples) // step_samples + 1
        if num_windows <= 0: continue

        window_features_for_trial = [] # Collect features for all windows in this trial
        for i in range(num_windows):
            start = i * step_samples; end = start + window_samples; window = signal[:, start:end]
            if window.shape[1] != window_samples: continue
            try:
                # --- Feature calculation for the window ---
                if feature_type == 'emg':
                    mav=np.mean(np.abs(window),axis=1); var=np.var(window,axis=1); rms=np.sqrt(np.mean(window**2,axis=1))
                    diff=np.diff(window,axis=1); wl=np.sum(np.abs(diff),axis=1) if diff.size>0 else np.zeros(num_channels)
                    zc=np.sum(np.diff(np.sign(window+1e-9),axis=1)!=0,axis=1)/window_samples if window.shape[1]>1 else np.zeros(num_channels)
                    fft=np.fft.fft(window,axis=1); fft_abs=np.abs(fft[:,:window_samples//2])
                    if fft_abs.shape[1]==0: mf=np.zeros(num_channels); fs=np.zeros(num_channels)
                    else: mf=np.mean(fft_abs,axis=1); fs=np.std(fft_abs,axis=1)
                    window_feature=np.concatenate([mav,var,rms,wl,zc,mf,fs])
                elif feature_type == 'generic':
                    f_mean=np.mean(window,axis=1); f_std=np.std(window,axis=1); f_min=np.min(window,axis=1); f_max=np.max(window,axis=1)
                    f_range=f_max-f_min
                    window_feature=np.concatenate([f_mean,f_std,f_min,f_max,f_range])
                else: continue
                # --- End feature calculation ---

                if np.any(np.isnan(window_feature)) or np.any(np.isinf(window_feature)): continue
                window_features_for_trial.append(window_feature) # Append feature vector for this window
            except Exception: continue # Skip window on error

        # If valid windows were processed for this trial, append the whole sequence
        if window_features_for_trial:
            trial_feature_sequence = np.array(window_features_for_trial, dtype=np.float32) # Shape: (num_windows, num_features_per_window)
            # Basic check on the resulting sequence array
            if trial_feature_sequence.ndim == 2 and trial_feature_sequence.shape[0] > 0 and trial_feature_sequence.shape[1] > 0:
                 all_feature_sequences.append(trial_feature_sequence) # Append the 2D sequence array
                 processed_indices_map.append(idx) # Store index from input list
            # else: print(f"Warning: Invalid feature sequence shape {trial_feature_sequence.shape} for index {idx}")

    return all_feature_sequences, processed_indices_map # List of 2D arrays, list of indices


# --- Transformer Components (Unchanged) ---
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim; self.num_heads=num_heads; self.ff_dim=ff_dim; self.rate=rate

    def call(self, inputs, training=False, mask=None):
        attn_mask = mask[:, tf.newaxis, tf.newaxis, :] if mask is not None else None
        attn_output = self.att(inputs, inputs, attention_mask=attn_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config(); config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}); return config

# --- Model Creation Function (MODIFIED to accept hyperparameters) ---
# Takes hyperparameters as arguments, uses defaults if not provided by grid search
def create_transformer_model(input_shape_tuple, num_classes, num_layers=2, embed_dim=64, num_heads=4, ff_dim=128, dropout_rate=0.2, learning_rate=0.001):
    # Unpack potentially passed tuple
    if isinstance(input_shape_tuple, tuple) and len(input_shape_tuple) == 1:
         input_shape = input_shape_tuple[0]
    else:
         input_shape = input_shape_tuple # Assume it's already the correct shape tuple

    sequence_length, num_features = input_shape
    inputs = layers.Input(shape=input_shape)
    # Optional: Project features if embed_dim != num_features
    if embed_dim != num_features:
        x = layers.Dense(embed_dim, name="input_projection")(inputs)
    else:
        x = inputs
    # Add Positional Embedding
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim, name="position_embedding")(positions)
    x = x + pos_embeddings
    x = layers.Dropout(dropout_rate)(x)
    # Transformer Encoder Layers
    for _ in range(num_layers):
        # Use the passed hyperparameters
        x = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    # Pooling and Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Use the learning_rate hyperparameter
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Evaluation Function (Unchanged) ---
def evaluate_model(model, X_test, y_test, labels, task_name):
    print(f"\n--- Evaluating {task_name} Model ---")
    try:
        # For KerasClassifier, model might be the wrapper. Access the underlying Keras model if needed.
        if hasattr(model, 'model_'):
             keras_model = model.model_
        else:
             keras_model = model # Assume it's already a Keras model
        test_loss, test_acc = keras_model.evaluate(X_test, y_test, verbose=0)
        print(f"{task_name} Test Accuracy: {test_acc:.4f}\n{task_name} Test Loss: {test_loss:.4f}")
    except Exception as e: print(f"Error evaluate: {e}"); return
    try: y_pred_probs = model.predict_proba(X_test); y_pred_classes = np.argmax(y_pred_probs, axis=1) # Use predict_proba for wrapper
    except Exception as e: print(f"Error predict: {e}"); return
    if y_test.size == 0: print("Empty y_test"); return
    # Confusion Matrix (optional display)
    # try: cm = confusion_matrix(y_test, y_pred_classes, labels=list(range(len(labels)))); plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)*0.8))); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels); plt.xlabel('Predicted Label'); plt.ylabel('Actual Label'); plt.title(f'{task_name} Task - Confusion Matrix'); plt.tight_layout(); plt.show()
    # except Exception as e: print(f"Error CM: {e}")
    # Classification Report
    print(f"\n--- {task_name} Classification Report ---")
    try: print(classification_report(y_test, y_pred_classes, labels=list(range(len(labels))), target_names=labels, zero_division=0))
    except Exception as e: print(f"Error Report: {e}")


# --- Main Pipeline (MODIFIED FOR GRID SEARCH) ---
async def main_async():
    tasks_to_process = [(s, d, b) for s in range(1, 9) for d in [1, 2] for b in [1, 2]]
    # --- Phase 1: Loading ---
    print(f"--- Phase 1: Loading Multi-Modal Data ({len(tasks_to_process)} blocks) ---")
    loading_tasks = [load_multimodal_data_async(f"{BASE_DATA_PATH}/participant_{s}/participant{s}_day{d}_block{b}") for s, d, b in tasks_to_process]
    block_results_raw = await async_tqdm.gather(*loading_tasks, desc="Loading Blocks")
    block_results_raw = [res for res in block_results_raw if res is not None]
    if not block_results_raw: print("Error: No data loaded."); return None
    print(f"--- Phase 1 Finished: Loaded raw data from {len(block_results_raw)} blocks ---")

    # --- Phase 2a: Estimate Rates & Min Duration ---
    print("\n--- Phase 2a: Estimating Sampling Rates & Minimum Duration ---")
    trial_lengths_samples = {}; assumed_trial_length_s = None
    # ... (rate estimation and duration logic is unchanged) ...
    for i, block_data in enumerate(block_results_raw):
        (X_emg, X_finger, X_glove), _, params, valid_indices = block_data
        if not assumed_trial_length_s and params and 'trial_length_time' in params: assumed_trial_length_s = params['trial_length_time']; print(f"Assumed trial length: {assumed_trial_length_s}s")
        if not assumed_trial_length_s: continue
        for idx in valid_indices:
            current_global_idx = (i, idx)
            if idx in X_emg and X_emg[idx].ndim == 2: trial_lengths_samples.setdefault('emg', {})[current_global_idx] = X_emg[idx].shape[1]
            if X_finger and idx in X_finger and X_finger[idx].ndim == 2: trial_lengths_samples.setdefault('finger', {})[current_global_idx] = X_finger[idx].shape[1]
            if X_glove and idx in X_glove and X_glove[idx].ndim == 2: trial_lengths_samples.setdefault('glove', {})[current_global_idx] = X_glove[idx].shape[1]
    if not assumed_trial_length_s or not trial_lengths_samples: print("Error: Cannot determine trial length or no valid trials."); return None
    fs_estimates = {}; print("Estimated Sampling Rates (Hz):")
    for modality, lengths_dict in trial_lengths_samples.items():
         if lengths_dict: fs_estimates[modality] = np.mean(list(lengths_dict.values())) / assumed_trial_length_s; print(f"  {modality.capitalize()}: {fs_estimates[modality]:.2f}")
         else: print(f"  {modality.capitalize()}: Not found.")
    fs_emg = fs_estimates.get('emg', 2000.0); fs_finger = fs_estimates.get('finger'); fs_glove = fs_estimates.get('glove')
    if 'emg' not in fs_estimates: print("Warning: EMG sampling rate not estimated, using default 2000.0 Hz")
    min_duration_s = assumed_trial_length_s; print(f"Using minimum synchronized duration: {min_duration_s}s")
    target_samples = {}
    if fs_emg: target_samples['emg'] = int(min_duration_s * fs_emg)
    if fs_finger: target_samples['finger'] = int(min_duration_s * fs_finger)
    if fs_glove: target_samples['glove'] = int(min_duration_s * fs_glove)
    print("Target samples per modality:", target_samples)
    # --- End Phase 2a ---

    # --- Phase 2b: Sequential Processing & Feature Sequence Extraction ---
    print(f"\n--- Phase 2b: Processing Multi-Modal Data into Sequences ---")
    all_seq_features_emg = []; all_seq_features_finger = []; all_seq_features_glove = []
    all_y_grasp_processed = []; all_y_position_processed = []
    # Calculate window/step samples based on time duration and estimated rates
    window_samples = {}; step_samples = {}
    if fs_emg: window_samples['emg'] = int(WINDOW_DURATION_S * fs_emg); step_samples['emg'] = max(1, int(window_samples['emg'] * (1.0 - OVERLAP_RATIO)))
    if fs_finger: window_samples['finger'] = int(WINDOW_DURATION_S * fs_finger); step_samples['finger'] = max(1, int(window_samples['finger'] * (1.0 - OVERLAP_RATIO)))
    if fs_glove: window_samples['glove'] = int(WINDOW_DURATION_S * fs_glove); step_samples['glove'] = max(1, int(window_samples['glove'] * (1.0 - OVERLAP_RATIO)))
    print("Window samples per modality:", window_samples); print("Step samples per modality:", step_samples)

    for block_idx, block_data in enumerate(sync_tqdm(block_results_raw, desc="Processing Blocks")):
        (X_emg_raw, X_finger_raw, X_glove_raw), (y_grasp_dict, y_pos_dict), _, valid_indices = block_data
        # 1. Prepare truncated lists for each modality
        list_emg = []; list_finger = []; list_glove = []; list_grasp = []; list_pos = []; list_orig_indices_in_block = []
        for trial_idx in valid_indices:
            sig_emg=X_emg_raw.get(trial_idx); sig_finger=X_finger_raw.get(trial_idx) if X_finger_raw else None; sig_glove=X_glove_raw.get(trial_idx) if X_glove_raw else None
            valid_emg=sig_emg is not None and sig_emg.ndim == 2 and sig_emg.shape[1]>=target_samples.get('emg',0)
            valid_finger=fs_finger and sig_finger is not None and sig_finger.ndim == 2 and sig_finger.shape[1]>=target_samples.get('finger',0)
            valid_glove=fs_glove and sig_glove is not None and sig_glove.ndim == 2 and sig_glove.shape[1]>=target_samples.get('glove',0)
            if valid_emg:
                list_emg.append(sig_emg[:,:target_samples['emg']])
                list_finger.append(sig_finger[:,:target_samples['finger']] if valid_finger else None)
                list_glove.append(sig_glove[:,:target_samples['glove']] if valid_glove else None)
                list_grasp.append(y_grasp_dict[trial_idx]); list_pos.append(y_pos_dict[trial_idx]); list_orig_indices_in_block.append(trial_idx)
        if not list_emg: continue

        # 2. Preprocess
        emg_proc, idx_emg_p = preprocess_emg(list_emg, fs=fs_emg)
        finger_proc, idx_f_p = preprocess_generic([s for i, s in enumerate(list_finger) if i in idx_emg_p and s is not None])
        glove_proc, idx_g_p = preprocess_generic([s for i, s in enumerate(list_glove) if i in idx_emg_p and s is not None])
        map_idx_emg_to_f = {orig_idx: i for i, orig_idx in enumerate(idx_emg_p) if list_finger[orig_idx] is not None}
        map_idx_emg_to_g = {orig_idx: i for i, orig_idx in enumerate(idx_emg_p) if list_glove[orig_idx] is not None}

        # 3. Extract Feature Sequences (Calls MODIFIED function)
        seq_feat_emg, idx_emg_f = extract_features_sequence(emg_proc, window_samples.get('emg', 1), step_samples.get('emg', 1), 'emg')
        seq_feat_finger, idx_f_f_rel = extract_features_sequence(finger_proc, window_samples.get('finger', 1), step_samples.get('finger', 1), 'generic') if fs_finger and finger_proc else ([], [])
        seq_feat_glove, idx_g_f_rel = extract_features_sequence(glove_proc, window_samples.get('glove', 1), step_samples.get('glove', 1), 'generic') if fs_glove and glove_proc else ([], [])

        # Correctly map back relative indices to original indices within the block's valid EMG list
        processed_finger_indices = [k for k, v in map_idx_emg_to_f.items() if v in idx_f_p] # Indices from list_emg that had finger data and were preprocessed
        processed_glove_indices = [k for k, v in map_idx_emg_to_g.items() if v in idx_g_p] # Indices from list_emg that had glove data and were preprocessed

        idx_f_f = [processed_finger_indices[i] for i in idx_f_f_rel] # Map indices from finger_proc back to indices in list_emg
        idx_g_f = [processed_glove_indices[i] for i in idx_g_f_rel] # Map indices from glove_proc back to indices in list_emg

        # Map indices from emg_proc back to indices in list_emg
        original_indices_for_emg_features = [idx_emg_p[i] for i in idx_emg_f]

        seq_feat_emg_dict = {orig_idx: seq for orig_idx, seq in zip(original_indices_for_emg_features, seq_feat_emg)}
        seq_feat_finger_dict = {orig_idx: seq for orig_idx, seq in zip(idx_f_f, seq_feat_finger)}
        seq_feat_glove_dict = {orig_idx: seq for orig_idx, seq in zip(idx_g_f, seq_feat_glove)}

        # 4. Store sequences for trials where EMG features were extracted (using original index from list_emg)
        for idx_orig in original_indices_for_emg_features:
            all_seq_features_emg.append(seq_feat_emg_dict[idx_orig])
            all_seq_features_finger.append(seq_feat_finger_dict.get(idx_orig, None))
            all_seq_features_glove.append(seq_feat_glove_dict.get(idx_orig, None))
            all_y_grasp_processed.append(list_grasp[idx_orig])
            all_y_position_processed.append(list_pos[idx_orig])
    # --- End Phase 2b ---
    print(f"--- Phase 2 Finished: Sequence processing complete ---")
    if not all_seq_features_emg: print("Error: No EMG feature sequences extracted."); return None

    # --- Phase 3: Pad Sequences, Combine, Split ---
    print("\n--- Phase 3: Padding, Combining, Splitting ---")
    # Pad sequences (using max EMG length as target)
    max_len_emg = max(len(seq) for seq in all_seq_features_emg if seq is not None and len(seq) > 0) if all_seq_features_emg else 0
    if max_len_emg == 0: print("Error: Max EMG sequence length is 0."); return None
    print(f"Padding sequences to max length (num_windows): {max_len_emg}")
    X_emg_padded = pad_sequences(all_seq_features_emg, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0) # Use value=0.0 for padding

    # Pad optional modalities, creating zero arrays if data was missing entirely
    num_trials = len(all_seq_features_emg)
    num_finger_features = 0; X_finger_padded = None
    if fs_finger:
        temp_list = [s for s in all_seq_features_finger if s is not None and len(s)>0 and s.ndim==2]
        if temp_list:
             num_finger_features = temp_list[0].shape[1]
             # Pad only the valid sequences
             padded_valid_seqs = pad_sequences(temp_list, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0)
             # Create the full array initialized with zeros
             X_finger_padded = np.zeros((num_trials, max_len_emg, num_finger_features), dtype='float32')
             valid_indices = [i for i, s in enumerate(all_seq_features_finger) if s is not None and len(s)>0 and s.ndim==2]
             # Fill in the padded data at the correct original indices
             if len(valid_indices) == padded_valid_seqs.shape[0]:
                 for i_padded, orig_idx in enumerate(valid_indices):
                      len_to_copy = min(padded_valid_seqs[i_padded].shape[0], max_len_emg)
                      feat_len_to_copy = min(padded_valid_seqs[i_padded].shape[1], num_finger_features)
                      X_finger_padded[orig_idx, :len_to_copy, :feat_len_to_copy] = padded_valid_seqs[i_padded][:len_to_copy, :feat_len_to_copy]
             else: print(f"Warning: Mismatch in finger padding indices ({len(valid_indices)} vs {padded_valid_seqs.shape[0]})")

    num_glove_features = 0; X_glove_padded = None
    if fs_glove:
        temp_list = [s for s in all_seq_features_glove if s is not None and len(s)>0 and s.ndim==2]
        if temp_list:
            num_glove_features = temp_list[0].shape[1]
            padded_valid_seqs = pad_sequences(temp_list, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0)
            X_glove_padded = np.zeros((num_trials, max_len_emg, num_glove_features), dtype='float32')
            valid_indices = [i for i, s in enumerate(all_seq_features_glove) if s is not None and len(s)>0 and s.ndim==2]
            if len(valid_indices) == padded_valid_seqs.shape[0]:
                 for i_padded, orig_idx in enumerate(valid_indices):
                    len_to_copy = min(padded_valid_seqs[i_padded].shape[0], max_len_emg)
                    feat_len_to_copy = min(padded_valid_seqs[i_padded].shape[1], num_glove_features)
                    X_glove_padded[orig_idx, :len_to_copy, :feat_len_to_copy] = padded_valid_seqs[i_padded][:len_to_copy, :feat_len_to_copy]
            else: print(f"Warning: Mismatch in glove padding indices ({len(valid_indices)} vs {padded_valid_seqs.shape[0]})")


    # Combine features
    features_to_combine = [X_emg_padded]
    if X_finger_padded is not None: features_to_combine.append(X_finger_padded)
    if X_glove_padded is not None: features_to_combine.append(X_glove_padded)
    try: all_X_combined_padded = np.concatenate(features_to_combine, axis=-1)
    except ValueError as e: print(f"Error concatenating padded features: {e}"); return None
    all_y_grasp = np.array(all_y_grasp_processed); all_y_position = np.array(all_y_position_processed)

    print(f"Total usable sequences: {all_X_combined_padded.shape[0]}")
    if all_X_combined_padded.shape[0] == 0: print("Error: No sequences available."); return None
    print(f"Combined Padded Feature shape: {all_X_combined_padded.shape}") # Should be (num_trials, max_len_emg, total_features)
    print("Final Grasp Label Distribution:", Counter(all_y_grasp)); print("Final Position Label Distribution:", Counter(all_y_position))

    # --- Single Data Split (Train/Validation/Test) ---
    print("\nSplitting data (Stratify by Grasp)...")
    try:
        # Split into Train+Val and Test first
        (X_train_val, X_test, y_grasp_train_val, y_grasp_test, y_pos_train_val, y_pos_test) = train_test_split(
            all_X_combined_padded, all_y_grasp, all_y_position, test_size=0.2, random_state=RANDOM_STATE, stratify=all_y_grasp)

        # Split Train+Val into Train and Validation
        # We need the indices to create PredefinedSplit later
        n_train_val = len(X_train_val)
        indices_train_val = np.arange(n_train_val)
        (X_train, X_val, y_grasp_train, y_grasp_val, y_pos_train, y_pos_val, indices_train, indices_val) = train_test_split(
            X_train_val, y_grasp_train_val, y_pos_train_val, indices_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_grasp_train_val) # 0.25 * 0.8 = 0.2

        print(f"Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    except ValueError as e: print(f"Error splitting data: {e}"); return None

    # --- Standardization (Handle 3D) ---
    print("Standardizing combined features...")
    n_features_total = X_train.shape[2]; n_trials_train = X_train.shape[0]
    # Reshape, fit scaler ONLY on training data, transform all sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features_total)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features_total)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features_total)).reshape(X_test.shape)
    # Combine scaled train and val for GridSearchCV input
    X_train_val_scaled = np.concatenate((X_train_scaled, X_val_scaled), axis=0)
    y_grasp_train_val = np.concatenate((y_grasp_train, y_grasp_val), axis=0)
    y_pos_train_val = np.concatenate((y_pos_train, y_pos_val), axis=0)
    print(f"Scaled shapes: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
    print(f"Combined Train+Val shape for GridSearch: {X_train_val_scaled.shape}")

    # --- Create PredefinedSplit ---
    # Create an array where -1 indicates training sample and 0 indicates validation sample
    split_index = np.full(n_train_val, -1, dtype=int) # Initialize all to -1 (train)
    # Use the indices returned by train_test_split to mark validation samples
    # These indices_val are relative to the *original* X_train_val array
    split_index[indices_val] = 0 # Mark validation samples with 0
    predefined_split = PredefinedSplit(test_fold=split_index)
    print(f"Created PredefinedSplit for GridSearchCV with {np.sum(split_index == -1)} train and {np.sum(split_index == 0)} validation samples.")


    # --- Class Weights (Calculated on combined train+val for potential final fit) ---
    print("Calculating class weights for Position model...")
    unique_pos_classes = np.unique(y_pos_train_val); position_class_weights_dict = None
    try:
        pos_weights = compute_class_weight('balanced', classes=unique_pos_classes, y=y_pos_train_val)
        position_class_weights_dict = dict(zip(unique_pos_classes, pos_weights))
        print("Position Class Weights:", {k: f"{v:.2f}" for k,v in position_class_weights_dict.items()})
    except Exception as e: print(f"Warning: Error computing class weights: {e}.")

    # --- Hyperparameter Grid Definition ---
    # Define smaller grids first to test; expand later
    param_grid = {
        'model__num_layers': [2, 3],             # Tunable parameter for create_transformer_model
        'model__embed_dim': [64, 96],            # Tunable parameter for create_transformer_model
        'model__num_heads': [4, 8],             # Tunable parameter for create_transformer_model
        'model__ff_dim': [128, 256],           # Tunable parameter for create_transformer_model
        'model__dropout_rate': [0.15, 0.25],     # Tunable parameter for create_transformer_model
        'model__learning_rate': [1e-3, 5e-4],    # Tunable parameter for create_transformer_model
        'batch_size': [32, 64],                # Tunable parameter for model.fit via KerasClassifier
        # 'epochs': [50]  # Usually controlled by EarlyStopping, but can be set
    }
    # Common parameters for KerasClassifier not in the grid
    input_tf_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2]) # (max_len_emg, total_combined_features)
    num_grasp_classes = len(np.unique(all_y_grasp)); num_position_classes = len(np.unique(all_y_position))

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0), # Monitor training loss
             tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0)] # Monitor training loss

    # --- Grid Search for Grasp Model ---
    print(f"\n--- Starting Grid Search for Grasp Model ---")
    print(f"Input shape: {input_tf_shape}")
    print(f"Parameter Grid: {param_grid}")

    # Use KerasClassifier wrapper
    # Pass non-tuned Keras model parameters via KerasClassifier constructor
    # Pass non-tuned fit parameters (like callbacks, epochs) via KerasClassifier constructor
    grasp_clf = KerasClassifier(
        model=create_transformer_model,
        model__input_shape_tuple=input_tf_shape, # Pass shape here, prefixed with model__
        model__num_classes=num_grasp_classes,    # Pass fixed class number here
        loss="sparse_categorical_crossentropy", # Required by KerasClassifier
        optimizer="adam",                       # Optimizer type, LR is tuned in grid
        metrics=["accuracy"],                   # Metrics
        callbacks=callbacks,                    # Callbacks for fitting
        epochs=100,                             # Max epochs (EarlyStopping will likely stop sooner)
        verbose=0                               # Suppress Keras fit logs during grid search
        # batch_size will be tuned by GridSearchCV
    )

    grasp_grid_search = GridSearchCV(
        estimator=grasp_clf,
        param_grid=param_grid,
        scoring='accuracy',       # Score based on validation accuracy
        cv=predefined_split,      # Use our specific train/val split
        refit=True,               # Refit the best model on the whole train+val data
        verbose=2,                # Show progress
        n_jobs=1                  # Often better to run TF on 1 GPU, n_jobs=-1 might interfere
    )

    # Fit using the combined training and validation data
    grasp_grid_search.fit(X_train_val_scaled, y_grasp_train_val)

    print("\n--- Grasp Model Grid Search Results ---")
    print(f"Best Parameters: {grasp_grid_search.best_params_}")
    print(f"Best Cross-Validation Score (Accuracy): {grasp_grid_search.best_score_:.4f}")
    best_grasp_model = grasp_grid_search.best_estimator_ # This is the refit KerasClassifier

    # --- Grid Search for Position Model ---
    print(f"\n--- Starting Grid Search for Position Model ---")
    print(f"Input shape: {input_tf_shape}")
    print(f"Parameter Grid: {param_grid}")

    # Create classifier for position task
    position_clf = KerasClassifier(
        model=create_transformer_model,
        model__input_shape_tuple=input_tf_shape,
        model__num_classes=num_position_classes,
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        callbacks=callbacks,
        epochs=100,
        verbose=0
        # batch_size tuned by GridSearchCV
    )

    # Define fit parameters (like class_weight) to be passed during grid search fit
    position_fit_params = {'class_weight': position_class_weights_dict}

    position_grid_search = GridSearchCV(
        estimator=position_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=predefined_split,
        refit=True,
        verbose=2,
        n_jobs=1
    )

    # Pass fit_params to the fit method of GridSearchCV
    position_grid_search.fit(X_train_val_scaled, y_pos_train_val, **position_fit_params)

    print("\n--- Position Model Grid Search Results ---")
    print(f"Best Parameters: {position_grid_search.best_params_}")
    print(f"Best Cross-Validation Score (Accuracy): {position_grid_search.best_score_:.4f}")
    best_position_model = position_grid_search.best_estimator_ # Refit KerasClassifier

    # --- Final Evaluation on Test Set using Best Models ---
    print("\n--- Final Evaluation on Held-Out Test Set ---")
    grasp_labels = [f"Grasp {i}" for i in range(num_grasp_classes)]; position_labels = [f"Pos {i}" for i in range(num_position_classes)]

    # Evaluate the best models found by grid search
    evaluate_model(best_grasp_model, X_test_scaled, y_grasp_test, grasp_labels, "Grasp (Best Transformer)")
    evaluate_model(best_position_model, X_test_scaled, y_pos_test, position_labels, "Position (Best Transformer)")

    # You can access the underlying Keras model if needed:
    # final_keras_grasp_model = best_grasp_model.model_
    # final_keras_position_model = best_position_model.model_

    # Return the best *estimators* (KerasClassifier wrappers) found by GridSearchCV
    return best_grasp_model, best_position_model


# --- Execution ---
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    # GPU setup...
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus),"Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e: print(e)
    else:
        print("No GPU detected. Running on CPU.")

    # Run the async main function
    results = asyncio.run(main_async())

    if results:
        print("\nGrid search and evaluation pipeline finished successfully.")
        # results contains (best_grasp_model, best_position_model)
    else:
        print("\nPipeline did not complete successfully.")