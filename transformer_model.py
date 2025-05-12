import asyncio
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # Use layers alias
# **** Make sure train_test_split is imported ****
from sklearn.model_selection import train_test_split
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

# --- Global Configuration ---
BASE_DATA_PATH = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data" # <-- SET YOUR PATH!
RANDOM_STATE = 42
WINDOW_DURATION_S = 0.150
OVERLAP_RATIO = 0.5
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DFF = 128
DROPOUT_RATE = 0.2

# --- Helper Functions ---
# _load_generic_hdf5_data, _parse_recording_parameters, load_multimodal_data_async (Unchanged - assuming they work)
# ... (paste the unchanged helper functions here) ...
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
    if finger_data and isinstance(finger_data, dict): potential_indices&=set(finger_data.keys())
    if glove_data and isinstance(glove_data, dict): potential_indices&=set(glove_data.keys())

    for trial_idx in sorted(list(potential_indices)):
        try:
            emg_signal=emg_data[trial_idx]
            if not isinstance(emg_signal,np.ndarray)or emg_signal.ndim!=2 or emg_signal.shape[0]==0 or emg_signal.shape[1]==0: continue
            finger_signal=None; glove_signal=None
            if finger_data and isinstance(finger_data, dict) and trial_idx in finger_data:
                fsr=finger_data[trial_idx]; finger_signal=fsr.reshape(-1,fsr.shape[-1]) if isinstance(fsr,np.ndarray)and fsr.ndim>=1 and fsr.size>0 and fsr.ndim==1 else fsr if isinstance(fsr,np.ndarray)and fsr.ndim==2 and fsr.size>0 else None
            if glove_data and isinstance(glove_data, dict) and trial_idx in glove_data:
                 gsr=glove_data[trial_idx]; glove_signal=gsr.reshape(-1,gsr.shape[-1]) if isinstance(gsr,np.ndarray)and gsr.ndim>=1 and gsr.size>0 and gsr.ndim==1 else gsr if isinstance(gsr,np.ndarray)and gsr.ndim==2 and gsr.size>0 else None
            trial_info=trials_df.iloc[trial_idx]; X_emg_raw[trial_idx]=emg_signal
            if finger_signal is not None and finger_signal.ndim==2: X_finger_raw[trial_idx]=finger_signal
            if glove_signal is not None and glove_signal.ndim==2: X_glove_raw[trial_idx]=glove_signal
            y_grasp_dict[trial_idx]=int(trial_info['grasp']-1); y_position_dict[trial_idx]=int(trial_info['target_position']-1); valid_trial_indices.append(trial_idx)
        except Exception as e: print(f"Warn proc trial {trial_idx} in {subject_path}: {e}"); continue
    if not valid_trial_indices: return None
    return (X_emg_raw,X_finger_raw,X_glove_raw),(y_grasp_dict,y_position_dict),params,sorted(valid_trial_indices)


# preprocess_emg, preprocess_generic, extract_features_sequence (Unchanged)
# ... (paste the unchanged preprocessing/feature extraction functions here) ...
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

def preprocess_generic(signal_list):
    processed_signals = []; original_indices = []
    for i, signal in enumerate(signal_list):
         if isinstance(signal, np.ndarray) and signal.ndim == 2 and signal.shape[0] > 0 and signal.shape[1] > 0:
             processed_signals.append(signal); original_indices.append(i)
    return processed_signals, original_indices

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
    valid_signals = [s for s in processed_signals_list if isinstance(s, np.ndarray)]
    if not valid_signals: return all_feature_sequences, processed_indices_map

    for sig in valid_signals:
        if sig.ndim == 2 and sig.shape[0] > 0: num_channels = sig.shape[0]; break
    if num_channels == 0: return all_feature_sequences, processed_indices_map

    for idx, signal in enumerate(processed_signals_list):
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_samples: continue
        signal_length = signal.shape[1]; num_windows = (signal_length - window_samples) // step_samples + 1
        if num_windows <= 0: continue

        window_features_for_trial = []
        for i in range(num_windows):
            start = i * step_samples; end = start + window_samples; window = signal[:, start:end]
            if window.shape[1] != window_samples: continue
            try:
                if feature_type == 'emg':
                    mav=np.mean(np.abs(window),axis=1); var=np.var(window,axis=1); rms=np.sqrt(np.mean(window**2,axis=1))
                    diff=np.diff(window,axis=1); wl=np.sum(np.abs(diff),axis=1) if diff.size>0 else np.zeros(num_channels)
                    if window.shape[1] > 1: zc=np.sum(np.diff(np.sign(window+1e-9),axis=1)!=0,axis=1)/window_samples
                    else: zc = np.zeros(num_channels)
                    fft=np.fft.fft(window,axis=1); fft_abs=np.abs(fft[:,:window_samples//2])
                    if fft_abs.shape[1]==0: mf=np.zeros(num_channels); fs=np.zeros(num_channels)
                    else: mf=np.mean(fft_abs,axis=1); fs=np.std(fft_abs,axis=1)
                    window_feature=np.concatenate([mav,var,rms,wl,zc,mf,fs])
                elif feature_type == 'generic':
                    f_mean=np.mean(window,axis=1); f_std=np.std(window,axis=1); f_min=np.min(window,axis=1); f_max=np.max(window,axis=1)
                    f_range=f_max-f_min
                    window_feature=np.concatenate([f_mean,f_std,f_min,f_max,f_range])
                else: continue
                if np.any(np.isnan(window_feature)) or np.any(np.isinf(window_feature)): continue
                window_features_for_trial.append(window_feature)
            except Exception: continue

        if window_features_for_trial:
            trial_feature_sequence = np.array(window_features_for_trial, dtype=np.float32)
            if trial_feature_sequence.ndim == 2 and trial_feature_sequence.shape[0] > 0 and trial_feature_sequence.shape[1] > 0:
                 all_feature_sequences.append(trial_feature_sequence)
                 processed_indices_map.append(idx)

    return all_feature_sequences, processed_indices_map


# --- Transformer Components (Unchanged) ---
# ... (paste the TransformerEncoderLayer class here) ...
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
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=attn_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config(); config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}); return config


# --- Model Creation (Unchanged) ---
# ... (paste the create_transformer_model function here) ...
def create_transformer_model(input_shape, num_classes, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    if isinstance(input_shape, list) and len(input_shape) == 2: sequence_length, num_features = input_shape
    elif isinstance(input_shape, tuple) and len(input_shape) == 2: sequence_length, num_features = input_shape
    else: raise ValueError(f"Expected input_shape to be a tuple or list of length 2, but got {input_shape}")
    if sequence_length is None or num_features is None: raise ValueError(f"Sequence length and num_features cannot be None. Got shape: ({sequence_length}, {num_features})")

    inputs = layers.Input(shape=(sequence_length, num_features))
    if embed_dim != num_features: x = layers.Dense(embed_dim, name="input_projection")(inputs)
    else: x = inputs
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim, name="position_embedding")(positions)
    x = x + pos_embeddings
    x = layers.Dropout(dropout_rate)(x)
    for _ in range(num_layers): x = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Evaluation Function (Unchanged) ---
# ... (paste the evaluate_model function here) ...
def evaluate_model(model, X_test, y_test, labels, task_name):
    print(f"\n--- Evaluating {task_name} Model ---")
    try:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{task_name} Test Accuracy: {test_acc:.4f}\n{task_name} Test Loss: {test_loss:.4f}")
    except Exception as e: print(f"Error during model evaluation: {e}"); return
    try:
        if hasattr(model, 'predict_proba'): y_pred_probs = model.predict_proba(X_test)
        else: y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
    except Exception as e: print(f"Error during prediction: {e}"); return
    if y_test.size == 0: print("Evaluation skipped: y_test is empty."); return

    label_indices = list(range(len(labels)))
    unique_test_labels = np.unique(y_test)
    unique_pred_labels = np.unique(y_pred_classes)
    max_label = max(unique_test_labels.max() if len(unique_test_labels)>0 else -1, unique_pred_labels.max() if len(unique_pred_labels)>0 else -1)

    try:
        cm = confusion_matrix(y_test, y_pred_classes, labels=label_indices)
        plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)*0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.title(f'{task_name} Task - Confusion Matrix')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying Confusion Matrix: {e}")

    if max_label >= len(labels):
         print(f"Warning: Max label index ({max_label}) exceeds provided labels length ({len(labels)}). Extending labels for report.")
         labels.extend([f"Class {i}" for i in range(len(labels), max_label + 1)])
         label_indices = list(range(max_label + 1))
    print(f"\n--- {task_name} Classification Report ---")
    try: print(classification_report(y_test, y_pred_classes, labels=label_indices, target_names=labels, zero_division=0))
    except Exception as e: print(f"Error generating Classification Report: {e}")


# --- Main Pipeline ---
async def main_async():
    # --- Phase 1 & 2a (Loading, Rate Estimation) --- (Keep unchanged, robust version)
    # ... (paste the robust Phase 1 and 2a code from the previous response) ...
    tasks_to_process = [(s, d, b) for s in range(1, 9) for d in [1, 2] for b in [1, 2]]
    print(f"--- Phase 1: Loading Multi-Modal Data ({len(tasks_to_process)} blocks) ---")
    loading_tasks = [load_multimodal_data_async(f"{BASE_DATA_PATH}/participant_{s}/participant{s}_day{d}_block{b}") for s, d, b in tasks_to_process]
    block_results_raw = await async_tqdm.gather(*loading_tasks, desc="Loading Blocks")
    block_results_raw = [res for res in block_results_raw if res is not None]
    if not block_results_raw: print("Error: No data loaded."); return None
    print(f"--- Phase 1 Finished: Loaded raw data from {len(block_results_raw)} blocks ---")

    print("\n--- Phase 2a: Estimating Sampling Rates & Minimum Duration ---")
    trial_lengths_samples = {}; assumed_trial_length_s = None
    fs_emg = 2000.0; fs_finger = None; fs_glove = None
    has_finger = False; has_glove = False

    for i, block_data in enumerate(block_results_raw):
        (X_emg, X_finger, X_glove), _, params, valid_indices = block_data
        if X_finger and len(X_finger) > 0: has_finger = True
        if X_glove and len(X_glove) > 0: has_glove = True
        if not assumed_trial_length_s and params and 'trial_length_time' in params: assumed_trial_length_s = params['trial_length_time']; print(f"Assumed trial length: {assumed_trial_length_s}s")
        if not assumed_trial_length_s: continue
        for idx in valid_indices:
            current_global_idx = (i, idx)
            if idx in X_emg and isinstance(X_emg[idx], np.ndarray) and X_emg[idx].ndim == 2: trial_lengths_samples.setdefault('emg', {})[current_global_idx] = X_emg[idx].shape[1]
            if X_finger and idx in X_finger and isinstance(X_finger[idx], np.ndarray) and X_finger[idx].ndim == 2: trial_lengths_samples.setdefault('finger', {})[current_global_idx] = X_finger[idx].shape[1]
            if X_glove and idx in X_glove and isinstance(X_glove[idx], np.ndarray) and X_glove[idx].ndim == 2: trial_lengths_samples.setdefault('glove', {})[current_global_idx] = X_glove[idx].shape[1]

    if not assumed_trial_length_s or not trial_lengths_samples.get('emg'):
         print(f"Error: Cannot determine trial length ({assumed_trial_length_s}) or no valid EMG trials found.")
         return None

    fs_estimates = {}; print("Estimated Sampling Rates (Hz):")
    emg_lengths = list(trial_lengths_samples.get('emg', {}).values())
    if emg_lengths: fs_estimates['emg'] = np.mean(emg_lengths) / assumed_trial_length_s; fs_emg = fs_estimates['emg']; print(f"  {'emg'.capitalize()}: {fs_emg:.2f}")
    else: print(f"  {'emg'.capitalize()}: Using default {fs_emg:.2f} Hz (estimation failed).")
    if has_finger:
        finger_lengths = list(trial_lengths_samples.get('finger', {}).values())
        if finger_lengths: fs_estimates['finger'] = np.mean(finger_lengths) / assumed_trial_length_s; fs_finger = fs_estimates['finger']; print(f"  {'finger'.capitalize()}: {fs_finger:.2f}")
        else: print(f"  {'finger'.capitalize()}: Found data but could not estimate rate.")
    else: print(f"  {'finger'.capitalize()}: No data found.")
    if has_glove:
        glove_lengths = list(trial_lengths_samples.get('glove', {}).values())
        if glove_lengths: fs_estimates['glove'] = np.mean(glove_lengths) / assumed_trial_length_s; fs_glove = fs_estimates['glove']; print(f"  {'glove'.capitalize()}: {fs_glove:.2f}")
        else: print(f"  {'glove'.capitalize()}: Found data but could not estimate rate.")
    else: print(f"  {'glove'.capitalize()}: No data found.")

    min_duration_s = assumed_trial_length_s; print(f"Using minimum synchronized duration: {min_duration_s}s")
    target_samples = {}
    target_samples['emg'] = int(min_duration_s * fs_emg)
    if has_finger and fs_finger: target_samples['finger'] = int(min_duration_s * fs_finger)
    if has_glove and fs_glove: target_samples['glove'] = int(min_duration_s * fs_glove)
    print("Target samples per modality:", target_samples)
    if not target_samples.get('emg'): print("Error: Failed to calculate target samples for EMG."); return None

    # --- Phase 2b (Preprocessing, Feature Extraction) --- (Keep unchanged, robust version)
    # ... (paste the robust Phase 2b code from the previous response) ...
    print(f"\n--- Phase 2b: Processing Multi-Modal Data into Sequences ---")
    all_seq_features_emg = []; all_seq_features_finger = []; all_seq_features_glove = []
    all_y_grasp_processed = []; all_y_position_processed = []
    window_samples = {}; step_samples = {}
    window_samples['emg'] = int(WINDOW_DURATION_S * fs_emg); step_samples['emg'] = max(1, int(window_samples['emg'] * (1.0 - OVERLAP_RATIO)))
    if has_finger and fs_finger: window_samples['finger'] = int(WINDOW_DURATION_S * fs_finger); step_samples['finger'] = max(1, int(window_samples['finger'] * (1.0 - OVERLAP_RATIO)))
    if has_glove and fs_glove: window_samples['glove'] = int(WINDOW_DURATION_S * fs_glove); step_samples['glove'] = max(1, int(window_samples['glove'] * (1.0 - OVERLAP_RATIO)))
    print("Window samples per modality:", window_samples); print("Step samples per modality:", step_samples)
    if not window_samples.get('emg') or not step_samples.get('emg'): print("Error: Failed to calculate window/step samples for EMG."); return None

    for block_idx, block_data in enumerate(sync_tqdm(block_results_raw, desc="Processing Blocks")):
        (X_emg_raw, X_finger_raw, X_glove_raw), (y_grasp_dict, y_pos_dict), _, valid_indices = block_data
        list_emg = []; list_finger = []; list_glove = []; list_grasp = []; list_pos = []; list_orig_indices_in_block = []
        target_emg_len = target_samples.get('emg', 0)
        target_finger_len = target_samples.get('finger', 0) if has_finger and fs_finger else 0
        target_glove_len = target_samples.get('glove', 0) if has_glove and fs_glove else 0

        for trial_idx in valid_indices:
            sig_emg=X_emg_raw.get(trial_idx)
            sig_finger=X_finger_raw.get(trial_idx) if X_finger_raw and has_finger else None
            sig_glove=X_glove_raw.get(trial_idx) if X_glove_raw and has_glove else None
            valid_emg = (sig_emg is not None and isinstance(sig_emg, np.ndarray) and sig_emg.ndim == 2 and sig_emg.shape[1] >= target_emg_len)
            if not valid_emg: continue
            valid_finger = (target_finger_len > 0 and sig_finger is not None and isinstance(sig_finger, np.ndarray) and sig_finger.ndim == 2 and sig_finger.shape[1] >= target_finger_len)
            valid_glove = (target_glove_len > 0 and sig_glove is not None and isinstance(sig_glove, np.ndarray) and sig_glove.ndim == 2 and sig_glove.shape[1] >= target_glove_len)
            list_emg.append(sig_emg[:, :target_emg_len])
            list_finger.append(sig_finger[:, :target_finger_len] if valid_finger else None)
            list_glove.append(sig_glove[:, :target_glove_len] if valid_glove else None)
            list_grasp.append(y_grasp_dict[trial_idx]); list_pos.append(y_pos_dict[trial_idx]); list_orig_indices_in_block.append(trial_idx)
        if not list_emg: continue

        emg_proc, idx_emg_p = preprocess_emg(list_emg, fs=fs_emg)
        if not emg_proc: continue
        finger_proc, idx_f_p = [], []; glove_proc, idx_g_p = [], []
        if has_finger and fs_finger:
            finger_to_process = [s for i, s in enumerate(list_finger) if i in idx_emg_p and s is not None]
            if finger_to_process:
                 finger_proc, idx_f_p_rel = preprocess_generic(finger_to_process)
                 map_emg_idx_to_valid_finger_idx = {emg_idx: i for i, emg_idx in enumerate(idx_emg_p) if list_finger[emg_idx] is not None}
                 idx_f_p = [k for k, v in map_emg_idx_to_valid_finger_idx.items() if v in idx_f_p_rel]
        if has_glove and fs_glove:
            glove_to_process = [s for i, s in enumerate(list_glove) if i in idx_emg_p and s is not None]
            if glove_to_process:
                glove_proc, idx_g_p_rel = preprocess_generic(glove_to_process)
                map_emg_idx_to_valid_glove_idx = {emg_idx: i for i, emg_idx in enumerate(idx_emg_p) if list_glove[emg_idx] is not None}
                idx_g_p = [k for k, v in map_emg_idx_to_valid_glove_idx.items() if v in idx_g_p_rel]

        win_emg = window_samples.get('emg'); step_emg = step_samples.get('emg')
        if win_emg is None or step_emg is None: continue
        seq_feat_emg, idx_emg_f_rel = extract_features_sequence(emg_proc, win_emg, step_emg, 'emg')
        if not seq_feat_emg: continue
        idx_emg_f = [idx_emg_p[i] for i in idx_emg_f_rel]
        seq_feat_emg_dict = {orig_idx: seq for orig_idx, seq in zip(idx_emg_f, seq_feat_emg)}

        seq_feat_finger_dict = {}
        if has_finger and fs_finger and finger_proc:
            win_finger = window_samples.get('finger'); step_finger = step_samples.get('finger')
            if win_finger and step_finger:
                 seq_feat_finger, idx_f_f_rel_proc = extract_features_sequence(finger_proc, win_finger, step_finger, 'generic')
                 processed_finger_indices = idx_f_p
                 if len(processed_finger_indices) == len(finger_proc):
                      idx_f_f = [processed_finger_indices[i] for i in idx_f_f_rel_proc]
                      seq_feat_finger_dict = {orig_idx: seq for orig_idx, seq in zip(idx_f_f, seq_feat_finger)}
        seq_feat_glove_dict = {}
        if has_glove and fs_glove and glove_proc:
             win_glove = window_samples.get('glove'); step_glove = step_samples.get('glove')
             if win_glove and step_glove:
                 seq_feat_glove, idx_g_f_rel_proc = extract_features_sequence(glove_proc, win_glove, step_glove, 'generic')
                 processed_glove_indices = idx_g_p
                 if len(processed_glove_indices) == len(glove_proc):
                     idx_g_f = [processed_glove_indices[i] for i in idx_g_f_rel_proc]
                     seq_feat_glove_dict = {orig_idx: seq for orig_idx, seq in zip(idx_g_f, seq_feat_glove)}

        for idx_orig in idx_emg_f:
            all_seq_features_emg.append(seq_feat_emg_dict[idx_orig])
            all_seq_features_finger.append(seq_feat_finger_dict.get(idx_orig, None))
            all_seq_features_glove.append(seq_feat_glove_dict.get(idx_orig, None))
            all_y_grasp_processed.append(list_grasp[idx_orig])
            all_y_position_processed.append(list_pos[idx_orig])
    print(f"--- Phase 2 Finished: Sequence processing complete ---")
    if not all_seq_features_emg: print("Error: No EMG feature sequences extracted."); return None


    # --- Phase 3: Pad Sequences, Combine, Split, Train, Evaluate ---
    print("\n--- Phase 3: Padding, Combining, Splitting, Training & Evaluation ---")
    # --- Padding and Combining --- (Keep unchanged, robust version)
    # ... (paste the robust Padding/Combining code from the previous response) ...
    valid_emg_sequences = [seq for seq in all_seq_features_emg if seq is not None and len(seq) > 0]
    if not valid_emg_sequences: print("Error: No valid non-empty EMG sequences found after processing."); return None
    max_len_emg = max(len(seq) for seq in valid_emg_sequences)
    print(f"Padding sequences to max length (num_windows): {max_len_emg}")
    X_emg_padded = pad_sequences(valid_emg_sequences, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0)

    num_trials = len(all_y_grasp_processed)
    X_finger_padded = None; num_finger_features = 0
    if has_finger and fs_finger:
        valid_finger_data = [(i, s) for i, s in enumerate(all_seq_features_finger) if s is not None and len(s)>0 and s.ndim==2]
        if valid_finger_data:
            valid_indices_finger = [item[0] for item in valid_finger_data]; valid_seqs_finger = [item[1] for item in valid_finger_data]
            num_finger_features = valid_seqs_finger[0].shape[1]
            padded_valid_seqs = pad_sequences(valid_seqs_finger, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0)
            X_finger_padded = np.zeros((num_trials, max_len_emg, num_finger_features), dtype='float32')
            if len(valid_indices_finger) == padded_valid_seqs.shape[0]:
                 for i_padded, orig_idx in enumerate(valid_indices_finger):
                      if orig_idx < num_trials:
                          len_to_copy = min(padded_valid_seqs[i_padded].shape[0], max_len_emg); feat_len_to_copy = min(padded_valid_seqs[i_padded].shape[1], num_finger_features)
                          X_finger_padded[orig_idx, :len_to_copy, :feat_len_to_copy] = padded_valid_seqs[i_padded][:len_to_copy, :feat_len_to_copy]
                      else: print(f"Warning: Original index {orig_idx} out of bounds for finger padding.")
            else: print(f"Warning: Mismatch in finger padding indices ({len(valid_indices_finger)} vs {padded_valid_seqs.shape[0]})")
    X_glove_padded = None; num_glove_features = 0
    if has_glove and fs_glove:
        valid_glove_data = [(i, s) for i, s in enumerate(all_seq_features_glove) if s is not None and len(s)>0 and s.ndim==2]
        if valid_glove_data:
            valid_indices_glove = [item[0] for item in valid_glove_data]; valid_seqs_glove = [item[1] for item in valid_glove_data]
            num_glove_features = valid_seqs_glove[0].shape[1]
            padded_valid_seqs = pad_sequences(valid_seqs_glove, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0)
            X_glove_padded = np.zeros((num_trials, max_len_emg, num_glove_features), dtype='float32')
            if len(valid_indices_glove) == padded_valid_seqs.shape[0]:
                for i_padded, orig_idx in enumerate(valid_indices_glove):
                     if orig_idx < num_trials:
                         len_to_copy = min(padded_valid_seqs[i_padded].shape[0], max_len_emg); feat_len_to_copy = min(padded_valid_seqs[i_padded].shape[1], num_glove_features)
                         X_glove_padded[orig_idx, :len_to_copy, :feat_len_to_copy] = padded_valid_seqs[i_padded][:len_to_copy, :feat_len_to_copy]
                     else: print(f"Warning: Original index {orig_idx} out of bounds for glove padding.")
            else: print(f"Warning: Mismatch in glove padding indices ({len(valid_indices_glove)} vs {padded_valid_seqs.shape[0]})")

    if X_emg_padded.shape[0] != num_trials:
         print(f"Error: Mismatch between number of padded EMG sequences ({X_emg_padded.shape[0]}) and number of labels ({num_trials}). Reconciling.")
         min_count = min(X_emg_padded.shape[0], num_trials)
         X_emg_padded = X_emg_padded[:min_count]
         all_y_grasp = np.array(all_y_grasp_processed[:min_count]); all_y_position = np.array(all_y_position_processed[:min_count])
         if X_finger_padded is not None: X_finger_padded = X_finger_padded[:min_count]
         if X_glove_padded is not None: X_glove_padded = X_glove_padded[:min_count]
         num_trials = min_count
    else: all_y_grasp = np.array(all_y_grasp_processed); all_y_position = np.array(all_y_position_processed)

    features_to_combine = [X_emg_padded]
    if X_finger_padded is not None: features_to_combine.append(X_finger_padded)
    if X_glove_padded is not None: features_to_combine.append(X_glove_padded)
    try: all_X_combined_padded = np.concatenate(features_to_combine, axis=-1)
    except ValueError as e:
        print(f"Error concatenating padded features: {e}"); [print(f" F{i}: {f.shape if f is not None else 'None'}") for i,f in enumerate(features_to_combine)]; return None

    print(f"Total usable sequences after padding/combining: {all_X_combined_padded.shape[0]}")
    if all_X_combined_padded.shape[0] == 0: print("Error: No sequences available."); return None
    print(f"Combined Padded Feature shape: {all_X_combined_padded.shape}")
    print("Final Grasp Label Distribution Before Split:", Counter(all_y_grasp)); print("Final Position Label Distribution Before Split:", Counter(all_y_position))


    # --- Custom Data Split (Exactly 160 per class for Test) ---
    print("\nSplitting data using custom method for fixed test set size...")
    target_test_count_per_class = 160
    test_indices = []
    train_val_indices = []
    unique_grasp_classes, grasp_counts = np.unique(all_y_grasp, return_counts=True)

    print(f"Found grasp classes: {unique_grasp_classes} with counts: {grasp_counts}")

    possible_to_split = True
    for grasp_class, total_count in zip(unique_grasp_classes, grasp_counts):
        if total_count < target_test_count_per_class:
            print(f"Error: Cannot select {target_test_count_per_class} test samples for grasp class {grasp_class}. Only {total_count} available.")
            possible_to_split = False
            # Decide how to handle: exit, or maybe take all available for test? Let's exit for now.
            return None # Exit if goal is unachievable

        # Find indices for the current class
        class_indices = np.where(all_y_grasp == grasp_class)[0]

        # Randomly choose test indices for this class without replacement
        selected_test_indices = np.random.choice(class_indices, size=target_test_count_per_class, replace=False)
        test_indices.extend(selected_test_indices)

        # Get remaining indices for the train/val pool for this class
        remaining_indices = np.setdiff1d(class_indices, selected_test_indices)
        train_val_indices.extend(remaining_indices)

    if not possible_to_split: return None # Should have exited above, but double check

    # Convert lists to numpy arrays and shuffle them to mix classes
    test_indices = np.array(test_indices)
    train_val_indices = np.array(train_val_indices)
    np.random.shuffle(test_indices) # Shuffle test set indices
    np.random.shuffle(train_val_indices) # Shuffle train/val indices

    # Create the test set
    X_test = all_X_combined_padded[test_indices]
    y_grasp_test = all_y_grasp[test_indices]
    y_pos_test = all_y_position[test_indices]

    # Create the training/validation pool
    X_train_val = all_X_combined_padded[train_val_indices]
    y_grasp_train_val = all_y_grasp[train_val_indices]
    y_pos_train_val = all_y_position[train_val_indices]

    print(f"\nCustom Split Result:")
    print(f"Test set size: {len(X_test)}")
    print(f"Test set grasp distribution: {Counter(y_grasp_test)}")
    print(f"Train/Validation pool size: {len(X_train_val)}")
    print(f"Train/Validation pool grasp distribution: {Counter(y_grasp_train_val)}")

    # Now, split the train_val pool into actual training and validation sets
    # Use test_size=0.25 to get approx 20% of original data for validation (since train_val is ~80%)
    try:
        (X_train, X_val,
         y_grasp_train, y_grasp_val,
         y_pos_train, y_pos_val) = train_test_split(
            X_train_val, y_grasp_train_val, y_pos_train_val,
            test_size=0.25, # 25% of the remaining data for validation
            random_state=RANDOM_STATE,
            stratify=y_grasp_train_val # Stratify this split too
        )
        print(f"\nFinal Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Train grasp distribution: {Counter(y_grasp_train)}")
        print(f"Validation grasp distribution: {Counter(y_grasp_val)}")

    except ValueError as e:
         print(f"Error splitting train/validation pool: {e}")
         print(f"Train/Validation pool shape: {X_train_val.shape}")
         print(f"Train/Validation grasp labels: {Counter(y_grasp_train_val)}")
         return None
    # --- End Custom Data Split ---


    # --- Standardization (Handle 3D) --- (Keep unchanged, robust version)
    # ... (paste the robust Standardization code from the previous response) ...
    print("Standardizing combined features...")
    n_features_total = X_train.shape[2];
    if X_train.shape[0] > 0: X_train_reshaped = X_train.reshape(-1, n_features_total)
    else: X_train_reshaped = np.empty((0, n_features_total))
    scaler = StandardScaler();
    if X_train_reshaped.shape[0] > 0:
        scaler.fit(X_train_reshaped)
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    else: X_train_scaled = X_train
    if X_val.shape[0] > 0: X_val_scaled = scaler.transform(X_val.reshape(-1, n_features_total)).reshape(X_val.shape)
    else: X_val_scaled = X_val
    if X_test.shape[0] > 0: X_test_scaled = scaler.transform(X_test.reshape(-1, n_features_total)).reshape(X_test.shape)
    else: X_test_scaled = X_test
    print(f"Scaled shapes: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")


    # --- Class Weights --- (Keep unchanged, robust version)
    # ... (paste the robust Class Weights code from the previous response) ...
    print("Calculating class weights for Position model...")
    unique_pos_classes, counts_pos_train = np.unique(y_pos_train, return_counts=True)
    position_class_weights_dict = None
    if len(unique_pos_classes) > 1:
        try:
            pos_weights = compute_class_weight('balanced', classes=unique_pos_classes, y=y_pos_train)
            position_class_weights_dict = dict(zip(unique_pos_classes, pos_weights))
            print("Position Class Weights:", {k: f"{v:.2f}" for k,v in position_class_weights_dict.items()})
        except Exception as e: print(f"Warning: Error computing class weights: {e}.")
    elif len(unique_pos_classes) == 1: print(f"Warning: Only one position class ({unique_pos_classes[0]}) present in training data. Class weights not computed.")
    else: print("Warning: No position labels found in training data. Class weights not computed.")


    # --- Model Creation (Transformer) --- (Keep unchanged, robust version)
    # ... (paste the robust Model Creation code from the previous response) ...
    num_grasp_classes = len(np.unique(all_y_grasp)); num_position_classes = len(np.unique(all_y_position))
    if X_train_scaled.shape[1] is None or X_train_scaled.shape[2] is None:
         print(f"Error: Invalid input shape determined after scaling: {X_train_scaled.shape}")
         return None
    input_tf_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    print(f"\nInput shape for Transformer models: {input_tf_shape}")
    print(f"Num Grasp Classes: {num_grasp_classes}, Num Position Classes: {num_position_classes}")
    print("Creating Transformer models...")
    try:
        grasp_model = create_transformer_model(input_tf_shape, num_grasp_classes, NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT_RATE)
        position_model = create_transformer_model(input_tf_shape, num_position_classes, NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT_RATE)
        print("Models created successfully.")
    except Exception as e: print(f"Error creating Keras model: {e}"); return None


    # --- Training --- (Keep unchanged, robust version)
    # ... (paste the robust Training code from the previous response) ...
    if X_val_scaled.shape[0] == 0 or y_grasp_val.shape[0] == 0:
         print("Warning: Validation set is empty. Training without validation callbacks.")
         callbacks_grasp = []; validation_data_grasp = None
    else:
         callbacks_grasp = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)]
         validation_data_grasp = (X_val_scaled, y_grasp_val)
    if X_val_scaled.shape[0] == 0 or y_pos_val.shape[0] == 0:
         print("Warning: Validation set is empty. Training position model without validation callbacks.")
         callbacks_pos = []; validation_data_pos = None
    else:
          callbacks_pos = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)]
          validation_data_pos = (X_val_scaled, y_pos_val)
    if X_train_scaled.shape[0] == 0: print("Error: Training data is empty. Cannot train models."); return None

    print("\n--- Starting Grasp Model Training ---")
    grasp_history = grasp_model.fit(X_train_scaled, y_grasp_train, validation_data=validation_data_grasp, epochs=150, batch_size=32, callbacks=callbacks_grasp, verbose=1)
    print("\n--- Starting Position Model Training ---")
    position_history = position_model.fit(X_train_scaled, y_pos_train, validation_data=validation_data_pos, epochs=150, batch_size=32, callbacks=callbacks_pos, verbose=1, class_weight=position_class_weights_dict)


    # --- Evaluation --- (Keep unchanged, robust version)
    # ... (paste the robust Evaluation code from the previous response) ...
    if X_test_scaled.shape[0] == 0:
         print("\nEvaluation skipped: Test set is empty.")
         return grasp_model, grasp_history, position_model, position_history # Still return models if trained
    grasp_labels = [f"Grasp {i}" for i in range(num_grasp_classes)]; position_labels = [f"Pos {i}" for i in range(num_position_classes)]
    evaluate_model(grasp_model, X_test_scaled, y_grasp_test, grasp_labels, "Grasp (Transformer Multi-Modal)")
    evaluate_model(position_model, X_test_scaled, y_pos_test, position_labels, "Position (Transformer Multi-Modal)")

    return grasp_model, grasp_history, position_model, position_history


# --- Execution --- (Keep unchanged, robust version)
# ... (paste the robust Execution code from the previous response) ...
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus),"Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e: print(e)
    else: print("No GPU detected. Running on CPU.")
    results = asyncio.run(main_async())
    if results: print("\nAsync pipeline finished successfully.")
    else: print("\nAsync pipeline did not complete successfully.")