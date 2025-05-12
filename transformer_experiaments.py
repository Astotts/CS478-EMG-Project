import asyncio
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # Use layers alias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
# import re # No longer used
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
NUM_LAYERS = 2 # Number of TransformerEncoderLayers
DFF = 128 # Dimensionality of the feed-forward network in TransformerEncoderLayer
DROPOUT_RATE = 0.2

# --- Helper Functions ---
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
    if trial_len_key not in params: return None # Essential for rate estimation
    return params

async def load_multimodal_data_async(subject_path):
    loop = asyncio.get_running_loop()
    emg_path=f"{subject_path}/emg_data.hdf5"
    finger_path=f"{subject_path}/finger_data.hdf5"
    glove_path=f"{subject_path}/glove_data.hdf5"
    csv_path=f"{subject_path}/trials.csv"
    params_path=f"{subject_path}/recording_parameters.txt"

    emg_task=loop.run_in_executor(None,_load_generic_hdf5_data,emg_path)
    finger_task=loop.run_in_executor(None,_load_generic_hdf5_data,finger_path)
    glove_task=loop.run_in_executor(None,_load_generic_hdf5_data,glove_path)
    params_task=loop.run_in_executor(None,_parse_recording_parameters,params_path)
    results = None; trials_df = None
    try:
        trials_df_task=loop.run_in_executor(None,pd.read_csv,csv_path)
        results=await asyncio.gather(emg_task,finger_task,glove_task,params_task,trials_df_task,return_exceptions=True)
    except Exception as e:
        print(f"Critical error during asyncio.gather for {subject_path}: {e}"); return None

    if results is None: return None
    emg_data, finger_data, glove_data, params, trials_df = results

    # Handle potential exceptions from asyncio.gather
    if isinstance(emg_data, Exception): emg_data=None
    if isinstance(finger_data, Exception): finger_data=None
    if isinstance(glove_data, Exception): glove_data=None
    if isinstance(params, Exception): params=None
    if isinstance(trials_df, Exception): trials_df=None

    if emg_data is None or trials_df is None or params is None: return None
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty or 'grasp' not in trials_df.columns:
        # Removed 'target_position' check as it's no longer used
        print(f"Warning: trials.csv for {subject_path} is invalid or missing 'grasp' column.")
        return None
    if not isinstance(emg_data, dict) or not emg_data: return None

    X_emg_raw={}; X_finger_raw={}; X_glove_raw={}; y_grasp_dict={}; valid_trial_indices=[]
    # Determine common trial indices based on availability in EMG, CSV, and optionally finger/glove
    potential_indices = set(emg_data.keys()) & set(range(len(trials_df)))
    if finger_data and isinstance(finger_data, dict): potential_indices &= set(finger_data.keys())
    if glove_data and isinstance(glove_data, dict): potential_indices &= set(glove_data.keys())

    for trial_idx in sorted(list(potential_indices)):
        try:
            emg_signal=emg_data[trial_idx]
            # Basic validation of EMG signal structure
            if not isinstance(emg_signal,np.ndarray)or emg_signal.ndim!=2 or emg_signal.shape[0]==0 or emg_signal.shape[1]==0: continue

            finger_signal=None; glove_signal=None
            if finger_data and isinstance(finger_data, dict) and trial_idx in finger_data:
                fsr=finger_data[trial_idx]
                finger_signal=fsr.reshape(-1,fsr.shape[-1]) if isinstance(fsr,np.ndarray)and fsr.ndim>=1 and fsr.size>0 and fsr.ndim==1 else fsr if isinstance(fsr,np.ndarray)and fsr.ndim==2 and fsr.size>0 else None
            if glove_data and isinstance(glove_data, dict) and trial_idx in glove_data:
                gsr=glove_data[trial_idx]
                glove_signal=gsr.reshape(-1,gsr.shape[-1]) if isinstance(gsr,np.ndarray)and gsr.ndim>=1 and gsr.size>0 and gsr.ndim==1 else gsr if isinstance(gsr,np.ndarray)and gsr.ndim==2 and gsr.size>0 else None

            trial_info=trials_df.iloc[trial_idx]
            X_emg_raw[trial_idx]=emg_signal
            if finger_signal is not None and finger_signal.ndim==2: X_finger_raw[trial_idx]=finger_signal
            if glove_signal is not None and glove_signal.ndim==2: X_glove_raw[trial_idx]=glove_signal

            y_grasp_dict[trial_idx]=int(trial_info['grasp']-1) # Grasp labels are 1-indexed in CSV
            valid_trial_indices.append(trial_idx)
        except Exception as e:
            print(f"Warning: Error processing trial {trial_idx} in {subject_path}: {e}"); continue

    if not valid_trial_indices: return None
    # Return structure changed: y_position_dict removed
    return (X_emg_raw,X_finger_raw,X_glove_raw), y_grasp_dict, params, sorted(valid_trial_indices)


def preprocess_emg(emg_signals_list, fs=2000):
    processed_signals = []; original_indices = []
    for i, signal in enumerate(emg_signals_list):
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0: continue
        rectified=np.abs(signal); cutoff_freq=20; filter_order=4; nyquist_freq=fs/2; normalized_cutoff=cutoff_freq/nyquist_freq; smoothed=rectified
        if 0<normalized_cutoff<1: # Check for valid cutoff
            try:
                b,a=butter(filter_order,normalized_cutoff,'low', analog=False)
                min_len=3*max(len(a),len(b)) # Ensure signal is long enough for filtfilt
                smoothed = filtfilt(b,a,rectified,axis=1) if rectified.shape[1]>=min_len else rectified
            except ValueError as e: # Keep rectified if filter fails (e.g., signal too short after all)
                # print(f"Butterworth filter ValueError: {e}. Using rectified signal for sample {i}.")
                pass # Smoothed remains rectified
        mean=smoothed.mean(axis=1,keepdims=True); std=smoothed.std(axis=1,keepdims=True); std_mask=std>1e-9; normalized=np.zeros_like(smoothed)
        np.divide(smoothed-mean,std+1e-9,out=normalized,where=std_mask); processed_signals.append(normalized); original_indices.append(i)
    return processed_signals, original_indices

def preprocess_generic(signal_list): # For finger/glove data
    processed_signals = []; original_indices = []
    for i, signal in enumerate(signal_list):
        if isinstance(signal, np.ndarray) and signal.ndim == 2 and signal.shape[0] > 0 and signal.shape[1] > 0:
            # Optional: Add normalization or other preprocessing for generic signals if needed
            # For now, just pass them through if they meet basic criteria
            processed_signals.append(signal); original_indices.append(i)
    return processed_signals, original_indices

def extract_features_sequence(processed_signals_list, window_samples, step_samples, feature_type='emg'):
    all_feature_sequences = []
    processed_indices_map = []
    num_channels = 0
    if not processed_signals_list: return all_feature_sequences, processed_indices_map
    valid_signals = [s for s in processed_signals_list if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] > 0]
    if not valid_signals: return all_feature_sequences, processed_indices_map
    num_channels = valid_signals[0].shape[0]

    for idx, signal in enumerate(processed_signals_list):
        if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.shape[0] != num_channels or signal.shape[1] < window_samples: continue
        signal_length = signal.shape[1]; num_windows = (signal_length - window_samples) // step_samples + 1
        if num_windows <= 0: continue

        window_features_for_trial = []
        for i in range(num_windows):
            start = i * step_samples; end = start + window_samples; window = signal[:, start:end]
            if window.shape[1] != window_samples: continue # Should not happen
            try:
                if feature_type == 'emg':
                    mav=np.mean(np.abs(window),axis=1); var=np.var(window,axis=1); rms=np.sqrt(np.mean(window**2,axis=1))
                    diff=np.diff(window,axis=1); wl=np.sum(np.abs(diff),axis=1) if diff.size>0 else np.zeros(num_channels)
                    if window.shape[1] > 1: zc=np.sum(np.diff(np.sign(window+1e-9),axis=1)!=0,axis=1)/window.shape[1] # Normalize by window_samples
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
            except Exception: continue # Skip faulty window

        if window_features_for_trial:
            trial_feature_sequence = np.array(window_features_for_trial, dtype=np.float32)
            if trial_feature_sequence.ndim == 2 and trial_feature_sequence.shape[0] > 0 and trial_feature_sequence.shape[1] > 0:
                all_feature_sequences.append(trial_feature_sequence)
                processed_indices_map.append(idx)
    return all_feature_sequences, processed_indices_map


class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads) # key_dim per head
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim; self.num_heads=num_heads; self.ff_dim=ff_dim; self.rate=rate

    def call(self, inputs, training=False, mask=None): # Mask not currently used but good practice
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config(); config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}); return config

def create_transformer_model(input_shape, num_classes, num_layers_enc, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    sequence_length, num_features = input_shape # (max_len_emg, num_combined_features)
    if sequence_length is None or num_features is None: raise ValueError(f"Invalid input shape: ({sequence_length}, {num_features})")

    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    if embed_dim != num_features:
        x = layers.Dense(embed_dim, name="input_projection")(inputs)
    else:
        x = inputs

    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim, name="positional_embedding")(positions)
    x = x + pos_embeddings
    x = layers.Dropout(dropout_rate, name="pos_encoding_dropout")(x)

    for i in range(num_layers_enc):
        x = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate, name=f"transformer_encoder_{i}")(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)
    x = layers.Dropout(dropout_rate, name="final_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax", dtype='float32')(x) # Output in float32 for stability with mixed precision
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(), # Default Adam, can be configured
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(model, X_eval, y_eval, labels_display, task_name): # labels_display: list of string names
    print(f"\n--- Evaluating {task_name} Model ---")
    if X_eval.shape[0] == 0 or y_eval.shape[0] == 0:
        print(f"Evaluation skipped for {task_name}: Data is empty."); return

    try:
        eval_loss, eval_acc = model.evaluate(X_eval, y_eval, verbose=0)
        print(f"{task_name} Accuracy: {eval_acc:.4f}\n{task_name} Loss: {eval_loss:.4f}")
    except Exception as e: print(f"Error during model evaluation for {task_name}: {e}"); return

    try:
        y_pred_probs = model.predict(X_eval)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
    except Exception as e: print(f"Error during prediction for {task_name}: {e}"); return

    # Integer labels for confusion matrix and report [0, 1, ..., N-1]
    cm_report_labels_indices = list(range(len(labels_display)))

    try:
        cm = confusion_matrix(y_eval, y_pred_classes, labels=cm_report_labels_indices)
        plt.figure(figsize=(max(8, len(labels_display)), max(6, len(labels_display)*0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_display, yticklabels=labels_display)
        plt.xlabel('Predicted Label'); plt.ylabel('Actual Label')
        plt.title(f'{task_name} - Confusion Matrix'); plt.tight_layout(); plt.show()
    except Exception as e: print(f"Error displaying Confusion Matrix for {task_name}: {e}")

    print(f"\n--- {task_name} Classification Report ---")
    try:
        print(classification_report(y_eval, y_pred_classes, labels=cm_report_labels_indices, target_names=labels_display, zero_division=0))
    except Exception as e: print(f"Error generating Classification Report for {task_name}: {e}")


async def main_async():
    # GPU and Mixed Precision Setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} Physical GPUs.")
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set for GPUs.")
            # Attempt to set mixed precision policy
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision policy set to 'mixed_float16'.")
        except RuntimeError as e: print(f"RuntimeError during GPU setup: {e}")
        except Exception as e_mp: print(f"Could not set mixed precision policy: {e_mp}")
    else:
        print("No GPU detected. Running on CPU. Mixed precision not applied.")

    tasks_to_process = [(s, d, b) for s in range(1, 9) for d in [1, 2] for b in [1, 2]]
    print(f"--- Phase 1: Loading Multi-Modal Data ({len(tasks_to_process)} blocks) ---")
    loading_tasks = [load_multimodal_data_async(f"{BASE_DATA_PATH}/participant_{s}/participant{s}_day{d}_block{b}") for s, d, b in tasks_to_process]
    block_results_raw = await async_tqdm.gather(*loading_tasks, desc="Loading Blocks")
    block_results_raw = [res for res in block_results_raw if res is not None]
    if not block_results_raw: print("Error: No data loaded after filtering."); return None
    print(f"--- Phase 1 Finished: Loaded raw data from {len(block_results_raw)} blocks ---")

    print("\n--- Phase 2a: Estimating Sampling Rates & Synchronized Duration ---")
    trial_lengths_samples = {}; assumed_trial_length_s = None
    fs_emg = 2000.0; fs_finger = None; fs_glove = None
    has_finger = False; has_glove = False

    for i, block_data_tuple in enumerate(block_results_raw):
        # Adjusted unpacking: y_grasp_dict is now directly the second element of the tuple part
        (X_emg, X_finger, X_glove), y_grasp_d, params, valid_indices = block_data_tuple
        if X_finger and len(X_finger) > 0: has_finger = True
        if X_glove and len(X_glove) > 0: has_glove = True
        if not assumed_trial_length_s and params and 'trial_length_time' in params:
            assumed_trial_length_s = params['trial_length_time']
            print(f"Assumed trial length from metadata: {assumed_trial_length_s}s")
        if not assumed_trial_length_s: continue

        for idx in valid_indices:
            current_global_idx = (i, idx)
            if idx in X_emg and isinstance(X_emg[idx], np.ndarray) and X_emg[idx].ndim == 2:
                trial_lengths_samples.setdefault('emg', {})[current_global_idx] = X_emg[idx].shape[1]
            if X_finger and idx in X_finger and isinstance(X_finger[idx], np.ndarray) and X_finger[idx].ndim == 2:
                trial_lengths_samples.setdefault('finger', {})[current_global_idx] = X_finger[idx].shape[1]
            if X_glove and idx in X_glove and isinstance(X_glove[idx], np.ndarray) and X_glove[idx].ndim == 2:
                trial_lengths_samples.setdefault('glove', {})[current_global_idx] = X_glove[idx].shape[1]

    if not assumed_trial_length_s or assumed_trial_length_s <= 0:
        print(f"Error: Invalid or undetermined 'trial_length_time' ({assumed_trial_length_s}). Cannot estimate rates."); return None

    fs_estimates = {}; print("Estimated Sampling Rates (Hz):")
    # ... (Sampling rate estimation logic - largely unchanged but ensure it uses assumed_trial_length_s correctly)
    emg_lengths = list(trial_lengths_samples.get('emg', {}).values())
    if emg_lengths: fs_estimates['emg'] = np.mean(emg_lengths) / assumed_trial_length_s; fs_emg = fs_estimates['emg']; print(f"  EMG: {fs_emg:.2f}")
    else: print(f"  EMG: Using default {fs_emg:.2f} Hz (estimation failed).")
    if has_finger:
        finger_lengths = list(trial_lengths_samples.get('finger', {}).values())
        if finger_lengths: fs_estimates['finger'] = np.mean(finger_lengths) / assumed_trial_length_s; fs_finger = fs_estimates['finger']; print(f"  Finger: {fs_finger:.2f}")
        else: print(f"  Finger: Found data but could not estimate rate.")
    if has_glove:
        glove_lengths = list(trial_lengths_samples.get('glove', {}).values())
        if glove_lengths: fs_estimates['glove'] = np.mean(glove_lengths) / assumed_trial_length_s; fs_glove = fs_estimates['glove']; print(f"  Glove: {fs_glove:.2f}")
        else: print(f"  Glove: Found data but could not estimate rate.")

    min_duration_s = assumed_trial_length_s
    print(f"Using synchronized duration: {min_duration_s}s for truncation.")
    target_samples = {}
    target_samples['emg'] = int(min_duration_s * fs_emg)
    if has_finger and fs_finger: target_samples['finger'] = int(min_duration_s * fs_finger)
    if has_glove and fs_glove: target_samples['glove'] = int(min_duration_s * fs_glove)
    print("Target samples per modality for truncation:", target_samples)
    if not target_samples.get('emg') or target_samples['emg'] == 0 : print("Error: Failed to calculate target samples for EMG."); return None

    print(f"\n--- Phase 2b: Processing Multi-Modal Data into Windowed Feature Sequences ---")
    all_seq_features_emg_raw = []; all_seq_features_finger_raw = []; all_seq_features_glove_raw = []
    all_y_grasp_processed_raw = [] # Only grasp labels now

    window_samples = {}; step_samples = {}
    window_samples['emg'] = int(WINDOW_DURATION_S * fs_emg); step_samples['emg'] = max(1, int(window_samples['emg'] * (1.0 - OVERLAP_RATIO)))
    if not window_samples.get('emg') or window_samples['emg'] == 0 : print("Error: Failed to calculate EMG window samples."); return None
    if has_finger and fs_finger:
        window_samples['finger'] = int(WINDOW_DURATION_S * fs_finger); step_samples['finger'] = max(1, int(window_samples['finger'] * (1.0 - OVERLAP_RATIO)))
    if has_glove and fs_glove:
        window_samples['glove'] = int(WINDOW_DURATION_S * fs_glove); step_samples['glove'] = max(1, int(window_samples['glove'] * (1.0 - OVERLAP_RATIO)))
    print("Window samples per modality:", window_samples); print("Step samples per modality:", step_samples)

    for block_idx, block_data_tuple in enumerate(sync_tqdm(block_results_raw, desc="Processing Blocks")):
        (X_emg_raw_block, X_finger_raw_block, X_glove_raw_block), y_grasp_dict_block, _, valid_indices_block = block_data_tuple
        list_emg_block, list_finger_block, list_glove_block, list_grasp_block = [], [], [], []

        target_emg_len = target_samples.get('emg', 0)
        target_finger_len = target_samples.get('finger', 0) if has_finger and fs_finger else 0
        target_glove_len = target_samples.get('glove', 0) if has_glove and fs_glove else 0

        temp_block_emg, temp_block_finger, temp_block_glove, temp_block_grasp = [],[],[],[]

        for trial_idx_in_block in valid_indices_block:
            sig_emg = X_emg_raw_block.get(trial_idx_in_block)
            if sig_emg is not None and isinstance(sig_emg, np.ndarray) and sig_emg.ndim == 2 and sig_emg.shape[1] >= window_samples['emg']:
                truncated_emg = sig_emg[:, :target_emg_len]
                temp_block_emg.append(truncated_emg)
                temp_block_grasp.append(y_grasp_dict_block[trial_idx_in_block])

                sig_finger = X_finger_raw_block.get(trial_idx_in_block) if X_finger_raw_block and has_finger else None
                if target_finger_len > 0 and sig_finger is not None and isinstance(sig_finger, np.ndarray) and sig_finger.ndim == 2 and sig_finger.shape[1] >= window_samples.get('finger', float('inf')):
                    temp_block_finger.append(sig_finger[:, :target_finger_len])
                else: temp_block_finger.append(None)

                sig_glove = X_glove_raw_block.get(trial_idx_in_block) if X_glove_raw_block and has_glove else None
                if target_glove_len > 0 and sig_glove is not None and isinstance(sig_glove, np.ndarray) and sig_glove.ndim == 2 and sig_glove.shape[1] >= window_samples.get('glove', float('inf')):
                    temp_block_glove.append(sig_glove[:, :target_glove_len])
                else: temp_block_glove.append(None)
            else: # Mismatch in lengths or EMG not valid
                if len(temp_block_emg) == len(temp_block_finger): temp_block_finger.append(None) # maintain alignment if this path is taken mid-block
                if len(temp_block_emg) == len(temp_block_glove): temp_block_glove.append(None)


        if not temp_block_emg: continue

        emg_proc_block, idx_map_emg_proc = preprocess_emg(temp_block_emg, fs=fs_emg)
        if not emg_proc_block: continue
        seq_feat_emg_block, idx_map_emg_feat = extract_features_sequence(emg_proc_block, window_samples['emg'], step_samples['emg'], 'emg')
        if not seq_feat_emg_block: continue

        # Align other data to successfully processed EMG sequences
        aligned_finger_signals = [temp_block_finger[idx_map_emg_proc[i]] for i in idx_map_emg_feat if idx_map_emg_proc[i] < len(temp_block_finger)]
        aligned_glove_signals = [temp_block_glove[idx_map_emg_proc[i]] for i in idx_map_emg_feat if idx_map_emg_proc[i] < len(temp_block_glove)]
        aligned_grasp_labels = [temp_block_grasp[idx_map_emg_proc[i]] for i in idx_map_emg_feat]

        all_seq_features_emg_raw.extend(seq_feat_emg_block)
        all_y_grasp_processed_raw.extend(aligned_grasp_labels)

        # Process and append finger features
        if has_finger and fs_finger and window_samples.get('finger'):
            finger_proc_block, idx_map_f_proc = preprocess_generic([s for s in aligned_finger_signals if s is not None])
            valid_finger_indices_in_aligned = [i for i,s in enumerate(aligned_finger_signals) if s is not None]

            if finger_proc_block:
                seq_feat_f_block, idx_map_f_feat = extract_features_sequence(finger_proc_block, window_samples['finger'], step_samples['finger'], 'generic')
                # Create a temporary list for this block's finger features, aligned with EMG features
                current_block_finger_feat = [None] * len(seq_feat_emg_block)
                for k, feat_idx_in_proc in enumerate(idx_map_f_feat): # feat_idx_in_proc is index into finger_proc_block
                    original_aligned_idx = valid_finger_indices_in_aligned[idx_map_f_proc[feat_idx_in_proc]]
                    current_block_finger_feat[original_aligned_idx] = seq_feat_f_block[k]
                all_seq_features_finger_raw.extend(current_block_finger_feat)
            else: all_seq_features_finger_raw.extend([None] * len(seq_feat_emg_block))
        else: all_seq_features_finger_raw.extend([None] * len(seq_feat_emg_block))

        # Process and append glove features (similar logic)
        if has_glove and fs_glove and window_samples.get('glove'):
            glove_proc_block, idx_map_g_proc = preprocess_generic([s for s in aligned_glove_signals if s is not None])
            valid_glove_indices_in_aligned = [i for i,s in enumerate(aligned_glove_signals) if s is not None]

            if glove_proc_block:
                seq_feat_g_block, idx_map_g_feat = extract_features_sequence(glove_proc_block, window_samples['glove'], step_samples['glove'], 'generic')
                current_block_glove_feat = [None] * len(seq_feat_emg_block)
                for k, feat_idx_in_proc in enumerate(idx_map_g_feat):
                    original_aligned_idx = valid_glove_indices_in_aligned[idx_map_g_proc[feat_idx_in_proc]]
                    current_block_glove_feat[original_aligned_idx] = seq_feat_g_block[k]
                all_seq_features_glove_raw.extend(current_block_glove_feat)
            else: all_seq_features_glove_raw.extend([None] * len(seq_feat_emg_block))
        else: all_seq_features_glove_raw.extend([None] * len(seq_feat_emg_block))


    print(f"--- Phase 2 Finished: Sequence processing complete ({len(all_seq_features_emg_raw)} EMG sequences) ---")
    if not all_seq_features_emg_raw: print("Error: No EMG feature sequences extracted."); return None

    print("\n--- Phase 3: Padding, Combining, Splitting, Training & Evaluation ---")
    valid_emg_sequences = all_seq_features_emg_raw
    all_y_grasp = np.array(all_y_grasp_processed_raw)
    num_trials = len(valid_emg_sequences)

    # Ensure all lists are aligned after processing
    if not (len(all_y_grasp) == num_trials and \
            len(all_seq_features_finger_raw) == num_trials and \
            len(all_seq_features_glove_raw) == num_trials):
        print(f"Critical Error: Length mismatch before padding. EMG: {num_trials}, Grasp: {len(all_y_grasp)}, Finger: {len(all_seq_features_finger_raw)}, Glove: {len(all_seq_features_glove_raw)}")
        return None

    max_len_emg = 0
    for seq in valid_emg_sequences:
        if seq is not None and hasattr(seq, 'shape') and len(seq.shape) > 0: max_len_emg = max(max_len_emg, seq.shape[0])
    if max_len_emg == 0 : print("Error: max_len_emg is 0. No valid sequences for padding."); return None
    print(f"Padding sequences to max length (num_windows): {max_len_emg}")

    X_emg_padded = pad_sequences(valid_emg_sequences, maxlen=max_len_emg, padding='post', dtype='float32', value=0.0)
    X_finger_padded, X_glove_padded = None, None

    # Pad Finger Data
    if has_finger and fs_finger:
        first_valid_finger_seq = next((s for s in all_seq_features_finger_raw if s is not None and s.ndim == 2 and s.shape[0] > 0 and s.shape[1] > 0), None)
        if first_valid_finger_seq is not None:
            num_finger_features = first_valid_finger_seq.shape[1]
            X_finger_padded = np.zeros((num_trials, max_len_emg, num_finger_features), dtype='float32')
            for i, seq_finger in enumerate(all_seq_features_finger_raw):
                if seq_finger is not None and seq_finger.ndim == 2 and seq_finger.shape[0] > 0 and seq_finger.shape[1] == num_finger_features:
                    len_to_copy = min(seq_finger.shape[0], max_len_emg)
                    X_finger_padded[i, :len_to_copy, :] = seq_finger[:len_to_copy, :]
        else: print("No valid finger sequences with features to pad.")

    # Pad Glove Data
    if has_glove and fs_glove:
        first_valid_glove_seq = next((s for s in all_seq_features_glove_raw if s is not None and s.ndim == 2 and s.shape[0] > 0 and s.shape[1] > 0), None)
        if first_valid_glove_seq is not None:
            num_glove_features = first_valid_glove_seq.shape[1]
            X_glove_padded = np.zeros((num_trials, max_len_emg, num_glove_features), dtype='float32')
            for i, seq_glove in enumerate(all_seq_features_glove_raw):
                if seq_glove is not None and seq_glove.ndim == 2 and seq_glove.shape[0] > 0 and seq_glove.shape[1] == num_glove_features:
                    len_to_copy = min(seq_glove.shape[0], max_len_emg)
                    X_glove_padded[i, :len_to_copy, :] = seq_glove[:len_to_copy, :]
        else: print("No valid glove sequences with features to pad.")

    features_to_combine = [X_emg_padded]
    if X_finger_padded is not None: features_to_combine.append(X_finger_padded)
    if X_glove_padded is not None: features_to_combine.append(X_glove_padded)

    try: all_X_combined_padded = np.concatenate(features_to_combine, axis=-1)
    except ValueError as e: print(f"Error concatenating features: {e}"); return None
    print(f"Combined Padded Feature shape: {all_X_combined_padded.shape}")
    print("Grasp Label Distribution Before Split:", Counter(all_y_grasp))

    # Data Splitting (20% Test, then 80/20 for Train/Val of remaining)
    print("\nSplitting data...")
    unique_grasp_classes, grasp_counts = np.unique(all_y_grasp, return_counts=True)
    stratify_grasp = all_y_grasp if len(unique_grasp_classes) > 1 and grasp_counts.min() >= 2 else None
    if stratify_grasp is None and len(unique_grasp_classes)>0 : print("Warning: Stratification for initial split may not be possible due to class counts.")

    try:
        X_train_val, X_test, y_grasp_train_val, y_grasp_test = train_test_split(
            all_X_combined_padded, all_y_grasp, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify_grasp)
    except ValueError: # Fallback if stratification fails
        print("Stratification failed for initial split. Splitting without.")
        X_train_val, X_test, y_grasp_train_val, y_grasp_test = train_test_split(
            all_X_combined_padded, all_y_grasp, test_size=0.20, random_state=RANDOM_STATE, stratify=None)


    unique_grasp_tv, grasp_counts_tv = np.unique(y_grasp_train_val, return_counts=True)
    stratify_grasp_tv = y_grasp_train_val if len(unique_grasp_tv) > 1 and grasp_counts_tv.min() >= 2 else None
    if stratify_grasp_tv is None and len(unique_grasp_tv)>0: print("Warning: Stratification for train/val split may not be possible.")

    try:
        X_train, X_val, y_grasp_train, y_grasp_val = train_test_split(
            X_train_val, y_grasp_train_val, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify_grasp_tv) # 0.20 of 0.80 = 0.16 (16% val)
    except ValueError:
        print("Stratification failed for train/val split. Splitting without.")
        X_train, X_val, y_grasp_train, y_grasp_val = train_test_split(
            X_train_val, y_grasp_train_val, test_size=0.20, random_state=RANDOM_STATE, stratify=None)

    total_samples = len(all_X_combined_padded)
    print(f"\nFinal Split Sizes (Total: {total_samples}):")
    print(f"  Training:   {len(X_train)} ({len(X_train)/total_samples*100:.2f}%)")
    print(f"  Validation: {len(X_val)} ({len(X_val)/total_samples*100:.2f}%)")
    print(f"  Test:       {len(X_test)} ({len(X_test)/total_samples*100:.2f}%)")
    print(f"Train Grasp Dist: {Counter(y_grasp_train)}, Val Grasp Dist: {Counter(y_grasp_val)}, Test Grasp Dist: {Counter(y_grasp_test)}")

    # Standardization
    print("\nStandardizing features...")
    if X_train.shape[0] == 0: print("Error: Training set is empty for standardization."); return None
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    if X_val.shape[0] > 0: X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
    else: X_val_scaled = X_val
    if X_test.shape[0] > 0: X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    else: X_test_scaled = X_test
    print(f"Scaled shapes: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")

    # Class Weights for Grasp Model
    print("\nCalculating class weights for Grasp model...")
    unique_grasp_classes_train, counts_grasp_train = np.unique(y_grasp_train, return_counts=True)
    grasp_class_weights_dict = None
    if len(unique_grasp_classes_train) > 1:
        try:
            grasp_weights_values = compute_class_weight('balanced', classes=unique_grasp_classes_train, y=y_grasp_train)
            grasp_class_weights_dict = dict(zip(unique_grasp_classes_train, grasp_weights_values))
            print("Grasp Class Weights:", {k: f"{v:.2f}" for k,v in grasp_class_weights_dict.items()})
        except Exception as e: print(f"Warning: Error computing grasp class weights: {e}.")
    else: print(f"Warning: Not enough unique grasp classes ({len(unique_grasp_classes_train)}) in training data for class weights.")

    # Model Creation
    num_grasp_classes_total = len(np.unique(all_y_grasp))
    input_tf_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    print(f"\nInput shape for Grasp Transformer model: {input_tf_shape}, Num Grasp Classes: {num_grasp_classes_total}")
    try:
        grasp_model = create_transformer_model(input_tf_shape, num_grasp_classes_total, NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT_RATE)
        grasp_model.summary()
        print("Grasp model created successfully.")
    except Exception as e: print(f"Error creating Grasp Keras model: {e}"); return None

    # Training
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    ]
    validation_data_fit = (X_val_scaled, y_grasp_val) if X_val_scaled.shape[0] > 0 and y_grasp_val.shape[0] > 0 else None
    if validation_data_fit is None : print("Warning: Validation data is empty. Training without validation callbacks.")

    print("\n--- Starting Grasp Model Training ---")
    grasp_history = grasp_model.fit(X_train_scaled, y_grasp_train,
                                    validation_data=validation_data_fit,
                                    epochs=150, batch_size=32, # Consider making epochs/batch_size configurable
                                    callbacks=callbacks_list if validation_data_fit else [],
                                    verbose=1,
                                    class_weight=grasp_class_weights_dict)

    # Evaluation
    grasp_labels_display = [f"Grasp {i}" for i in range(num_grasp_classes_total)]
    print("\n--- Grasp Model Evaluation ---")
    if X_train_scaled.shape[0] > 0:
        evaluate_model(grasp_model, X_train_scaled, y_grasp_train, grasp_labels_display, "Grasp Train")
    if X_val_scaled.shape[0] > 0 and validation_data_fit is not None:
        evaluate_model(grasp_model, X_val_scaled, y_grasp_val, grasp_labels_display, "Grasp Validation")
    if X_test_scaled.shape[0] > 0:
        evaluate_model(grasp_model, X_test_scaled, y_grasp_test, grasp_labels_display, "Grasp Test")

    return grasp_model, grasp_history


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    # For saving/loading models with custom layers like TransformerEncoderLayer:
    # from tensorflow.keras.utils import get_custom_objects
    # get_custom_objects().update({'TransformerEncoderLayer': TransformerEncoderLayer})

    results = asyncio.run(main_async())

    if results: print("\nAsync grasp model pipeline finished successfully.")
    else: print("\nAsync grasp model pipeline did not complete successfully or exited early.")