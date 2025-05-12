import asyncio
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm  # Import the regular tqdm
import os

async def load_data_async(subject_path):
    loop = asyncio.get_running_loop()
    # Load EMG data asynchronously
    emg_data = await loop.run_in_executor(None, _load_emg_data, f"{subject_path}/emg_data.hdf5")
    # Load labels asynchronously
    trials_df = await loop.run_in_executor(None, pd.read_csv, f"{subject_path}/trials.csv")

    # Find minimum signal length across all trials
    min_length = min([data.shape[1] for data in emg_data.values()])

    X = []
    y_grasp = []
    y_position = []

    for trial_idx in range(150):
        try:
            trial_info = trials_df.iloc[trial_idx]
            standardized_signal = emg_data[trial_idx][:, :min_length]
            X.append(standardized_signal)
            y_grasp.append(trial_info['grasp'] - 1)
            y_position.append(trial_info['target_position'] - 1)
        except Exception as e:
            print(f"Error processing trial {trial_idx}: {str(e)}")
            continue

    return np.array(X), (np.array(y_grasp), np.array(y_position))

def _load_emg_data(file_path):
    with h5py.File(file_path, 'r') as f:
        return {int(key): np.array(f[key]) for key in f.keys()}

async def process_subject_block_async(args):
    subject, day, block = args
    path = f"C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data/participant_{subject}/participant{subject}_day{day}_block{block}"
    print(f"Processing subject {subject}, day {day}, block {block}")
    try:
        X, (y_grasp, y_position) = await load_data_async(path)
        processed_X = preprocess_emg(X)
        features = extract_features(processed_X)
        return features, y_grasp, y_position
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

async def main_async():
    tasks = []
    for subject in range(1, 9):
        for day in [1, 2]:
            for block in [1, 2]:
                tasks.append((subject, day, block))

    all_X = []
    all_y_grasp = []
    all_y_position = []

    progress_bar = tqdm(total=len(tasks), desc="Processing Blocks")
    for task in tasks:
        result = await process_subject_block_async(task)
        if result is not None:
            features, y_grasp, y_position = result
            all_X.append(features)
            all_y_grasp.extend(y_grasp)
            all_y_position.extend(y_position)
        progress_bar.update(1)
    progress_bar.close()

    all_X = np.concatenate(all_X)
    all_y_grasp = np.array(all_y_grasp)
    all_y_position = np.array(all_y_position)

    (X_train, X_test,
     y_grasp_train, y_grasp_test,
     y_position_train, y_position_test) = train_test_split(
        all_X, all_y_grasp, all_y_position,
        test_size=0.2,
        random_state=42,
        stratify=all_y_grasp
    )

    (X_train, X_val,
     y_grasp_train, y_grasp_val,
     y_position_train, y_position_val) = train_test_split(
        X_train, y_grasp_train, y_position_train,
        test_size=0.25,
        random_state=42,
        stratify=y_grasp_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    num_grasp_classes = len(np.unique(all_y_grasp))
    num_position_classes = len(np.unique(all_y_position))

    model = create_multi_task_model(
        (X_train.shape[1],),
        num_grasps=num_grasp_classes,
        num_positions=num_position_classes
    )

    history = model.fit(
        X_train,
        {'grasp': y_grasp_train, 'position': y_position_train},
        validation_data=(
            X_val,
            {'grasp': y_grasp_val, 'position': y_position_val}
        ),
        epochs=100,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    return model

if __name__ == "__main__":
    model = asyncio.run(main_async())