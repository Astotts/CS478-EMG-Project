import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM

# --- Define File Paths (Adapt these to your full dataset structure) ---
base_path = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data"
participant_folders = [f"participant_{i}" for i in range(1, 9)]
print(participant_folders) # Add this line

all_emg_data = []
all_labels = []

# --- Data Loading and Basic Feature Extraction (Illustrative) ---
def extract_basic_features(emg_signal, window_size=150, step_size=75):
    """Extracts Mean Absolute Value (MAV) for each channel in sliding windows."""
    features = []
    num_channels, num_samples = emg_signal.shape
    for i in range(0, num_samples - window_size + 1, step_size):
        window = emg_signal[:, i:i + window_size]
        mav = np.mean(np.abs(window), axis=1)
        features.append(mav)
    return np.array(features).T # Transpose to (num_windows, num_channels)

for participant_folder in participant_folders:
    for day in [1, 2]:
        for block in [1, 2]:
            participant_number = participant_folder.split('_')[1]
            folder_path = f"{base_path}/{participant_folder}/participant{participant_number}_day{day}_block{block}"
            emg_file_path = f"{folder_path}/emg_data.hdf5"
            trials_file_path = f"{folder_path}/trials.csv"

            try:
                print(emg_file_path)
                
                with h5py.File(emg_file_path, 'r') as f:
                    emg_data = {int(key): np.array(f[key]) for key in f.keys()}

                trials_df = pd.read_csv(trials_file_path)

                for trial_index in range(emg_data.shape[0]):
                    emg_signal = emg_data[trial_index, :, :]
                    features = extract_basic_features(emg_signal) # Shape: (num_windows, 16)
                    label_row = trials_df[trials_df['Trial_no'] == trial_index]
                    if not label_row.empty:
                        grasp_label = label_row['grasp'].values[0] - 1 # Adjust to 0-based indexing
                        position_label = label_row['target_position'].values[0] - 1 # Adjust to 0-based indexing

                        # For simplicity, let's just use grasp as the target for now
                        labels = np.full(features.shape[0], grasp_label) # Repeat label for each window
                        all_emg_data.append(features)
                        all_labels.append(labels)

            except FileNotFoundError:
                print(f"Could not find files in {folder_path}")

# --- Prepare Data for Model ---
if all_emg_data and all_labels:
    all_emg_data = np.vstack(all_emg_data) # Shape: (total_windows, 16)
    all_labels = np.hstack(all_labels)

    # Normalize features
    scaler = StandardScaler()
    scaled_emg_data = scaler.fit_transform(all_emg_data)
    scaled_emg_data = scaled_emg_data.reshape(-1, 16, 1) # Reshape for CNN (samples, channels, time_steps=1)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = encoder.fit_transform(all_labels.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(scaled_emg_data, encoded_labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # --- Build a Simple CNN Model ---
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(16, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=6, activation='softmax') # 6 grasp types
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # --- Train the Model ---
    epochs = 20
    batch_size = 32
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # --- Evaluate the Model ---
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

else:
    print("No data loaded.")