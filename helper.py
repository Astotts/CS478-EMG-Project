import h5py

file_path = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data/participant_1/participant1_day1_block1/emg_data.hdf5" # Use one of your actual file paths

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}: {list(f.keys())}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error opening {file_path}: {e}")