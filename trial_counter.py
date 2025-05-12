import pandas as pd
import os
# Removed unused imports: asyncio, h5py, numpy, tensorflow, sklearn, scipy, matplotlib, seaborn, tqdm

def count_positions_in_trials(base_data_path):
    totalCounts = [0,0,0,0,0,0,0,0,0,0]
    """
    Iterates through subject/day/block directories, reads trials.csv,
    and counts the occurrences of each target_position.

    Args:
        base_data_path (str): The root path containing the participant folders.
    """
    print(f"Starting position count analysis in: {base_data_path}")
    print("-" * 30)

    # Define the structure to iterate through
    subjects = range(1, 9) # Subjects 1-8
    days = [1, 2]
    blocks = [1, 2]

    # Dictionary to potentially store overall counts, though the request is per file
    # overall_position_counts = {}

    for subject in subjects:
        for day in days:
            for block in blocks:
                # Construct the path to the specific trials.csv file
                # Using os.path.join for better cross-platform compatibility
                block_folder = f"participant{subject}_day{day}_block{block}"
                participant_folder = f"participant_{subject}"
                file_path = os.path.join(
                    base_data_path,
                    participant_folder,
                    block_folder,
                    "trials.csv"
                )

                print(f"Processing: {file_path}")

                try:
                    # Read the CSV file using pandas
                    trials_df = pd.read_csv(file_path)

                    # Check if the required column exists
                    if 'target_position' not in trials_df.columns:
                        print(f"  -> Warning: 'target_position' column not found in this file. Skipping.")
                        print("-" * 10) # Separator
                        continue

                    # Count the occurrences of each value in the 'target_position' column
                    # .value_counts() returns a Series (value: count)
                    # .sort_index() sorts the counts by the position number (index)
                    position_counts = trials_df['target_position'].value_counts().sort_index()
                    for i in position_counts.index:
                        totalCounts[i] += position_counts[i]

                    if position_counts.empty:
                        print("  -> No 'target_position' data found or file is empty.")
                    else:
                        print("  Position Counts:")
                        # Iterate through the Series and print counts
                        
                        for position, count in position_counts.items():
                            print(f"    Position {position}: {count} entries")

                        # Example: Add to overall counts (optional)
                        # for pos, count in position_counts.items():
                        #     overall_position_counts[pos] = overall_position_counts.get(pos, 0) + count

                except FileNotFoundError:
                    print(f"  -> File not found. Skipping.")
                except pd.errors.EmptyDataError:
                     print(f"  -> File is empty. Skipping.")
                except Exception as e:
                    # Catch other potential errors during file reading or processing
                    print(f"  -> Error reading or processing file: {str(e)}. Skipping.")

                print("-" * 10) # Separator between files

    print("\nFinished position count analysis.")
    return totalCounts
    # Optionally print overall counts if you calculated them
    # print("\nOverall Position Counts Across All Files:")
    # if overall_position_counts:
    #     for position in sorted(overall_position_counts.keys()):
    #         print(f"  Position {position}: {overall_position_counts[position]} total entries")
    # else:
    #     print("No position data was successfully processed.")


# Main execution block
if __name__ == "__main__":
    # Define the base path to your dataset *exactly* as it is on your system
    # Make sure the path uses the correct slashes for your OS (Windows: \ or /; Linux/macOS: /)
    # Using raw string literal r"..." or forward slashes "/" is often safer on Windows
    data_directory = r"C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data"
    # or: data_directory = "C:/Users/Alexa/Documents/CS478 Dataset/posture_dataset_collection/data"

    if not os.path.isdir(data_directory):
        print(f"ERROR: Base data directory not found at '{data_directory}'")
        print("Please verify the path.")
    else:
        # Run the counting function
        finalCount = count_positions_in_trials(data_directory)

        for i in range(1,9):
            print(f"Position {i}: ", finalCount[i])