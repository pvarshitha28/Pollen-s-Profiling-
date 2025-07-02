# culprit_finder.py, during the process of training the model, few errors occured the primary cause of those errors is due to the problems within the dataset
#this script analyses the dataset and moves the unvalid files into a folder called "corrupted",  the cause for the corruption is not been found out
#there might be many reasons for the corruption
# ihave re - downloaded the dataset using thekaggle module within the code file itself
# this is through the file "kaggle_dataset_download.pu"
import os
import re
import tensorflow as tf
import shutil # Import shutil for file operations (moving files)

# --- Configuration (must match your dataset setup) ---
DATASET_PATH = 'dataset' # Your dataset folder containing all image files
CORRUPTED_FOLDER_NAME = 'corrupted_images' # Name of the folder to move corrupted files to
IMAGE_HEIGHT = 224 # Target height (can be arbitrary for this check, as decoding is the main point)
IMAGE_WIDTH = 224  # Target width (can be arbitrary for this check)

# Helper function for allowed file extensions (same as train_model.py)
def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# --- Main Culprit Finder Logic ---
def find_image_culprits(dataset_path, corrupted_dir_name):
    """
    Scans the specified dataset path for image files that TensorFlow cannot decode.
    Moves problematic files into a specified 'corrupted' folder.
    """
    print(f"--- Scanning '{dataset_path}' for corrupted or invalid image files ---")

    corrupted_path = os.path.join(dataset_path, corrupted_dir_name)

    # Create the 'corrupted_images' folder if it doesn't exist
    if not os.path.exists(corrupted_path):
        os.makedirs(corrupted_path)
        print(f"Created folder for corrupted files: '{corrupted_path}'")
    else:
        print(f"Corrupted files will be moved to existing folder: '{corrupted_path}'")


    problematic_files_moved = []
    total_files_scanned = 0

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found.")
        return

    # List all files in the dataset directory
    # Filter out the 'corrupted_images' folder itself to prevent scanning it recursively
    files_to_scan = [f for f in os.listdir(dataset_path) if f != corrupted_dir_name]

    for filename in files_to_scan:
        filepath = os.path.join(dataset_path, filename)

        # Skip directories
        if os.path.isdir(filepath):
            continue

        if allowed_file(filename):
            total_files_scanned += 1
            try:
                # Attempt to read and decode the image using TensorFlow
                img = tf.io.read_file(filepath)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

            except tf.errors.InvalidArgumentError as e:
                print(f"  [X] INVALID FORMAT/CORRUPTED: {filepath}")
                destination_path = os.path.join(corrupted_path, filename)
                try:
                    shutil.move(filepath, destination_path)
                    problematic_files_moved.append(destination_path)
                    print(f"      Moved to: {destination_path} (Reason: {e.message.splitlines()[0]})")
                except shutil.Error as move_error:
                    print(f"      ERROR moving file {filepath} to {destination_path}: {move_error}")
                except Exception as other_move_error:
                    print(f"      UNEXPECTED ERROR while moving {filepath}: {other_move_error}")

            except Exception as e:
                # Catch any other unexpected errors during processing
                print(f"  [X] UNEXPECTED ERROR: {filepath}")
                destination_path = os.path.join(corrupted_path, filename)
                try:
                    shutil.move(filepath, destination_path)
                    problematic_files_moved.append(destination_path)
                    print(f"      Moved to: {destination_path} (Reason: {e})")
                except shutil.Error as move_error:
                    print(f"      ERROR moving file {filepath} to {destination_path}: {move_error}")
                except Exception as other_move_error:
                    print(f"      UNEXPECTED ERROR while moving {filepath}: {other_move_error}")

        # else:
        #     # Uncomment the line below if you want to see files skipped due to extension
        #     print(f"  [-] SKIPPED (not allowed file type): {filepath}")

    print(f"\n--- Scan Complete ---")
    print(f"Total files scanned: {total_files_scanned}")

    if problematic_files_moved:
        print(f"\n!!! Successfully moved {len(problematic_files_moved)} problematic files to '{corrupted_path}':")
        for p_file in problematic_files_moved:
            print(f"- {p_file}")
        print("\nAction Required: Please inspect these files in the 'corrupted_images' folder. They might be corrupted, incomplete, or have incorrect extensions for their content. Consider repairing or permanently deleting them from that folder.")
    else:
        print("No corrupted or invalid image files found. Your dataset appears to be clean.")

if __name__ == '__main__':
    find_image_culprits(DATASET_PATH, CORRUPTED_FOLDER_NAME)