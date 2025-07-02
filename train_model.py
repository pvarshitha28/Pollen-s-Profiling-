# train_model.py
import os
import re # Import regex module for filename parsing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
from collections import Counter # Import Counter for counting class frequencies



# --- Configuration ---
DATASET_PATH = 'dataset' # Your dataset folder containing all image files
MODEL_SAVE_PATH = 'pollen_cnn_model.h5'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20 # Adjust this based on your dataset size and complexity. Use EarlyStopping if you re-introduce validation.
MIN_SAMPLES_PER_CLASS = 1 # Keep classes even with a single image
OTHER_TYPE_CLASS_NAME = "Other_Type" # Define the name for merged classes (will be unused if MIN_SAMPLES_PER_CLASS is 1)

# In your train_model.py file

# ... (other imports and configurations) ...

# --- Filename Parsing Function ---
def extract_class_name(filename):
    """
    Extracts the pollen grain type from the filename, simplifying 'name_XX' to 'name'.
    Assumes format: "pollen_type_name (_optional_number).extension" or "pollen_type_name.extension".
    Example: "urochola (1).jpg" -> "urochola", "anadenanthera_16.jpg" -> "anadenanthera", "senegalia.png" -> "senegalia"
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]

    # First, handle the " (number)" pattern like "urochola (1)"
    match_paren = re.match(r'(.+?)(?:\s*\(\d+\))?$', name_without_ext)
    base_name = match_paren.group(1).strip() if match_paren else name_without_ext.strip()

    # Second, handle the "_number" pattern like "tridax_01"
    # This will split by the first underscore and take the first part
    if '_' in base_name:
        return base_name.split('_')[0]
    else:
        return base_name

# ... (rest of your train_model.py code) ...
# --- Model Definition ---
def build_cnn_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=5):
    """
    Builds a Sequential CNN model for pollen grain classification.
    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of distinct pollen grain classes.
    Returns:
        tf.keras.Model: The compiled Keras CNN model.
    """
    model = Sequential([
        # Add a preprocessing layer for normalization
        # Normalizes pixel values to [0, 1]. This replaces img_array /= 255.0
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),

        # Convolutional Layer 1
        Conv2D(32, (3, 3), activation='relu',
               padding='same', kernel_initializer='he_normal'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Convolutional Layer 2
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Convolutional Layer 3
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten the feature maps
        Flatten(),

        # Dense Layer 1
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.5),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot encoded labels
                  metrics=['accuracy'])
    return model

# --- Helper functions for tf.data pipeline ---
def parse_image_and_label(image_path, label):
    # Load and decode image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False) # Ensure 3 channels for RGB
    img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    # Normalization (1./255) is handled by the tf.keras.layers.Rescaling layer in the model.
    return img, label

def augment_data(image, label):
    # Apply various random augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Random brightness and contrast (adjust limits as needed)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random hue and saturation (adjust limits as needed)
    image = tf.image.random_hue(image, max_delta=0.08)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image, label


# --- Main Training Function ---
def train_pollen_classifier():
    """
    Prepares the dataset, builds, trains, and saves the CNN model.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found.")
        print("Please ensure your pollen grain images are directly inside the '{DATASET_PATH}' folder.")
        return

    all_image_paths = []
    all_labels_raw = []

    # Collect all image paths and extract labels from filenames
    print(f"\n--- Scanning dataset folder: {DATASET_PATH} ---")
    # No longer filtering 'corrupted_images' folder, assuming it's removed.
    for filename in os.listdir(DATASET_PATH):
        filepath = os.path.join(DATASET_PATH, filename)
        if os.path.isdir(filepath): # Still skip any *other* subdirectories that might exist
            continue
        if allowed_file(filename): # Check if it's a valid image file
            class_name = extract_class_name(filename)
            all_image_paths.append(filepath)
            all_labels_raw.append(class_name)

    if not all_image_paths:
        print(f"Error: No image files found in '{DATASET_PATH}'. Please check the folder content.")
        return

    # --- Logic for merging single-sample classes (will keep all if MIN_SAMPLES_PER_CLASS is 1) ---
    print("\n--- Processing labels ---")
    label_counts = Counter(all_labels_raw)
    single_sample_classes = {label for label, count in label_counts.items() if count < MIN_SAMPLES_PER_CLASS}

    if single_sample_classes:
        # This block will now only execute if a class literally has 0 samples (which shouldn't happen here)
        # or if MIN_SAMPLES_PER_CLASS was set higher than 1.
        print(f"Found classes with less than {MIN_SAMPLES_PER_CLASS} samples (will be merged into '{OTHER_TYPE_CLASS_NAME}'):")
        for cls in single_sample_classes:
            print(f"- '{cls}' (Count: {label_counts[cls]})")
    else:
        print(f"No classes found with less than {MIN_SAMPLES_PER_CLASS} samples.")
        if MIN_SAMPLES_PER_CLASS == 1:
            print("INFO: All detected classes have at least one sample and will be kept distinct.")

    all_labels_merged = []
    for label in all_labels_raw:
        if label in single_sample_classes:
            all_labels_merged.append(OTHER_TYPE_CLASS_NAME)
        else:
            all_labels_merged.append(label)

    # --- End of logic ---

    # Encode string labels to integers based on the MERGED labels
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(all_labels_merged)
    # Get the final list of class names after merging, in the order they are encoded
    CLASS_NAMES = list(label_encoder.classes_)
    num_classes = len(CLASS_NAMES)

    # Print class names in the order they are encoded (alphabetical)
    print("\n--- Detected Pollen Classes (Alphabetical Order) ---")
    print("INFO: Please copy the following list into CLASS_NAMES in app.py:")
    print(CLASS_NAMES)
    print("----------------------------------------------------\n")

    # Convert integer labels to one-hot encoded format (for model output)
    one_hot_labels = to_categorical(integer_encoded_labels, num_classes=num_classes)

    # --- Use all data for training ---
    # No train_test_split as VALIDATION_SPLIT is effectively 0
    train_image_paths = all_image_paths
    train_labels_one_hot = one_hot_labels

    print(f"Total images found: {len(all_image_paths)}")
    print(f"Training images: {len(train_image_paths)} (All available images)")
    print(f"Number of classes: {num_classes}")


    # --- Create tf.data.Dataset for Training ---
    print("\n--- Creating tf.data.Dataset for efficient loading and augmentation ---")

    AUTOTUNE = tf.data.AUTOTUNE

    # Training Dataset Pipeline (uses all data)
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_image_paths), tf.constant(train_labels_one_hot)))
    train_ds = train_ds.map(parse_image_and_label, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(augment_data, num_parallel_calls=AUTOTUNE) # Apply augmentation
    train_ds = train_ds.shuffle(buffer_size=1000) # Shuffle the dataset
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # No validation dataset (val_ds) is created or used for training/callbacks

    # --- Build the Model ---
    print("\n--- Building CNN Model ---")
    model = build_cnn_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=num_classes)
    model.summary()

    # --- Callbacks for Training ---
    # ModelCheckpoint will now monitor training accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='accuracy', # Changed from 'val_accuracy' to 'accuracy'
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    # EarlyStopping is generally not recommended without a separate validation set
    # as it might stop based on overfitting to the training data.
    # If you still want to use it, you'd monitor 'loss' and mode='min'.
    # For now, we'll remove it as per the request to train on all data.
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor='accuracy', # Changed from 'val_accuracy' to 'accuracy'
    #     patience=5,
    #     verbose=1,
    #     mode='max',
    #     restore_best_weights=True
    # )

    callbacks_list = [checkpoint] # Only checkpoint remains

    # --- Train the Model ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds, # Use the tf.data.Dataset directly
        epochs=EPOCHS,
        # No validation_data since all data is for training
        callbacks=callbacks_list
    )
    print("\n--- Model Training Finished ---")

    print(f"Best model saved to {MODEL_SAVE_PATH}")

    # Optional: Evaluate the final model on the training set (shows training performance)
    print("\n--- Evaluating Final Model on Training Data ---")
    loss, accuracy = model.evaluate(train_ds, verbose=1) # Evaluate on train_ds
    print(f"Training Loss: {loss:.4f}")
    print(f"Training Accuracy: {accuracy*100:.2f}%")


# Helper function for allowed file extensions (same as app.py)
def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


if __name__ == '__main__':
    train_pollen_classifier()