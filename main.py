import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Update the dataset paths to match the user's specific paths
DATASET_PATH = 'dataset'  # Base dataset path
FLOODED_PATH = os.path.join(DATASET_PATH, 'flood_images')  # Updated path
NONFLOODED_PATH = os.path.join(DATASET_PATH, 'non_flood_images')  # Updated path
MODEL_PATH = 'flood_detection_model.h5'

# Image parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# Check if dataset exists
if not os.path.exists(FLOODED_PATH) or not os.path.exists(NONFLOODED_PATH):
    print(f"Dataset not found. Please ensure you have 'flood_images' and 'non_flood_images' folders in {DATASET_PATH}")
    exit(1)

# Count images
flooded_count = len(os.listdir(FLOODED_PATH))
nonflooded_count = len(os.listdir(NONFLOODED_PATH))
print(f"Found {flooded_count} flooded images and {nonflooded_count} non-flooded images")

# Create a combined dataset for training with proper labels
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# We need to handle the separate folders differently
# For training data
train_flood_generator = train_datagen.flow_from_directory(
    os.path.dirname(FLOODED_PATH),  # Use parent directory
    classes=['flood_images'],  # Specify the folder name
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

train_nonflood_generator = train_datagen.flow_from_directory(
    os.path.dirname(NONFLOODED_PATH),  # Use parent directory
    classes=['non_flood_images'],  # Specify the folder name
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# For validation data
val_flood_generator = train_datagen.flow_from_directory(
    os.path.dirname(FLOODED_PATH),  # Use parent directory
    classes=['flood_images'],  # Specify the folder name
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

val_nonflood_generator = train_datagen.flow_from_directory(
    os.path.dirname(NONFLOODED_PATH),  # Use parent directory
    classes=['non_flood_images'],  # Specify the folder name
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


# Combine generators
def combine_generators(gen1, gen2):
    while True:
        x1, y1 = next(gen1)  # Use next() function instead of .next() method
        x2, y2 = next(gen2)  # Use next() function instead of .next() method
        # Assign 1 to flood images and 0 to non-flood images
        y1[:] = 1
        y2[:] = 0
        yield np.concatenate([x1, x2]), np.concatenate([y1, y2])


# Combined generators
train_generator = combine_generators(train_flood_generator, train_nonflood_generator)
validation_generator = combine_generators(val_flood_generator, val_nonflood_generator)

# Build the CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),

    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (flood or non-flood)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
]

# Update steps_per_epoch and validation_steps
steps_per_epoch = min(train_flood_generator.samples, train_nonflood_generator.samples) // BATCH_SIZE * 2
validation_steps = min(val_flood_generator.samples, val_nonflood_generator.samples) // BATCH_SIZE * 2

# Then in the model.fit call, update the steps_per_epoch and validation_steps:
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=callbacks
)

# Save the model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_steps)
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Validation loss: {val_loss:.4f}")

# Print class indices for reference
print("Class indices:", train_flood_generator.class_indices)

