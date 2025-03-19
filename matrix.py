import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
from tqdm import tqdm  # For progress bars

# Set paths - using the specific model path you provided
MODEL_PATH = r"D:\Flood detection\flood_detection_model.h5"

# Keep the original dataset path structure
DATASET_PATH = 'dataset'  # Base dataset path
FLOODED_PATH = os.path.join(DATASET_PATH, 'flood_images')
NONFLOODED_PATH = os.path.join(DATASET_PATH, 'non_flood_images')

# Image parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# Option to limit the number of images for testing
# Set to None to process all images, or a number like 100 to process only that many
SAMPLE_LIMIT = None  # Change to a number like 100 for quick testing

print("Starting model visualization script...")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

# Load the pre-trained model
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Create data generators for evaluation
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create generators for flood and non-flood images
print("Creating data generators...")
try:
    flood_generator = test_datagen.flow_from_directory(
        os.path.dirname(FLOODED_PATH),
        classes=['flood_images'],
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    nonflood_generator = test_datagen.flow_from_directory(
        os.path.dirname(NONFLOODED_PATH),
        classes=['non_flood_images'],
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"Found {flood_generator.samples} flood images and {nonflood_generator.samples} non-flood images")

    # Apply sample limit if specified
    if SAMPLE_LIMIT is not None:
        flood_generator.samples = min(flood_generator.samples, SAMPLE_LIMIT)
        nonflood_generator.samples = min(nonflood_generator.samples, SAMPLE_LIMIT)
        print(f"Limited to {flood_generator.samples} flood images and {nonflood_generator.samples} non-flood images")

except Exception as e:
    print(f"Error creating data generators: {e}")
    exit(1)


# Function to get predictions and true labels with progress bar
def get_predictions(generator, is_flood=True, desc="Processing"):
    generator.reset()
    y_true = []
    y_pred = []
    confidences = []

    # Set the correct label based on image type
    true_label = 1 if is_flood else 0

    # Get predictions for all images
    steps = generator.samples // BATCH_SIZE + (1 if generator.samples % BATCH_SIZE > 0 else 0)

    # Use tqdm for progress bar
    for i in tqdm(range(steps), desc=desc, unit="batch"):
        try:
            x_batch, _ = next(generator)
            # Set verbose=0 to suppress output
            pred_batch = model.predict(x_batch, verbose=0)
            pred_labels = (pred_batch > 0.5).astype(int).flatten()

            # Add batch predictions and true labels
            y_pred.extend(pred_labels)
            y_true.extend([true_label] * len(pred_labels))
            confidences.extend(pred_batch.flatten())

            if len(y_true) >= generator.samples:
                break

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

    # Trim to actual sample count
    y_true = y_true[:generator.samples]
    y_pred = y_pred[:generator.samples]
    confidences = confidences[:generator.samples]

    return np.array(y_true), np.array(y_pred), np.array(confidences)


# Create output directory for visualizations
output_dir = "model_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Get predictions with timing
print("Evaluating model on flood images...")
start_time = time.time()
flood_true, flood_pred, flood_conf = get_predictions(flood_generator, is_flood=True, desc="Processing flood images")
flood_time = time.time() - start_time
print(f"Flood image evaluation completed in {flood_time:.2f} seconds")

print("Evaluating model on non-flood images...")
start_time = time.time()
nonflood_true, nonflood_pred, nonflood_conf = get_predictions(nonflood_generator, is_flood=False,
                                                              desc="Processing non-flood images")
nonflood_time = time.time() - start_time
print(f"Non-flood image evaluation completed in {nonflood_time:.2f} seconds")

# Combine results
y_true = np.concatenate([flood_true, nonflood_true])
y_pred = np.concatenate([flood_pred, nonflood_pred])

# Calculate accuracy
accuracy = np.mean(y_true == y_pred)
print(f"Overall accuracy: {accuracy:.4f}")

# Calculate class-wise accuracy
flood_accuracy = np.mean(flood_true == flood_pred)
nonflood_accuracy = np.mean(nonflood_true == nonflood_pred)

print(f"Flood images accuracy: {flood_accuracy:.4f}")
print(f"Non-flood images accuracy: {nonflood_accuracy:.4f}")

print("Generating visualizations...")

# Create bar chart for accuracy visualization
plt.figure(figsize=(10, 6))
categories = ['Flood Images', 'Non-Flood Images', 'Overall']
accuracies = [flood_accuracy, nonflood_accuracy, accuracy]
colors = ['#3498db', '#2ecc71', '#f39c12']

plt.bar(categories, accuracies, color=colors)
plt.title('Flood Detection Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_accuracy_evaluation.png'))
print(f"Saved accuracy chart to {os.path.join(output_dir, 'model_accuracy_evaluation.png')}")

# Create confusion matrix visualization
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Flood', 'Flood'],
            yticklabels=['Non-Flood', 'Flood'])
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
print(f"Saved confusion matrix to {os.path.join(output_dir, 'confusion_matrix.png')}")

# Generate confidence distribution plot
plt.figure(figsize=(10, 6))
plt.hist(flood_conf, alpha=0.7, bins=20, label='Flood Images', color='#3498db')
plt.hist(nonflood_conf, alpha=0.7, bins=20, label='Non-Flood Images', color='#e74c3c')
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('Confidence Score (Probability of Flood)', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.title('Model Confidence Distribution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
print(f"Saved confidence distribution to {os.path.join(output_dir, 'confidence_distribution.png')}")

# Generate ROC curve
# Combine confidences
all_confidences = np.concatenate([flood_conf, nonflood_conf])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, all_confidences)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
print(f"Saved ROC curve to {os.path.join(output_dir, 'roc_curve.png')}")

# Generate precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, all_confidences)
avg_precision = average_precision_score(y_true, all_confidences)

# Plot precision-recall curve
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="lower left", fontsize=12)
plt.grid(linestyle='--', alpha=0.6)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
print(f"Saved precision-recall curve to {os.path.join(output_dir, 'precision_recall_curve.png')}")

print(f"\nVisualization complete! All charts saved to '{output_dir}' directory.")