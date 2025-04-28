import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shutil

# Define dataset path
DATASET_DIR = "dataset"  # Your folder containing 'real' and 'fake' images

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# Create temporary training and validation directories
TRAIN_DIR = "temp_dataset/train"
VAL_DIR = "temp_dataset/val"

for category in ["real", "fake"]:
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)

# Split data automatically
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    files = os.listdir(source_dir)
    np.random.shuffle(files)

    split_idx = int(len(files) * split_ratio)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

# Apply to both classes
split_data(os.path.join(DATASET_DIR, "real"), os.path.join(TRAIN_DIR, "real"), os.path.join(VAL_DIR, "real"))
split_data(os.path.join(DATASET_DIR, "fake"), os.path.join(TRAIN_DIR, "fake"), os.path.join(VAL_DIR, "fake"))

print("✅ Dataset split completed!")

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (real/fake)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save model
model.save("fake_image_detector.h5")
print("✅ Model trained and saved!")

# Plot accuracy and loss
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()
