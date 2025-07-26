import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# dataset
dataset_path = r"C:\Users\acer\Desktop\SEM 5\CO543 Image Processing\Mini project\landscape Images\color"

processed_images = []

# preparing data
# Load and clean dataset
for file_name in os.listdir(dataset_path):
    try:
        img_path = os.path.join(dataset_path, file_name)
        img = Image.open(img_path).convert('RGB')  # Ensure all images are in RGB
        img = img.resize((128, 128))  # Resize for uniformity
        processed_images.append(np.array(img))  # Convert to numpy array
    except Exception as e:
        print(f"Skipping corrupt file: {file_name} - {e}")

processed_images = np.array(processed_images) / 255.0  # Normalize to [0, 1]
print(f"Loaded {len(processed_images)} images.")

# create variations of an image by applying random transformations
# These transformations help artificially increase the diversity of the dataset since have limited data sets
# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example: Augment a single image
img_sample = processed_images[0].reshape((1, 128, 128, 3))  
augmented_images = []

for _ in range(5):  # Generate 5 augmented images
    for batch in datagen.flow(img_sample, batch_size=1):
        augmented_images.append(batch[0])
        break

# Display augmented images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis('off')
plt.show()

# applies damage 
def add_damage(image, mask_size=(32, 32)):
    damaged_image = image.copy()
    h, w, _ = image.shape

    # Randomly select a top-left corner for the mask
    x = np.random.randint(0, w - mask_size[0])
    y = np.random.randint(0, h - mask_size[1])

    # Add a black mask
    damaged_image[y:y + mask_size[1], x:x + mask_size[0], :] = 0
    return damaged_image

# Apply damage to images
damaged_images = [add_damage(img) for img in processed_images]

# Display original and damaged images
plt.figure(figsize=(10, 5))
# just only 5 images to check
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(processed_images[i])
    plt.axis('off')

for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(damaged_images[i])
    plt.axis('off')

plt.show()

# Build the Inpainting Model
# encoder-decoder-based CNN model
# Convolutional Neural Network (CNN) model for image inpainting
def build_inpainting_model():
    # Encoder
    inputs = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

# Instantiate and compile the model
model = build_inpainting_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Model summary
# model summary provides a detailed breakdown of the architecture, layer shapes, and parameters.
model.summary()

# trained to reconstruct original images from damaged images.
# Use 80% for training and 20% for testing
train_images, test_images, train_damaged, test_damaged = train_test_split(
    processed_images, damaged_images, test_size=0.2, random_state=42
)

# Train the model
# total number of batches in one epoch
# is determined by the size of the training dataset and the batch size that specified when training the model
history = model.fit(
    x=np.array(train_damaged),  # Input: Damaged images
    y=np.array(train_images),  # Output: Original images
    epochs=10,
    batch_size=16,
    validation_data=(np.array(test_damaged), np.array(test_images))
)
# Save test data for later use
np.save('test_damaged.npy', test_damaged)
np.save('test_images.npy', test_images)

model.save('inpainting_model.h5')
