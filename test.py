import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('inpainting_model.h5')

# Load a new image
new_image_path = r"C:\Users\Dineth\Desktop\Image Inpainting_ Restoring Damaged Areas\Images\testing01.jpeg"
new_image = Image.open(new_image_path).convert('RGB')  # Ensure RGB format

# Preprocess the image
new_image = new_image.resize((128, 128))  # Resize to match model input
new_image = np.array(new_image) / 255.0   # Normalize to [0, 1]

# Add a 16x16 black square
damaged_image = new_image.copy()
mask_size = 16
x = np.random.randint(0, 128 - mask_size)
y = np.random.randint(0, 128 - mask_size)
damaged_image[y:y + mask_size, x:x + mask_size, :] = 0

# Add batch dimension
damaged_image = np.expand_dims(damaged_image, axis=0)

# Predict the restoration
restored_image = model.predict(damaged_image)

# Display the damaged and restored images
plt.figure(figsize=(10, 5))

# Damaged image
plt.subplot(1, 2, 1)
plt.title("Damaged Image")
plt.imshow(np.squeeze(damaged_image)) 
plt.axis('off')

# Restored image
plt.subplot(1, 2, 2)
plt.title("Restored Image")
plt.imshow(np.squeeze(restored_image)) 
plt.axis('off')

plt.show()