import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_black_square(image_path, output_path, mask_size=(32, 32)):
    
    image = Image.open(image_path).convert('RGB') 
    image = np.array(image) / 255.0  

    # Get image dimensions
    h, w, _ = image.shape

    # Randomly select a top-left corner for the mask
    x = np.random.randint(0, w - mask_size[0])
    y = np.random.randint(0, h - mask_size[1])

    # Add a black square
    image[y:y + mask_size[1], x:x + mask_size[0], :] = 0

    # Save the damaged image
    damaged_image = (image * 255).astype(np.uint8)  # Convert back to [0, 255]
    Image.fromarray(damaged_image).save(output_path)

    # Display the damaged image
    plt.imshow(image)
    plt.title("Damaged Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # input image
    input_image_path = r"C:\Users\Dineth\Desktop\Image Inpainting_ Restoring Damaged Areas\Images\pexels-photo-2379005.jpeg"

    # damaged image
    output_image_path = r"C:\Users\Dineth\Desktop\Image Inpainting_ Restoring Damaged Areas\Images\result.jpeg"

    # Add a 32x32 black square to the image
    add_black_square(input_image_path, output_image_path, mask_size=(32, 32))