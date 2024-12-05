import requests
import os
from PIL import Image
import numpy as np

# Define the image path
image_path = "panda.jpg"

# Check if the image file exists
if os.path.exists(image_path):
    print(f"Found image: {image_path}")

    # Preprocess the image
    original_image = Image.open(image_path)
    resized_image = original_image.resize((64, 64))
    images_to_predict = np.expand_dims(np.array(resized_image), axis=0)

    # Test the healthcheck endpoint
    response = requests.get("http://localhost:8000/healthcheck")
    print(response.json())

    # Test the upload image endpoint
    with open(image_path, "rb") as img_file:
        response = requests.post("http://localhost:8000/upload/image", files={"img": img_file})
        print(response.json())
else:
    print(f"Image not found: {image_path}")