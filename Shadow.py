import cv2
import numpy as np
from PIL import Image, ImageChops
import os

# Define the root directory and output directory here 
root_directory = '/home/raptor1/txt/Objects_Sorted'
output_directory = 'output_folder'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Function to apply shadow to an image
def apply_shadow(image_path, yolo_coordinates):
    # Load the PNG image with transparency
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is not None:
        # Apply a stronger Gaussian blur to the entire image
        sigmaX = 6  # Adjust this value for a stronger blur
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaX)

        # Get the dimensions of the original image
        image_height, image_width = image.shape[:2]

        # Define the X and Y offsets to move the blurred image
        x_offset = 20  # Adjust this value to move the blurred image left or right
        y_offset = -10  # Adjust this value to move the blurred image up or down

        # Calculate the dimensions of the canvas based on both images
        canvas_height = max(image_height, abs(y_offset) + image_height)
        canvas_width = max(image_width, abs(x_offset) + image_width)

        # Create a blank canvas with the appropriate dimensions
        canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

        # Calculate the position for the original and blurred images
        original_x = max(0, x_offset)
        original_y = max(0, y_offset)
        blurred_x = max(0, -x_offset)
        blurred_y = max(0, -y_offset)

        # Place the blurred image on the canvas, including transparency
        canvas[blurred_y:blurred_y + image_height, blurred_x:blurred_x + image_width] = blurred_image

        for line in yolo_coordinates:
            # Parse YOLO coordinates
            class_id, center_x, center_y, width, height = map(float, line.split())

            # Calculate pixel coordinates
            img_height, img_width, _ = image.shape
            x1 = int((center_x - width / 2) * img_width)
            y1 = int((center_y - height / 2) * img_height)
            x2 = int((center_x + width / 2) * img_width)
            y2 = int((center_y + height / 2) * img_height)

            # Crop the image
            cropped_image = Image.fromarray(image[y1:y2, x1:x2], 'RGBA')

            # Determine the region on the canvas to place the cropped image
            canvas_x1 = original_x + x1
            canvas_x2 = canvas_x1 + (x2 - x1)
            canvas_y1 = original_y + y1
            canvas_y2 = canvas_y1 + (y2 - y1)

            # Use alpha_composite to overlay the cropped image on the canvas
            canvas_image = Image.fromarray(canvas, 'RGBA')
            canvas_image.paste(cropped_image, (canvas_x1, canvas_y1), cropped_image)

        # Save the combined image with transparency
        output_path = os.path.join(output_directory, os.path.basename(image_path))
        canvas_image.save(output_path)

# Iterate through the root directory and process each image
for folder, _, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.png'):
            image_path = os.path.join(folder, file)
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as file:
                    yolo_coordinates = file.readlines()
                apply_shadow(image_path, yolo_coordinates)
