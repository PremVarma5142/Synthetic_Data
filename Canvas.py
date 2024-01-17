from PIL import Image
import cv2
import glob
import os
import numpy as np

images = glob.glob("path/*.png")
output_folder = "output_folder"

os.makedirs(output_folder, exist_ok=True)

for i in range(len(images)):
    filename = images[i].split('/')[-1][:-4]
    numeric_value = 12
    image = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)  
    with open("/home/raptor1/Downloads/big_new/" + filename + ".txt", "r") as f:
        lines = f.readlines()
        try:
            for idx, line in enumerate(lines):
                line = line.split(" ")
                label = int(line[0])
                x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                w = int(w * image.shape[1])
                h = int(h * image.shape[0])
                x = int(x * image.shape[1] - w / 2)
                y = int(y * image.shape[0] - h / 2)
                padding = 8
                x -= padding
                y -= padding
                w += 2 * padding
                h += 2 * padding
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)

                char_image = image[y:y + h, x:x + w]

                # Calculate canvas size dynamically based on the numeric value from the filename
                canvas_size = int(8 * numeric_value + max(char_image.shape[0], char_image.shape[1]) + 8 * padding)

                # Create transparent canvas
                canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

                # Determine the position on the canvas to place the cropped image (centered)
                canvas_x1 = (canvas_size - char_image.shape[1]) // 2
                canvas_x2 = canvas_x1 + char_image.shape[1]
                canvas_y1 = (canvas_size - char_image.shape[0]) // 2
                canvas_y2 = canvas_y1 + char_image.shape[0]

                # Use alpha_composite to overlay the cropped image on the canvas
                canvas_image = Image.fromarray(canvas, 'RGBA')
                canvas_image.paste(Image.fromarray(char_image), (canvas_x1, canvas_y1), Image.fromarray(char_image))

                # Convert the result back to numpy array
                result_image = np.array(canvas_image)

                # Save the character image while preserving transparency with the same name as the original image
                new_filename = f"{filename}_{idx}.png"
                cv2.imwrite(os.path.join(output_folder, new_filename), result_image)
        except Exception as E:
            continue
