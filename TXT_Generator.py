import cv2
import os

input_folder = "ip"
output_folder = "op"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")): 
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] == 4:
            alpha_channel = image[:, :, 3]
            contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imwrite(output_path, image)
            txt_filename = os.path.splitext(output_path)[0] + ".txt"
            with open(txt_filename, "w") as txt_file:
                # YOLO format: <class_id> <center_x> <center_y> <width> <height>
                center_x = (x + x + w) / 2 / result_image.shape[1]
                center_y = (y + y + h) / 2 / result_image.shape[0]
                box_width = w / result_image.shape[1]
                box_height = h / result_image.shape[0]
                txt_file.write(f"1 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")




