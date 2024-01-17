import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import random

class SyntheticGenerator:

    def __init__(self, path_of_objects, bg_folder=None, output_folder=None, num_overlays=2, num_of_synthetic_data=1000, 
                 center=False, init_augmentation=False , Motion_Blur = False):
        
        self.path_of_objects = path_of_objects
        self.bg_folder = bg_folder
        self.output_folder = output_folder
        self.num_overlays = num_overlays
        self.num_of_synthetic_data = num_of_synthetic_data
        self.center = center
        self.init_augmentation = init_augmentation
        self.Motion_Blur = Motion_Blur

    def adjust_coordinates(self, x, y, box_width, box_height, overlay_w, overlay_h, bg_w, bg_h, bg_x, bg_y):
        x = (x * overlay_w + bg_x) / bg_w
        y = (y * overlay_h + bg_y) / bg_h
        box_width = box_width * overlay_w / bg_w
        box_height = box_height * overlay_h / bg_h
        return x, y, box_width, box_height

    def add_transparent_image(self, background, foreground, lines, x_offset=None, y_offset=None, flip=True, rotate=True):
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape
        if fg_w > bg_w or fg_h > bg_h:
            return []

        assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

        if self.center and self.num_overlays > 1:
            raise ValueError("center=True is applicable only when num_overlays is 1.")

        if self.center:
            x_offset = (bg_w - fg_w) // 2
            y_offset = (bg_h - fg_h) // 2
        else:
            if x_offset is None:
                x_offset = random.randint(100, bg_w - fg_w - 100)
            if y_offset is None:
                y_offset = random.randint(100, bg_h - fg_h - 100)

        if flip and random.random() < 0.5:
            foreground = cv2.flip(foreground, 1)  # 1 for horizontal flip
            x_offset = bg_w - x_offset - fg_w  # Adjust x-coordinate after flipping

            for i in range(len(lines)):
                line = lines[i].split(" ")
                x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                x = 1.0 - x 
                lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(line[0]), x, y, box_width, box_height)

        if rotate:
            rotation_angle = 0  
            if random.random() < 0.25:
                rotation_angle = 90  
            elif random.random() < 0.25:
                rotation_angle = 180 
            elif random.random() < 0.25:
                rotation_angle = 270
          
            if rotation_angle == 90:
                foreground = cv2.rotate(foreground, cv2.ROTATE_90_CLOCKWISE)
                fg_w, fg_h = fg_h, fg_w

                for i in range(len(lines)):
                    line = lines[i].split(" ")
                    x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    new_x = (1.0 - (y + box_height / 2)) + box_height / 2
                    new_y = x + box_width / 2 - box_width / 2
                    new_width = box_height
                    new_height = box_width
                    lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(line[0]), new_x, new_y, new_width, new_height)

                if x_offset is not None and y_offset is not None:
                    new_x_offset = x_offset - (fg_h - fg_w) // 2
                    new_y_offset = y_offset + (fg_h - fg_w) // 2
                    x_offset, y_offset = new_x_offset, new_y_offset

            elif rotation_angle == 180:
                foreground = cv2.rotate(foreground, cv2.ROTATE_180)
                fg_w, fg_h = fg_w, fg_h  

                for i in range(len(lines)):
                    line = lines[i].split(" ")
                    x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    new_x = 1.0 - x
                    new_y = 1.0 - y
                    new_width = box_width
                    new_height = box_height
                    lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(line[0]), new_x, new_y, new_width, new_height)

                if x_offset is not None and y_offset is not None:
                    new_x_offset = x_offset
                    new_y_offset = y_offset
                    x_offset, y_offset = new_x_offset, new_y_offset

            elif rotation_angle == 270:
                foreground = cv2.rotate(foreground, cv2.ROTATE_90_CLOCKWISE)
                fg_w, fg_h = fg_h, fg_w

                for i in range(len(lines)):
                    line = lines[i].split(" ")
                    x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    new_x = (1.0 - (y + box_height / 2)) + box_height / 2
                    new_y = x + box_width / 2 - box_width / 2
                    new_width = box_height
                    new_height = box_width
                    lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(line[0]), new_x, new_y, new_width, new_height)

                if x_offset is not None and y_offset is not None:
                    new_x_offset = x_offset - (fg_h - fg_w) // 2
                    new_y_offset = y_offset + (fg_h - fg_w) // 2
                    x_offset, y_offset = new_x_offset, new_y_offset

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1:
            return[]

        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
        new_lines = []
        for line in lines:
            line = line.split(" ")
            index = int(line[0])
            x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x, y, box_width, box_height = self.adjust_coordinates(x, y, box_width, box_height, fg_w, fg_h, bg_w, bg_h, bg_x, bg_y)
            new_lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(line[0]), x, y, box_width, box_height))

        return new_lines
    
    def MotionBlur(self, image):
        if image.shape[-1] == 4:
            alpha_channel = image[:, :, 3]
            contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(alpha_channel)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            result_image_copy = image.copy()
            result_image_copy[:, :, :3][mask != alpha_channel] = cv2.GaussianBlur(result_image_copy[:, :, :3][mask != alpha_channel], (35, 35), 0)
            
            kernel = np.ones((15, 15), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=25)
            result_image_copy[:, :, :3][dilated_mask != mask] = cv2.GaussianBlur(result_image_copy[:, :, :3][dilated_mask != mask], (35, 35), 0)
            
            motion_blur_kernel = np.array([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]]) / 16  
            num_additional_blurs = 5
            for _ in range(num_additional_blurs):
                result_image_copy[:, :, :3] = cv2.filter2D(result_image_copy[:, :, :3], -1, motion_blur_kernel)
            
            return result_image_copy

    def apply_augmentation(self, overlay_img_resized, brightness_prob=0.3, brightness_factor=120, 
                       contrast_prob=0.3, contrast_factor=1.0, color_change_prob=0.3, color_change_factor=1.0):
        def apply_brightness(image, factor):
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv_image)
            v_channel = cv2.add(v_channel, factor)
            v_channel = np.clip(v_channel, 0, 255)
            hsv_image = cv2.merge([h_channel, s_channel, v_channel])
            new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            alpha_channel = image[:, :, 3]
            bright_image = cv2.merge([new_image, alpha_channel])
            return bright_image

        def apply_contrast(image, factor):
            contrast_image = cv2.multiply(image, np.array([factor]))
            return contrast_image

        def apply_color_change(image, factor):
            b, g, r, alpha = cv2.split(image)
            b = cv2.add(b, factor)
            g = cv2.add(g, factor)
            r = cv2.add(r, factor)
            color_changed_image = cv2.merge([b, g, r, alpha])
            return color_changed_image

        if random.random() < brightness_prob:
            overlay_img_resized = apply_brightness(overlay_img_resized, brightness_factor)

        if random.random() < contrast_prob:
            overlay_img_resized = apply_contrast(overlay_img_resized, contrast_factor)

        if random.random() < color_change_prob:
            overlay_img_resized = apply_color_change(overlay_img_resized, color_change_factor)

        return overlay_img_resized


    def augment_and_save_images(self, overlay_resize_size=(200, 200), flip=True, rotate=True):

        overlay_images = glob.glob(os.path.join(self.path_of_objects, '*.[jJpP][pPnN][eEgG]'))
        overlay_bbox_files = [os.path.join(self.path_of_objects, os.path.basename(image)[:-4] + ".txt") for image in overlay_images]
        bgs = glob.glob(os.path.join(self.bg_folder, '*.[jJpP][pPnN][eEgG]'))

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        overlay_width = overlay_resize_size[0]
        overlay_height = overlay_resize_size[1]

        for i in tqdm(range(self.num_of_synthetic_data), desc='Generating Synthetic Data'):
            bg_path = random.choice(bgs)
            bg = cv2.imread(bg_path)
            bgg_h, bgg_w = bg.shape[:2]

            all_new_lines = []
            img_with_overlays = bg.copy()

            for _ in range(self.num_overlays):
                overlay_index = random.randint(0, len(overlay_images) - 1)
                overlay_image_path = overlay_images[overlay_index]
                overlay_bbox_file = overlay_bbox_files[overlay_index]

                overlay_img = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

                overlay_img_resized = cv2.resize(overlay_img, overlay_resize_size)

                if self.Motion_Blur:
                    overlay_img_resized = self.MotionBlur(overlay_img_resized)

                if self.init_augmentation:
                    overlay_img_resized = self.apply_augmentation(overlay_img_resized)

                if self.center and self.num_overlays != 1:
                    raise ValueError("center=True is applicable only when num_overlays is 1.")

                if self.center:
                    x_offset = (bgg_w - overlay_width) // 2
                    y_offset = (bgg_h - overlay_height) // 2
                else:
                    x_offset = random.randint(100, bgg_w - overlay_width - 100)
                    y_offset = random.randint(100, bgg_h - overlay_height - 100)

                txt_lines_overlay = open(overlay_bbox_file).readlines()

                new_lines = self.add_transparent_image(img_with_overlays, overlay_img_resized, txt_lines_overlay,
                                                       x_offset=x_offset, y_offset=y_offset, flip=flip, rotate=rotate)
                all_new_lines.extend(new_lines)

            bg_filename = os.path.basename(bg_path).split('.')[0]
            output_filename = f"{bg_filename}_synthetic_{i}"

            try:
                with open(os.path.join(self.output_folder, output_filename + ".txt"), "w") as f:
                    f.writelines(all_new_lines)
            except Exception as e:
                print("Error: ", e)

            cv2.imwrite(os.path.join(self.output_folder, f"{output_filename}.jpg"), img_with_overlays)

if __name__ == "__main__":
    path_of_objects = "Objects" # Includes TXT with transperent images for objects in YOLO format
    bg_folder = "oct12_bg" # Background images where we place our object
    output_folder = "Generated_Data"
    num_overlays = 1  # Set the desired number of overlays per background image
    num_of_synthetic_data = 10  # Set the desired number of synthetic images
    center = True # True only when num_overlays = 1
    init_augmentation = True  # Enable initial augmentation for overlays
    flip = True  # Flip overlay with 50% probability
    rotate = False # Rotate overlay with 25% probability
    Motion_Blur = False

    processor = SyntheticGenerator(path_of_objects, bg_folder, output_folder, num_overlays, num_of_synthetic_data, center, 
                                   init_augmentation, Motion_Blur)
    processor.augment_and_save_images(flip=flip,rotate=rotate)

