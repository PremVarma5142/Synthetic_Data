import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import random
from src.augmentation import apply_augmentation
from src.transparent import transparent

class SyntheticGenerator:

    def __init__(self, path_of_objects, bg_folder=None, output_folder=None, num_of_synthetic_data=1000, 
                 center=False, init_augmentation=False , Motion_Blur = False):
        
        self.path_of_objects = path_of_objects
        self.bg_folder = bg_folder
        self.output_folder = output_folder
        self.num_of_synthetic_data = num_of_synthetic_data
        self.center = center
        self.init_augmentation = init_augmentation
        self.Motion_Blur = Motion_Blur

    def MotionBlur(self, image, kernel_size=5, intensity=1.0):
        if image.shape[-1] == 4:
            alpha_channel = image[:, :, 3]
            contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(alpha_channel)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            result_image_copy = image.copy()
            result_image_copy[:, :, :3][mask != alpha_channel] = cv2.GaussianBlur(result_image_copy[:, :, :3][mask != alpha_channel], (17, 17), 0)

            motion_blur_kernel = np.zeros((kernel_size, kernel_size))
            motion_blur_kernel[int((kernel_size - 1) / 2), :] = 1

            motion_blur_kernel /= kernel_size

            result_image_copy[:, :, :3] = cv2.filter2D(result_image_copy[:, :, :3], -1, motion_blur_kernel * intensity)

            return result_image_copy
        
    def augment_and_save_images(self, flip=True, rotate=True):

        overlay_images = glob.glob(os.path.join(self.path_of_objects, '*.[jJpP][pPnN][eEgG]'))
        overlay_bbox_files = [os.path.join(self.path_of_objects, os.path.basename(image)[:-4] + ".txt") for image in overlay_images]
        bgs = glob.glob(os.path.join(self.bg_folder, '*.[jJpP][pPnN][eEgG]'))

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for i in tqdm(range(self.num_of_synthetic_data), desc='Generating Synthetic Data'):


            bg_path = random.choice(bgs)
            bg = cv2.imread(bg_path)
            bgg_h, bgg_w = bg.shape[:2]

            all_new_lines = []
            img_with_overlays = bg.copy()

            num_overlays = random.randint(1, 4)

            for _ in range(num_overlays):
                overlay_index = random.randint(0, len(overlay_images) - 1)
                overlay_image_path = overlay_images[overlay_index]
                overlay_bbox_file = overlay_bbox_files[overlay_index]

                overlay_img = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

                if self.center:
                    x_offset = (bgg_w - 50) // 2
                    y_offset = (bgg_h - 50) // 2
                else:
                    x_offset = random.randint(100, bgg_w - 50 - 100)
                    y_offset = random.randint(100, bgg_h - 50 - 100)

                if y_offset > 600:
                    overlay_size_h = random.randint(120, 130)
                    overlay_size_w = random.randint(125, 130)

                elif 400 < y_offset < 599:
                    overlay_size_h = random.randint(115, 124)
                    overlay_size_w = random.randint(120, 124)

                elif 200 < y_offset < 399:
                    overlay_size_h = random.randint(105, 114)
                    overlay_size_w = random.randint(110, 114)

                elif 50 < y_offset < 199:
                    overlay_size_h = random.randint(95, 104)
                    overlay_size_w = random.randint(100, 104)

                else:
                    overlay_size_h = random.randint(85, 94)
                    overlay_size_w = random.randint(90, 94)
                                

                overlay_img_resized = cv2.resize(overlay_img, (overlay_size_h, overlay_size_w))

                if self.Motion_Blur:
                    overlay_img_resized = self.MotionBlur(overlay_img_resized)

                if self.init_augmentation:
                    overlay_img_resized = self.apply_augmentation(overlay_img_resized)

                if self.center and self.num_overlays != 1:
                    raise ValueError("center=True is applicable only when num_overlays is 1.")


                txt_lines_overlay = open(overlay_bbox_file).readlines()
                transparent_cls = transparent(self.center, num_overlays)

                new_lines = transparent_cls.add_transparent_image(img_with_overlays, overlay_img_resized, txt_lines_overlay,
                                                       x_offset=x_offset, y_offset=y_offset, flip=flip, rotate=rotate)
                all_new_lines.extend(new_lines)

            bg_filename = os.path.basename(bg_path).split('.')[0]
            output_filename = f"{bg_filename}_synthetic_mar20s_{i}"

            try:
                with open(os.path.join(self.output_folder, output_filename + ".txt"), "w") as f:
                    f.writelines(all_new_lines)
            except Exception as e:
                print("Error: ", e)

            cv2.imwrite(os.path.join(self.output_folder, f"{output_filename}.jpg"), img_with_overlays)

if __name__ == "__main__":
    path_of_objects = "big"
    bg_folder = "bg_fod"
    output_folder = "Generated_Data_PRINCE"
    num_of_synthetic_data = 50
    center = False
    init_augmentation = False
    flip = True
    rotate = False
    Motion_Blur = False

    processor = SyntheticGenerator(path_of_objects, bg_folder, output_folder, num_of_synthetic_data, center, 
                                   init_augmentation, Motion_Blur)
    processor.augment_and_save_images(flip=flip,rotate=rotate)