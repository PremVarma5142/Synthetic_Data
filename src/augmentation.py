import cv2
import numpy as np
import random

class ImageAugmentation:

    def __init__(self):
        pass

    def apply_brightness(self, image, factor):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        v_channel = cv2.add(v_channel, factor)
        v_channel = np.clip(v_channel, 0, 255)
        hsv_image = cv2.merge([h_channel, s_channel, v_channel])
        new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        alpha_channel = image[:, :, 3]
        bright_image = cv2.merge([new_image, alpha_channel])
        return bright_image

    def apply_contrast(self, image, factor):
        contrast_image = cv2.multiply(image, np.array([factor]))
        return contrast_image

    def apply_color_change(self, image, factor):
        b, g, r, alpha = cv2.split(image)
        b = cv2.add(b, factor)
        g = cv2.add(g, factor)
        r = cv2.add(r, factor)
        color_changed_image = cv2.merge([b, g, r, alpha])
        return color_changed_image

def apply_augmentation(overlay_img_resized, brightness_prob=0.0, brightness_factor=120, 
                       contrast_prob=0.2, contrast_factor=0.3, color_change_prob=0.2, color_change_factor=1.0):
    augmentation = ImageAugmentation()

    if random.random() < brightness_prob:
        overlay_img_resized = augmentation.apply_brightness(overlay_img_resized, brightness_factor)

    if random.random() < contrast_prob:
        overlay_img_resized = augmentation.apply_contrast(overlay_img_resized, contrast_factor)

    if random.random() < color_change_prob:
        overlay_img_resized = augmentation.apply_color_change(overlay_img_resized, color_change_factor)

    return overlay_img_resized
