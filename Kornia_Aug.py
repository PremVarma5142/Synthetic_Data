
import os
import kornia
import cv2
import random
from kornia.augmentation import (
    CenterCrop, ColorJiggle, ColorJitter, PadTo, RandomAffine, RandomBoxBlur,
    RandomBrightness, RandomChannelShuffle, RandomContrast, RandomCrop,
    RandomCutMixV2, RandomElasticTransform, RandomEqualize, RandomErasing,
    RandomFisheye, RandomGamma, RandomGaussianBlur, RandomGaussianNoise,
    RandomGrayscale, RandomHorizontalFlip, RandomHue, RandomInvert,
    RandomJigsaw, RandomMixUpV2, RandomMosaic, RandomMotionBlur,
    RandomPerspective, RandomPlanckianJitter, RandomPlasmaBrightness,
    RandomPlasmaContrast, RandomPlasmaShadow, RandomPosterize,
    RandomResizedCrop, RandomRGBShift, RandomRotation, RandomSaturation,
    RandomSharpness, RandomSolarize, RandomThinPlateSpline, RandomVerticalFlip
)
# Define the folder containing your images
# input_folder = "/home/raptor1/kornia/images"
# output_folder = "/home/raptor1/kornia/aug"
# os.makedirs(output_folder, exist_ok=True)
# List all image files in the input folder
# image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
# Define the number of augmented images you want to generate for each input image
num_augmented_images = 25  # Change this to your desired number
# Define an augmentation function to apply all augmentations
def apply_augmentations(image, folder):
    output_folder = f"/home/raptor1/Desktop/Vessels/Data/database/aug/{folder}"
    os.makedirs(output_folder, exist_ok=True)
    augmented_image = image
    # Random horizontal flip
    if random.random() > 0.5:
        augmented_image = RandomHorizontalFlip()(augmented_image)
    # Random rotation within a valid range
    rotation_degree = random.uniform(0, 20)  # Degrees in the range [0, 360]
    augmented_image = RandomRotation(rotation_degree, align_corners=True)(augmented_image)
    # augmented_image = RandomBoxBlur((7, 2), "reflect")(augmented_image)
    augmented_image = RandomBrightness(brightness=(0.5, 0.9), clip_output=True)(augmented_image)
    augmented_image = RandomChannelShuffle()(augmented_image)
    augmented_image = RandomGrayscale()(augmented_image)
    augmented_image = RandomGaussianBlur((7, 7), (0.2, 0.8), "reflect")(augmented_image)
    augmented_image = RandomHue((-0.2, 0.4))(augmented_image)
    # augmented_image = RandomMotionBlur((7, 7), 35.0, 0.5, "reflect", "nearest")(augmented_image)
    # augmented_image = RandomSaturation((1.0, 1.0))(augmented_image)
    return augmented_image, output_folder
# Iterate through the image files and apply augmentations
for folder in os.listdir("/home/raptor1/Desktop/Vessels/Data/database"):
    input_folder = os.path.join("/home/raptor1/Desktop/Vessels/Data/database", folder)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img_type = kornia.io.ImageLoadType.RGB32
        img = kornia.io.load_image(image_path, img_type, "cpu")[None]
        if len(os.listdir(image_path)) < 5000:
            for i in range(num_augmented_images):
                augmented_img, output_folder = apply_augmentations(img, folder)
                # Convert the augmented image to a NumPy array
                augmented_img_np = augmented_img[0].cpu().numpy().transpose(1, 2, 0)
                # Save the augmented image using OpenCV
                output_file = f"{os.path.splitext(image_file)[0]}_augmented_{i}.jpg"
                save_path = os.path.join(output_folder, output_file)
                cv2.imwrite(save_path, (augmented_img_np * 255).astype('uint8'))
        else:
            break
    print(f"{num_augmented_images} augmented images saved in {output_folder}")
