# 🚀🤖 Synthetic Data Generator 🚀🤖

## Overview

This Python script generates synthetic data for object detection tasks using OpenCV. It places transparent objects on background images, providing a diverse dataset for training and testing computer vision models.

## 🌟 Features

- **Overlay Placement**: Place transparent objects on background images.
- **Augmentation**: Apply various augmentations such as flipping, rotation, and motion blur to enhance dataset diversity.
- **YOLO Format**: Objects are defined in YOLO format, making it compatible with popular object detection frameworks.
- **Customization**: Easily customize the number of overlays, synthetic images, and other parameters.

## 📂 Required Folders
#### Objects Folder (path_of_objects):
Contains transparent images of objects.
Each object image should have a corresponding YOLO format bounding box file (.txt).
```bash
/path_of_objects
├── object1.png
├── object1.txt
├── object2.png
├── object2.txt
├── ...
```
#### Background Images Folder (bg_folder):
Contains background images where objects will be placed.
```bash
/bg_folder
├── background1.jpg
├── background2.jpg
├── background3.jpg
├── ...
```
#### Output Folder (output_folder):
The generated synthetic data will be saved in this folder.
```bash
/Generated_Data
├── synthetic_image_1.jpg
├── synthetic_image_1.txt
├── synthetic_image_2.jpg
├── synthetic_image_2.txt
├── ...
```

## 🔎 How to Use

### 📓 Prerequisites

- Python (3.x)
- OpenCV
- NumPy
- tqdm

### 🛠️ Installation

```bash
pip install opencv-python 
pip install numpy
pip install tqdm
```
### 🚀 Usage

```bash
python synthetic_generator.py
```
### 📌 Parameters
```bash
path_of_objects: Path to the folder containing objects and their YOLO format bounding box files.
bg_folder: Path to the folder containing background images.
output_folder: Path to the folder where generated data will be saved.
num_overlays: Number of overlays per background image.
num_of_synthetic_data: Number of synthetic images to generate.
center: Place the overlay at the center of the background (True only when num_overlays is 1).
init_augmentation: Enable initial augmentation for overlays.
flip: Flip overlay with 50% probability.
rotate: Rotate overlay with 25% probability.
Motion_Blur: Apply motion blur to overlays.
```

