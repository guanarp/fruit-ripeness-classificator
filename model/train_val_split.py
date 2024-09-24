import os
import random
import shutil

# Paths
train_images_folder = "./fruit-ripeness-classificator/datasets/images/train/"
train_labels_folder = "./fruit-ripeness-classificator/datasets/labels/train/"
val_images_folder = "./fruit-ripeness-classificator/datasets/images/val/"
val_labels_folder = "./fruit-ripeness-classificator/datasets/labels/val/"

# Create the validation directories if they don't exist
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Split ratio (e.g., 0.2 means 20% of the data will be moved to val)
split_ratio = 0.2

# Get list of all images in the train folder
image_files = [f for f in os.listdir(train_images_folder) if f.endswith('.jpg')]

# Shuffle the list to randomize the split
random.shuffle(image_files)

# Calculate the number of images to move to the val set
num_val = int(len(image_files) * split_ratio)

# Select the validation images
val_images = image_files[:num_val]

# Move selected images and their corresponding label files to the val folder
for image_file in val_images:
    # Move image
    train_image_path = os.path.join(train_images_folder, image_file)
    val_image_path = os.path.join(val_images_folder, image_file)
    shutil.move(train_image_path, val_image_path)
    
    # Move corresponding label
    label_file = image_file.replace('.jpg', '.txt')
    train_label_path = os.path.join(train_labels_folder, label_file)
    val_label_path = os.path.join(val_labels_folder, label_file)
    
    if os.path.exists(train_label_path):
        shutil.move(train_label_path, val_label_path)

print(f"Moved {num_val} images and their corresponding labels to the validation set.")
