import os
import random
import shutil
from pathlib import Path

data_path = Path("data")

raw_path = Path("raw_images")

# path to destination folders
train_folder = data_path / "train"
val_folder = data_path / "val"
test_folder = data_path / "test"

# Deleting datapath if exists
if os.path.exists(data_path) and os.path.isdir(data_path):
    shutil.rmtree(data_path)


# Define a list of image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
for dirpath, dirnames, filenames in os.walk(raw_path):
    imgs_list=[]
    for dirname in dirnames:
        # Getting the subdir path which will be the class name
        # Example raw_path/damaged
        subdirpath = raw_path / dirname

        # Create a list of image filenames in 'data_path'
        imgs_list = [filename for filename in os.listdir(subdirpath) if os.path.splitext(filename)[-1] in image_extensions]
        
        # determine the number of images for each set
        train_size = int(len(imgs_list) * 0.7)
        val_size = int(len(imgs_list) * 0.2)
        test_size = int(len(imgs_list) * 0.1)

        # Sets the random seed 
        # random.seed(42)

        # Shuffle the list of image filenames
        random.shuffle(imgs_list)

        train_with_dir = train_folder / dirname
        test_with_dir = test_folder / dirname
        val_with_dir = val_folder / dirname

        # Create destination folders if they don't exist
        for folder_path in [train_with_dir, val_with_dir, test_with_dir]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


        # Copy image files to destination folders
        for i, f in enumerate(imgs_list):
            if i < train_size:
                dest_folder = train_with_dir
            elif i < train_size + val_size:
                dest_folder = val_with_dir
            else:
                dest_folder = test_with_dir
            shutil.copy(subdirpath / f, dest_folder / f)