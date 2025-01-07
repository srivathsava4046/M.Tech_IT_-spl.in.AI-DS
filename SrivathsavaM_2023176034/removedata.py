import os

# Specify the path to your dataset directory
dataset_dir = "D:/mtechprojects/projects/up-detr/tiny-imagenet-200/tiny-imagenet-200/train"

# Iterate through all files in the dataset directory
for filename in os.listdir(dataset_dir):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        file_path = os.path.join(dataset_dir, filename)
        os.remove(file_path)  # Remove the text file
        print(f"Removed: {file_path}")

