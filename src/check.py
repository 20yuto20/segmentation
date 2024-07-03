from PIL import Image
import os

"""
This code is used for searching and deleting the broken images and their corresponding files.
"""

def check_and_delete_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Confirm if the image is valid
                except Exception as e:
                    print(f"破損した画像ファイル {file_path}: {e}")
                    # Extract the file number (assuming the format is consistent and number can be extracted)
                    file_number = os.path.splitext(file)[0]
                    # Find and delete corresponding files in sibling directories
                    delete_corresponding_files(root, file_number)

def delete_corresponding_files(root, file_number):
    """
    This function is defined as follows.
    1. confirm the current dir that the broken image belongs to
    2. go to the parent dir 
    3. go into the every sub dir beloning to the parent dir
    4. delete the image if the number is as same as the broken image
    """
    base_dir = os.path.dirname(root)
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.startswith(file_number) and file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_to_delete = os.path.join(subdir_path, file)
                    os.remove(file_to_delete)
                    print(f"削除したファイル: {file_to_delete}")

check_and_delete_images('/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/CityScapes')
