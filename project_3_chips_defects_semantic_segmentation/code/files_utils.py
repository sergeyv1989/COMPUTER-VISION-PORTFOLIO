import imageio
import numpy as np
import os
from PIL import Image
from tifffile import imsave, imread
import yaml

def load_yaml(file_path: str):
    '''Load a .yaml file.'''

    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f'Error loading YAML from {file_path}: {e}')
            return None

def get_image_pairs_files(folders: list) -> dict:
    '''Return a dictionary of all pairs of file paths for the inspected and reference image in a directory.'''

    # Initializing variables
    file_pairs = {}

    # Going over the folder
    for folder in folders:
        files = os.listdir(folder)

        # Organizing files into pairs by going over the inspected images
        for file_name in files:
            if file_name.endswith('_inspected_image.tif'):
                case_id = file_name.split('_')[0] # Extract case ID
                inspected_image_file = os.path.join(folder, file_name)
                reference_image_file = os.path.join(folder, f'{case_id}_reference_image.tif')

                # Checking if the corresponding reference image exists
                if os.path.exists(reference_image_file):
                    file_pairs[case_id] = {
                        'inspected': inspected_image_file,
                        'reference': reference_image_file
                    }

    return file_pairs

def load_tif_image(file: str, return_np: bool=True, print_path: bool=True):
    try:
        image = imread(file)
        # image = Image.open(file)
        if return_np:
            image = np.array(image)
        if print_path:
            print(f'Loaded image from: {file}')
        return image
    except Exception as e:
        print(f"Error loading TIF image from {file}: {e}")
        return None
    
def save_tif_image(image: np.array, file: str):
    '''Save a NumPy array with values in the range [0,1] as a grayscale TIFF image.

    Args:
        image: NumPy array containing image data.
        file: File path to save the TIFF image.
    '''

    # Ensure the array is 2D (grayscale)
    if image.ndim != 2:
        raise ValueError("Input array must be 2D (grayscale).")

    # Convert data type to uint8 if not already
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Save the array as a TIFF image
    # tifffile.imsave(file, image, dtype=np.uint8)
    imageio.imwrite(file, image)
    print(f'Saved image to: {file}')