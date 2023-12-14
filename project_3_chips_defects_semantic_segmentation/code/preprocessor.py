import cv2
from numba import jit
import numpy as np
from scipy.signal import correlate2d
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from plots_utils import *

class Preprocessor():

    def __init__(self, cfg: dict) -> None:
        
        self.cfg = cfg

    def normalize_image_intensity(self, image: np.array, algo_type: str) -> np.array:
        '''Transform the pixel values of an image.'''

        if algo_type == 'scale_by_max':
            return 255 * (image / np.max(image))
        elif algo_type == 'divide_by_max':
            return image / np.max(image)
        elif algo_type == 'divide_by_constant':
            return image / 255
        elif algo_type == 'standardize':
            return (image - np.mean(image)) / np.std(image)
        else:
            raise ValueError(f"Unknown normalization algorithm: {algo_type}.")
    
    def matrix_mse(self, image_a, image_b):
        '''Calculate the MSE between two matrices.'''

        return np.sum((image_a - image_b) ** 2) / float(image_a.size)
    
    def register_images_by_submatrix_mse(self, image1: np.array, image2: np.array, **kwargs):
        '''Perform image registration for two images by comparing the similarity of their submatrices.'''

        # Initializing variables
        submatrix_size_ratio = kwargs['submatrix_size_ratio']
        image_size_step = kwargs['image_size_step']
        min_mse = float('inf')
        best_submatrix1 = None
        best_submatrix2 = None
        submatrix_size_x = int(submatrix_size_ratio * min(image1.shape[0], image2.shape[0]))
        submatrix_size_y = int(submatrix_size_ratio * min(image1.shape[1], image2.shape[1]))

        # Going over the possible submatrices in different positions and calculating the MSEs
        for i in tqdm(np.arange(0, image1.shape[0] - submatrix_size_x + 1, image_size_step), desc="image1 i"):
            for j in np.arange(0, image1.shape[1] - submatrix_size_y + 1, image_size_step):
                submatrix1 = image1[i:i+submatrix_size_x, j:j+submatrix_size_y]

                for x in np.arange(0, image2.shape[0] - submatrix_size_x + 1, image_size_step):
                    for y in np.arange(0, image2.shape[1] - submatrix_size_y + 1, image_size_step):
                        submatrix2 = image2[x:x+submatrix_size_x, y:y+submatrix_size_y]
                        current_mse = self.matrix_mse(submatrix1, submatrix2)
                        # Saving best results
                        if current_mse < min_mse:
                            min_mse = current_mse
                            best_submatrix1 = submatrix1
                            best_submatrix2 = submatrix2

        return best_submatrix1, best_submatrix2

    def register_images_by_cross_correlation(self, image1: np.array, image2: np.array):
        '''Perform image registration for two images by calculating their correlation.'''

        # Perform cross-correlation in 'full' mode
        cross_corr = correlate2d(image1, image2, mode='full')
        
        # Find the offset (shift)
        offset_y, offset_x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        
        # Crop the images based on the offset
        image1_cropped = image1[max(0, offset_y):min(image1.shape[0], image2.shape[0] + offset_y),
                                max(0, offset_x):min(image1.shape[1], image2.shape[1] + offset_x)]
        image2_cropped = image2[max(0, -offset_y):min(image2.shape[0], image1.shape[0] - offset_y),
                                max(0, -offset_x):min(image2.shape[1], image1.shape[1] - offset_x)]
        
        return image1_cropped, image2_cropped

    def register_images_by_cross_correlation_same(self, image1: np.array, image2: np.array):
        '''Perform image registration for two images by calculating their correlation.'''
        
        # Performing cross-correlation
        cross_corr = correlate2d(image1, image2, mode='same')
        # Finding the offset (shift)
        offset_y, offset_x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        # Cropping the images based on the offset
        image1_cropped = image1[max(0, offset_y):min(image1.shape[0], image2.shape[0] + offset_y),
                                    max(0, offset_x):min(image1.shape[1], image2.shape[1] + offset_x)]
        image2_cropped = image2[max(0, -offset_y):min(image2.shape[0], image1.shape[0] - offset_y),
                                    max(0, -offset_x):min(image2.shape[1], image1.shape[1] - offset_x)]

        return image1_cropped, image2_cropped
    
    def register_images_by_cv(self, image1, image2):
        '''Perform image registration for two images by finding the needed transformation based on similar keypoints.'''

        # Check if input images are 2D NumPy arrays
        if not (isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray)):
            raise ValueError("Input images must be 2D NumPy arrays.")

        # Ensure both images are treated as grayscale
        image1 = np.atleast_3d(image1)
        image2 = np.atleast_3d(image2)

        # Use ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Use Brute-Force Matcher to find best matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort the matches based on their distances
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the perspective transformation
        transformation_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        # Apply the perspective transformation to align the images
        aligned_image1 = cv2.warpPerspective(image1, transformation_matrix, (image1.shape[1], image1.shape[0]))
        aligned_image2 = cv2.warpPerspective(image2, transformation_matrix, (image2.shape[1], image2.shape[0]))

        return aligned_image1.squeeze(), aligned_image2.squeeze()
    
    def register_images(self, image1: np.array, image2: np.array, algo_type: str, kwargs: dict):
        '''Transform two images so that corresponding features in them will have similar spatial locations.'''

        algorithm = 'register_images_by_' + algo_type
        if hasattr(self, algorithm):
            algorithm_function = getattr(self, algorithm)
            if callable(algorithm_function):
                return algorithm_function(image1, image2, **kwargs)
            else:
                raise ValueError(f'Algorithm function is not callable.')
        else:
            raise ValueError(f"Unknown registration algorithm: {algo_type}.")

    def get_semantic_segmentation(self, img: np.array, algo_type: str):
        '''Perfrom semantic segmentation on an image.'''
            
        return img # Placeholder
    
    def process_images(self, inspected: np.array, reference: np.array):
        '''Preprocess two images in preparation for further analysis.'''
        
        inspected_processed = inspected.copy()
        reference_processed = reference.copy()

        for algorithm in self.cfg['preprocessor']['algorithms']:
            if algorithm not in ['normalization', 'registration', 'semantic_segmentation']:
                raise ValueError(f"Unknown processing algorithm: {algorithm}.")
            
            # Normalizing image intensities
            if algorithm == 'normalization':
                inspected_processed = self.normalize_image_intensity(
                    inspected_processed, self.cfg['preprocessor']['algorithms']['normalization']['type'])
                reference_processed = self.normalize_image_intensity(
                    reference_processed, self.cfg['preprocessor']['algorithms']['normalization']['type'])
            
            # Finding the overlapping area between the images
            if algorithm == 'registration':
                inspected_processed, reference_processed = self.register_images(
                    inspected_processed,
                    reference_processed,
                    self.cfg['preprocessor']['algorithms']['registration']['type'],
                    self.cfg['preprocessor']['algorithms']['registration']['kwargs'],
                )
                                 
            # Performing semantic segmentation on the images
            if algorithm == 'semantic_segmentation':
                inspected_processed = self.get_semantic_segmentation(
                    inspected_processed, self.cfg['preprocessor']['algorithms']['semantic_segmentation']['type'])
                reference_processed = self.get_semantic_segmentation(
                    reference_processed, self.cfg['preprocessor']['algorithms']['semantic_segmentation']['type']) 

        return inspected_processed, reference_processed