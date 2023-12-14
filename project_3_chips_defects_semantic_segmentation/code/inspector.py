import cv2
import numpy as np

from plots_utils import *

class Inspector():

    def __init__(self, cfg: dict) -> None:
        
        self.cfg = cfg

    def inspect_by_subtraction(self, image1: np.array, image2: np.array, **kwargs):
        '''Perform image subtraction between two grayscale images and apply thresholding.

        Args:
            image1 (numpy array): First grayscale image.
            image2 (numpy array): Second grayscale image.

        Returns:
            subtracted_binary_mask (numpy array): Binary mask resulting from thresholding.
        '''

        # Obtaining threshold value for binarization
        threshold = kwargs.get('threshold', 200)

        # Ensuring both images have the same shape
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same shape.")

        # Performing image subtraction
        subtracted_image = cv2.absdiff(image1, image2)
        
        # Applying thresholding
        _, subtracted_binary_mask = cv2.threshold(src=subtracted_image, thresh=threshold, maxval=1, type=cv2.THRESH_BINARY)

        return subtracted_binary_mask
    
    def calculate_correlation(self, window1, window2):
        '''Calculate the correlation between two matrices after standardization.'''
        
        # Normalize the images to have zero mean and unit variance
        window1 = (window1 - np.mean(window1)) / (np.std(window1) + 1e-5)
        window2 = (window2 - np.mean(window2)) / (np.std(window2) + 1e-5)
        
        # Calculate the correlation coefficient
        correlation = np.sum(window1 * window2)
        
        return correlation
    
    def calculate_mse(self, window1: np.array, window2: np.array):
        '''Calculate the MSE between two matrices.'''

        return np.mean((window1 - window2)**2)

    def inspect_by_similarity_in_window(self, image1: np.array, image2: np.array, **kwargs):
        '''Compare corresponding windows between two images and calculate a measure of their dissimilarity.'''

        # Initializing variables
        window_size, threshold = kwargs['window_size'], kwargs['threshold']

        # Getting the dimensions of the images
        height, width = image1.shape
        
        # Creating an empty binary mask
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Iterating through the image using a sliding window
        for y in range(0, height - window_size[0] + 1):
            for x in range(0, width - window_size[1] + 1):
                # Extract the window from both images
                window1 = image1[y:y+window_size[0], x:x+window_size[1]]
                window2 = image2[y:y+window_size[0], x:x+window_size[1]]
                
                # Calculating the dissimilarity
                dissimilarity = self.calculate_mse(window1, window2)
                
                # If the dissimilarity is above a threshold, mark the spot in the binary mask
                if dissimilarity > threshold:
                    binary_mask[y:y+window_size[0], x:x+window_size[1]] = 1
        
        return binary_mask

    def inspect_by_semantic_segmentation_hugging_face(self, inspected: np.array, reference: np.array, **kwargs):

        return np.random.choice([0, 1], size=inspected.shape) # Placeholder

    def inspect_by_cnn(self, inspected: np.array, reference: np.array, **kwargs):

        return np.random.choice([0, 1], size=inspected.shape) # Placeholder

    def remove_small_detections(self, matrix: np.array, ones_threshold: int, neighborhood_size: int) -> np.array:
        '''Return a new matrix where only the ones surrounded by at least the specified number of ones are kept.'''

        rows, cols = matrix.shape
        result = np.array([[0] * cols for _ in range(rows)])

        for i in range(neighborhood_size, rows - neighborhood_size):
            for j in range(neighborhood_size, cols - neighborhood_size):
                if matrix[i][j] == 1:
                    # Count the number of ones in the surrounding cells
                    count_ones = sum(matrix[x][y] for x in range(i-neighborhood_size, i+neighborhood_size+1)
                                        for y in range(j-neighborhood_size, j+neighborhood_size+1)) - matrix[i][j]

                    # Check if the count exceeds the threshold
                    if count_ones >= ones_threshold:
                        result[i][j] = 1

        return result

    def inspect_images(self, inspected: np.array, reference: np.array) -> np.array:
        '''Find defects in an inspected image, which contains a chip scan, by comparing it to a reference image.'''

        # Initializing variables
        defects_masks = []
        weights = []

        # Going over different inspection algorithms
        for algorithm, parameters in self.cfg['inspector']['algorithms'].items():
            algorithm_function_name = 'inspect_by_' + algorithm
            if hasattr(self, algorithm_function_name):
                algorithm_function = getattr(self, algorithm_function_name)
                if not callable(algorithm_function):
                    raise ValueError(f'Algorithm function is not callable.')
            else:
                raise ValueError(f'Unknown algorithm: {algorithm}.')
            result = algorithm_function(inspected, reference, **parameters['kwargs'])
            # Saving result
            defects_masks.append(result)
            weights.append(parameters['weight'])
            # Plotting result
            if self.cfg['monitoring']['plot_images']:
                print(f'Plotting result for algorithm "{algorithm}":')
                plot_images_and_result(inspected, reference, result)

        # Unifying defects from all algorithms
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        unified_defects_mask = np.zeros_like(defects_masks[0], dtype=float)
        for matrix, weight in zip(defects_masks, normalized_weights):
            unified_defects_mask += matrix * weight

        # Limiting the max value of the result
        unified_defects_mask[unified_defects_mask > 1] = 1

        # Removing defects that don't pass a predefined detection confidence threshold
        unified_defects_mask[unified_defects_mask < self.cfg['inspector']['unified_defects_mask_threshold']] = 0

        # Removing defects that are too small
        unified_defects_mask = self.remove_small_detections(
            matrix=unified_defects_mask,
            ones_threshold=self.cfg['inspector']['small_defects_ones_threshold'],
            neighborhood_size=self.cfg['inspector']['small_defects_neighborhood_size']
        )

        return unified_defects_mask