import matplotlib.pyplot as plt
import numpy as np

def plot_images_pair(inspected: np.array, reference: np.array):
    '''Display an inspected image and a reference image side by side.'''

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    plt.subplot(121)
    plt.imshow(inspected, cmap='gray')
    plt.title('Inspected')

    plt.subplot(122)
    plt.imshow(reference, cmap='gray')
    plt.title('Reference')
    plt.show()

def plot_images_and_result(inspected: np.array, reference: np.array, result: np.array):
    '''Display an inspected image, reference image, and the inspection result side by side.'''

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))

    plt.subplot(131)
    plt.imshow(inspected, cmap='gray')
    plt.title('Inspected')

    plt.subplot(132)
    plt.imshow(reference, cmap='gray')
    plt.title('Reference')

    plt.subplot(133)
    plt.imshow(result, cmap='gray')
    plt.title('Result')

    plt.show()