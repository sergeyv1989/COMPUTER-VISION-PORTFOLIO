from inspector import Inspector
from files_utils import *
from plots_utils import *
from preprocessor import Preprocessor

def main(cfg_file: str):
    '''Load pairs of images for inspection and reference, process them and find defects in the image for inspection.'''

    # Loading configurations
    cfg = load_yaml(cfg_file)
    processed_images_folder = cfg['preprocessor']['processed_images_folder'] # Folder for storing processed images
    results_folder = cfg['inspector']['results_folder'] # Folder for storing result images
    save_results = cfg['inspector']['save_results'] # Whether to save inspection results
    plot_images = cfg['monitoring']['plot_images'] # Whether to plot the images

    # Initializing
    preprocessor = Preprocessor(cfg) # Class for preprocessing the image pairs before inspection
    inspector = Inspector(cfg) # Class for comparing the image pairs
    defects_masks = {} # Will contain binary arrays with inspection results per test case
    if not os.path.exists(processed_images_folder): # Creating folder for storing processed images
        os.makedirs(processed_images_folder)
        print(f'Created processed images folder in: {processed_images_folder}')
    if not os.path.exists(results_folder): # Creating folder for storing result images
        os.makedirs(results_folder)
        print(f'Created results folder in: {results_folder}')

    # Finding image pairs file paths
    image_pairs = get_image_pairs_files(cfg['data']['images_folders'])

    for case_id in list(image_pairs.keys()):
        
        # Loading images
        print('Loading images.')
        inspected = load_tif_image(image_pairs[case_id]['inspected'])
        reference = load_tif_image(image_pairs[case_id]['reference'])

        # Plotting loaded images
        if plot_images:
            print(f'Plotting loaded images for "{case_id}":')
            plot_images_pair(inspected, reference)

        # Preprocessing images
        print('Preprocessing images.')
        if cfg['preprocessor']['process_images']:
            inspected_processed, reference_processed = preprocessor.process_images(inspected, reference)
            save_tif_image(inspected_processed, processed_images_folder + f'/{case_id}_inspected_processed.tif')
            save_tif_image(reference_processed, processed_images_folder + f'/{case_id}_reference_processed.tif')
        else: # Loading processed images
            inspected_processed = load_tif_image(processed_images_folder + f'/{case_id}_inspected_processed.tif')
            reference_processed = load_tif_image(processed_images_folder + f'/{case_id}_reference_processed.tif')

        # Plotting processed images
        if plot_images:
            print(f'Plotting preprocessed images for "{case_id}":')
            plot_images_pair(inspected_processed, reference_processed)

        # Finding defects in the inspected image
        print('Inspecting images.')
        defects_mask = inspector.inspect_images(inspected_processed, reference_processed)

        # Plotting results
        if plot_images:
            print(f'Plotting inspection results for "{case_id}":')
            plot_images_and_result(inspected_processed, reference_processed, defects_mask)

        # Saving results
        defects_masks[case_id] = defects_mask
        if save_results:
            save_tif_image(255 * defects_mask, results_folder + f'/{case_id}_result.tif') # Converting 1 to 255
        
        print('\n' + '#'*100)
        print('#'*100)
        print('#'*100 + '\n')

    return defects_masks

if __name__ == '__main__':

    defects_masks = main(cfg_file="C:\Sergey's Google Drive\Workspace\chips-defects-semantic-segmentation\code\cfg.yaml")