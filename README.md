# Computer vision portfolio
Projects portfolio in computer vision.

## (1) Localizing "bumps" in particle physics experimental results
The goal of this project is to identify "bumps" in particle colliders experimental results, which may hint on new particles.

It was published in the European Physics Journal C: https://doi.org/10.1140/epjc/s10052-022-10215-1

Abstract: We propose a data-directed paradigm (DDP) to search for new physics. Focusing on the data without using simulations, exclusive selections which exhibit significant deviations from known properties of the standard model can be identified efficiently and marked for further study. Different properties can be exploited with the DDP. Here, the paradigm is demonstrated by combining the promising potential of neural networks (NN) with the common bump-hunting approach. Using the NN, the resource-consuming tasks of background and systematic uncertainty estimation are avoided, allowing rapid testing of many final states with only a minor degradation in the sensitivity to bumps relative to standard analysis methods.

## (2) Noisy digits localization & classification
The goal of this project is to design and evaluate a deep neural network for (1) localizing and (2) classifying handwritten digits in images, where the digits aren't necessarily centered inside the images, and different noise types could be applied to them.

The stages of the project are as follows:

1) Data preparation - Loading a handwritten digits dataset (the MNIST dataset) and expanding the images, such that the digits are placed randomly inside a larger image.

2) Model development - Designing and training the deep learning model, and explaining the reasoning behind the different development decisions.

3) Performance evaluation - Evaluating the performance of the model in the localization and classification tasks, especially under the application of different noise types. Then suggesting performance improvement strategies.

4) Model deployment - Planning how to deploy the model at the front-end and the back-end, and how to deal with potential congestion issues.

## (3) Chips defects semantic segmentation
The goal of this project is to create a binary detection mask for defects found in chip manufacturing images.

Minimal assumptions are made on the properties of the possible defect types to create the most generic solution, and false detections are minimized by comparing the results of a subtraction-based algorithm and a window-similarity-based algorithm. 

### Assumptions
- The images have a single color channel (greyscale) with values in the range [0,255].
- The images can have varying dimensions.
- The images can have varying intensities.
- The images can be shifted relative to one another, but up to a limit we should know about.
- Any number of defects can be present.
- The defects can have any shape.
- The defects aren’t at the extreme edges of the chips, which might not appear in both of the images.
- Defects can appear both as brighter and darker regions.

### Methods
The inspection pipeline comprises the following steps:

Preprocessing
- Normalization - The pixel values are transformed into a certain range with a selectable method. E.g. the images can be transformed into the range [0,255] by dividing them by their maximum and multiplying by 255. This helps analyzing images with varying intensities. Transforming can also be done into the range [0,1], by dividing them by the max value, which also helps analyzing different intensities and helps input them into neural nets.
- Image registration - The overlapping area between the images is found, such that corresponding features in them will have similar spatial locations. Different methods can be selected, e.g. we can use a sliding window to go over all of the submatrices in both images and find the most similar submatrices to return the cropped images. Similarity can be defined in terms of the correlation or the MSE distance between the elements. Due to the cropping, we may miss defects which are located at the extreme edges of the chips.

Inspecting
- Defects detection - Defects are found in the image intended for inspection based on the reference image by multiple methods, each returning a binary detection mask where “ones” represent defects:
  * Subtraction-based inspection - The pixel values of the images are subtracted from each other and an absolute value is applied. The difference between dissimilar areas is large, so thresholding is applied to find the most dissimilar pixels.
  * Window similarity inspection - Corresponding windows are compared between the images and a measure of dissimilarity is calculated, to not only consider individual pixels.
- Defects unification - The defects found via all of the algorithms are unified, by performing a weighted average of the binary detection masks. The weights are a measure of our confidence in each algorithm.
- Defects verification by confidence - Defects that don't pass a predefined detection confidence threshold are removed.
- Defects verification by size - Defects that are too small are removed, i.e., there aren’t enough “ones” in their neighborhood in the binary mask.

## (4) Handwritten digits classification tutorial
The goal of this project is to teach how to train and evaluate a deep neural network for classifying handwritten digits in 10 easy steps, for educational purposes.

The steps are: 1) Import libraries, 2) Choose settings, 3) Load training & testing data, 4) Define useful variables, 5) Explore data, 6) Process data, 7) Define the neural network, 8) Train the model, 9) Evaluate model's performance, and 10) Inspect model's mistakes.
