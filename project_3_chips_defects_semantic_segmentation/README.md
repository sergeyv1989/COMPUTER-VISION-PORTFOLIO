# CHIPS DEFECTS SEMANTIC SEGMENTATION

## Assumptions
- The images have a single color channel (greyscale) with values in the range [0,255].
- The images can have varying dimensions.
- The images can have varying intensities.
- The images can be shifted relative to one another, but up to a limit we should know about.
- Any number of defects can be present.
- The defects can have any shape.
- The defects aren’t at the extreme edges of the chips, which might not appear in both of the images.
- Defects can appear both as brighter and darker regions.

## Methods
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