# Iris_recogonition
reproduce paper : Personal Identification Based on Iris Texture Analysis LiMa 2003

IrisLocalization.py:
This script is for detecting the pupil's center, radius, and the iris's outer boundary for an image. It takes an image as the input and return the centers and radius of the detected pupil and iris circles, and ROI (resized image). The specific steps are:
Converts the input image to grayscale.  
Uses Gaussian blur to reduce noise and improve contour detection.  
Uses binary thresholding to identify the pupil (the darkest area in the image).  
Resizes ROI based on the approximate center of the pupil.  
Applies adaptive thresholding and morphological operations to refine the pupil's contour.  Employs Hough Circle Transform to precisely detect the pupil and iris boundaries.  
Selects the most appropriate circles for the pupil and iris based on area comparison and proximity to the expected centers.  
Ensures the detected iris boundary does not exceed the image frame and adjusts the radius if necessary. 

IrisNormaliztion.py:
This script performs normalization of the iris region by unwrapping the circular pattern into a rectangular block. The function takes a localized iris as the input and returned the normalized iris. The specific steps are:
Initializes an array to hold the normalized iris pixels. 
Iterates over each pixel in the normalized image’s array. 
For each pixel position, it calculates the corresponding polar coordinates. 
Performs a linear interpolation between the boundaries of the pupil and iris in the direction of the calculated angle. 
Maps the pixels from ROI to the normalized rectangular block based on the interpolated coordinates.

IrisEnhancement.py:
This script is designed to enhance the visual features of the normalized iris image for better recognition. The function takes a normalized iris as the input and return the enhanced iris.
It firstly estimates and compensates for varying illumination conditions within the iris image, which is done by blurring it with a 25*25 kernel to approximate the background illumination and then resizing it back to the original dimensions. The estimated background is then subtracted from the normalized iris. By increasing the kernel size, the CRR increase.
At the end, using CLAHE with a clip limit of 50 and a tile grid size of 32*32, which can improve the local contrast and enhance the iris patterns.


Limitations:

IrisLocalization.py: its hard to select perfect iris contour for all image. For some images, the selected area for iris is too big/samll which may lead impresice compuatation for future steps. Same for pupil, the selected area my cover small part of iris. To improve this, use draw eclipse may help.

IrisEnhancement.py:
Using CLAHE to enhance the image contrast, but it may also enhance noise and affect the feature extraction. For improvement, I tried to apply Bilateral Filter to reduce the noise while not sabotaging iris patterns, but it didn't improve the accuracy significantly. 
