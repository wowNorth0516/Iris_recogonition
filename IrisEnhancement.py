import cv2
# IrisEnhancement.py: enhancing the normalized iris;
def IrisEnhancement(normalized_iris):
    # Approximate intensity variations
    # Compute the mean for each 25x25 block
    bgd_ill = cv2.resize(
        cv2.blur(normalized_iris, (25, 25)), 
        (normalized_iris.shape[1], normalized_iris.shape[0]), 
        interpolation = cv2.INTER_CUBIC)
    # Subtract the estimated background
    cps_img = cv2.subtract(normalized_iris, bgd_ill)
    # Histogram equalization in each 32x32 region
    clahe = cv2.createCLAHE(clipLimit = 50, tileGridSize = (32,32))
    clahe_iris = clahe.apply(cps_img)
    # Apply Bilateral Filter for noise reduction
    enhanced_iris = cv2.bilateralFilter(clahe_iris, d = 5, sigmaColor = 75, sigmaSpace = 75)
    return enhanced_iris