import cv2
import numpy as np
#IrisFeatureExtraction.py: filtering the iris and extracting features;
def IrisFeatureExtraction(enhanced_iris):
    # Compute the kernel of the defined spatial filter
    def kernel(x, y, f, sigma_x, sigma_y):
        M = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
        G = M * (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(- (x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
        return G
    # Filter the iris image with the given spatial filter
    def filter_iris(ROI, f, sigma_x, sigma_y):
        height, width = ROI.shape
        # Calculate the spatial filter
        spatial_filter = np.array([[kernel(x - width // 2, y - height // 2, f, sigma_x, sigma_y) 
                                    for x in range(width)] 
                                   for y in range(height)])
        
        return cv2.filter2D(ROI, -1, spatial_filter)
    # Extract statistical features from 8x8 blocks
    def extract_features(filtered_irises):
        features = []
        for filtered_iris in filtered_irises:
            height, width = filtered_iris.shape
            block_size = 8
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    block = filtered_iris[y:y+block_size, x:x+block_size]
                    mean = np.mean(block)
                    var = np.var(block)
                    features.extend([mean, var])
        return np.array(features)
    # Scale the iris that contains useful information
    ROI = enhanced_iris[:48, :]
    # Set number of the frequency
    f = 3
    # Set number of sigma_x and sigma_y  for the two channels
    sigmas = [(5, 7), (8, 6)]
    # Filter the ROI using both channels
    filtered_irises = [filter_iris(ROI, f, sigma_x, sigma_y) for sigma_x, sigma_y in sigmas]
    # Extract and return features
    return extract_features(filtered_irises)