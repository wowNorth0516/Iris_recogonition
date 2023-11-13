import os
import cv2
import numpy as np

# Import all sub functions from other scripts
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from IrisEnhancement import IrisEnhancement
from IrisFeatureExtraction import IrisFeatureExtraction
from IrisMatching import IrisMatching
from IrisPerformanceEvaluation import IrisPerformanceEvaluation

# Get paths for all iris images
base_path = './CASIA Iris Image Database (version 1.0)'
path = []
# The dataset has 108 classes of iris
for i in range(1, 109):
    # The dataset has two sessions, one for training and the other for testing
    for j in range(1, 3):
        if j == 1:
            # 3 iris images for training in session 1
            for k in range(1, 4):
                path.append((f"{base_path}/{str(i).zfill(3)}/1/{str(i).zfill(3)}_1_{k}.bmp"))
        else:
            # 4 iris images for testing in session 2
            for k in range(1, 5):
                path.append((f"{base_path}/{str(i).zfill(3)}/2/{str(i).zfill(3)}_2_{k}.bmp"))

# Split train and test sets
train = []
test = []
# Get corresponding class number for both train and test sets
c_train = []
c_test = []
for p in path:
    # Get the class numbers(from 1 to 108) from the path
    c = int(p.split('/')[2])
    # Get the session number(1 or 2) from the path
    group = int(p.split('/')[3])
    image = cv2.imread(p)
    # Get the localized iris
    localized_iris = IrisLocalization(image) 
    # Get the normalized iris
    normalized_iris = IrisNormalization(localized_iris)
    # Get the enhanced iris
    enhanced_iris = IrisEnhancement(normalized_iris)
    # Extract features from the enhanced iris
    features = IrisFeatureExtraction(enhanced_iris)
    # Split features data into train and test sets
    if group == 1:
        train.append(features)
        c_train.append(c)
    else:
        test.append(features)
        c_test.append(c)
# Convert train, test, c_train and c_test from lists to NumPy arrays
train = np.array(train)
test = np.array(test)
c_train = np.array(c_train)
c_test = np.array(c_test)

# Calculate CRRs for both original and reduced feature sets
# Calculate FMRs and FNMRs with different threshold values
original_CRRs = IrisMatching(train, test, c_train, c_test, 1536) # Original feature vector with dimension 1536
reduced_CRRs, thresholds, FMRs, FNMRs = IrisMatching(train, test, c_train, c_test, 107) # Reduced feature vector with dimension 107

# The L2 distance measure has the highest CRR
# Compute CRRs of the L2 distance measure using features of different dimensions
dimensions = [1, 20, 40, 60, 80, 107]
CRRs = []
for i in range(5):
    CRR = IrisMatching(train, test, c_train, c_test, dimensions[i])[2] 
    CRRs.append(CRR)
CRRs.append(IrisMatching(train, test, c_train, c_test, dimensions[5])[0][2])
# Print the table of CRRs using different similarity measures
# Plot CRRs of the cosine similarity measure against different dimensions
# Print the table of FMRs and FNMRs with different threshold values
# Plot ROC curve
IrisPerformanceEvaluation(original_CRRs, reduced_CRRs, dimensions, CRRs, thresholds, FMRs, FNMRs)