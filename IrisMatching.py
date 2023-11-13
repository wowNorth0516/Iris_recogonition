import numpy as np
from scipy.linalg import eigh
# IrisMatching.py: using Fisher linear discriminant for dimension reduction and nearest center classifier for classification
def IrisMatching(train, test, c_train, c_test, dim):
    # Define a function to use Fisher linear discriminant and get the ptojected matrix W
    def FLD_W(train_data, c_num, dim):
        # Initialize S_W and S_B
        S_W = np.zeros((train_data.shape[1], train_data.shape[1]))
        S_B = np.zeros((train_data.shape[1], train_data.shape[1]))
        # Calculate the overall mean
        mean = np.mean(train_data, axis = 0)
        # Get S_W and S_B
        for i in range(1, 109):
            x_i = train_data[c_num == i]
            mean_i = np.mean(x_i, axis = 0)
            mean_diff = (mean_i - mean).reshape(-1, 1)
            S_B += len(x_i) * mean_diff @ mean_diff.T
            S_W += np.cov(x_i, rowvar = False) * (len(x_i) - 1)
        # Use a small reg_factor to S_W
        reg_factor = 1e-6
        S_W += np.eye(S_W.shape[0]) * reg_factor
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(S_B, S_W)
        # Sort the eigenvectors based on descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Select the top eigenvectors of value of the dimension
        W = eigenvectors[:, :dim]
        # Return the projected matrix W
        return W

    # Define a funtion to use Nearest Center Classifier to get the classification results for all test cases
    def NCC(train_data, test_data, c_num, measure):
        predictions = []
        # Get the average value of feature vector of each class
        centers = {}
        for c in range(1, 109):
            centers[c] = np.mean(train_data[c_num == c], axis = 0)
        # For each test sample, find the nearest center using specified distance measure
        for sample in test_data:
            # Using the three measures specified by the paper
            if measure == 'd1':
                dist = {c: np.sum(np.abs(sample - center)) for c, center in centers.items()}
            elif measure == 'd2':
                dist = {c: np.linalg.norm(sample - center) ** 2 for c, center in centers.items()}
            elif measure == 'd3':
                dist = {c: 1 - np.dot(sample, center) / (np.linalg.norm(sample) * np.linalg.norm(center)) 
                        for c, center in centers.items()}
            # Get the classification result for each test case
            predictions.append(min(dist, key = dist.get)) 
        return predictions
    # Use reduced feature vector if dimension is less than 108
    W = FLD_W(train, c_train, dim)
    proj_train = train @ W
    proj_test = test @ W
    if dim < 108:
        results = []
        for measure in ['d1', 'd2', 'd3']:
            predictions = NCC(proj_train, proj_test, c_train, measure)
            results.append(predictions)
    # Use original feature vector if dimension is larger than 107
    else:
        results = []
        for measure in ['d1', 'd2', 'd3']:
            predictions = NCC(train, test, c_train, measure)
            results.append(predictions)
    # Calculate CRRs of three measures using the classification results
    CRRs = []
    for prediction in results:
        CRR = np.mean(np.array(prediction) == c_test) * 100
        CRRs.append(CRR)
    # Set threshold to compute FMR and FNMR
    thresholds = [0.013, 0.016, 0.019, 0.025]
    # Get the average value of reduced feature vector of each class
    centers = {}
    for c in range(1, 109):
        centers[c] = np.mean(proj_train[c_train == c], axis = 0)
    predictions = []
    min_dist = []
    for sample in proj_test:
        # Use the cosine similarity measure to calculate FMR and FNMR
        dist = {c: 1 - np.dot(sample, center) / (np.linalg.norm(sample) * np.linalg.norm(center)) 
                for c, center in centers.items()}
        # Get the classification result for each test case
        predictions.append(min(dist, key = dist.get)) 
        # Get the computed cosine similarity measure for each test case
        min_dist.append(min(dist.values()))
    # Return FMR and FNMR only when we use reduced feature vector of dimensions 107
    if dim == 107:      
        fmrs = []
        fnmrs = []
        for threshold in thresholds:
            # Number of false-matching cases
            fmr_num = 0
            # Number of false-non-matching cases
            fnmr_num = 0
            # Number of invalid cases
            invalid = 0
            # Number of valid cases
            valid = 0
            # Run over each test case
            for i in range(len(c_test)):
                # Set a boolean for correct classification
                is_match = predictions[i] == c_test[i]
                # If the computed similarity measure is higher than the threshold, the test case is valid
                score = min_dist[i]
                if score > threshold:
                    valid += 1
                    # If the test case is correctly classified, the number of FMR cases is increased by 1
                    if is_match:
                        fmr_num += 1
                # Otherwise, the test case is invalid
                if score <= threshold:
                    invalid += 1
                    # If the test case is wrongly classified, the number of FNMR cases is increased by 1
                    if not is_match:
                        fnmr_num += 1
            # Calculate FMR and FNMR
            if valid == 0:
                fmr = 0
            else:
                fmr = fmr_num / valid 
            if invalid == 0:
                fnmr = 0
            else:
                fnmr = fnmr_num / invalid 
            fmrs.append(fmr)
            fnmrs.append(fnmr)
        return CRRs, thresholds, fmrs, fnmrs
    # Return only CRRs when we use other dimensions
    else:
        return CRRs