import matplotlib.pyplot as plt
import numpy as np
# IrisNormalization.py: mapping the iris from Cartesian coordinates to polar coordinates;
def IrisNormalization(localized_iris):
    pupil_circle, iris_circle, roi = localized_iris
    # Extract pupil and iris center coordinates and radii
    cx_p, cy_p, r_p = pupil_circle
    cx_i, cy_i, r_i = iris_circle
    # Unwrap the iris circle to a rectangular block of size M x N.
    M = 64
    N = 512
    # Initialize the normalized iris
    normalized_iris = np.zeros((M, N), dtype = np.uint8)
    # Iterate over each pixel position in the normalized image
    for i in range(M):
        for j in range(N):
            # Calculate the angle in radians for the current position
            theta = 2 * np.pi * j / N
            # Compute x and y for the pupil boundary in the direction of theta
            x_p = cx_p + r_p * np.cos(theta)
            y_p = cy_p + r_p * np.sin(theta)
            # Compute x and y for the iris boundary in the direction of theta
            x_i = cx_i + r_i * np.cos(theta)
            y_i = cy_i + r_i * np.sin(theta)
            # Linearly interpolate between the pupil and iris boundaries based on i
            x = x_p + (x_i - x_p) * (i / M)
            y = y_p + (y_i - y_p) * (i / M)
            # Check for out-of-bound indices
            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                normalized_iris[i][j] = roi[int(y)][int(x)]
    return normalized_iris