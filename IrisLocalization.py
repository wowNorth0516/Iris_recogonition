import cv2
import numpy as np
import matplotlib.pyplot as plt
# IrisLocalization.py: detecting pupil and outer boundary of iris
def IrisLocalization(image):
    def compute_circle_area(radius): # compare selected area
        return np.pi * (radius ** 2)
    #convert the image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #locate darkest area(pupil) in original image for resize the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    value, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # two way to get resize image
    if contours and cv2.moments(max(contours, key=cv2.contourArea))["m00"]>200:
        pupil_origin = max(contours, key=cv2.contourArea)
        M = cv2.moments(pupil_origin)
        cx_pupil = int(M['m10']/M['m00'])
        cy_pupil = int(M['m01']/M['m00']) # approximate the center, not precise because it will incluse noise to pupil contour
        #resize the image
        roi_size = 120
        roi = gray[max(0,cy_pupil-roi_size):cy_pupil+roi_size,max(0,cx_pupil-roi_size):cx_pupil+roi_size] 
    else:
        #the essay's method to find center of eyes
        vertical_projection = np.sum(gray, axis=1)
        horizontal_projection = np.sum(gray, axis=0)
        y = np.argmin(vertical_projection)
        x = np.argmin(horizontal_projection)
        roi_size = 150
        roi = gray[max(0,y-roi_size):y+roi_size,max(0,x-roi_size):x+roi_size]
    
    #find the precise contour for pupil
    blurred1 = cv2.GaussianBlur(roi, (5, 5), 0)
    #Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2) # parameter choosen from documentation example
    #Morphological opening to remove eyelashes
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6)) # to select target area instead of select all : ways to ignore noise
    opened1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)

    #different setting for finding pupil, select the one with largest area
    pupils1 = cv2.HoughCircles(opened1, cv2.HOUGH_GRADIENT, dp=1.5, minDist=10000, param1=20, param2=1, minRadius=0, maxRadius=60)
    pupils2 = cv2.HoughCircles(opened1, cv2.HOUGH_GRADIENT, dp=2, minDist=10000, param1=50, param2=30, minRadius=0, maxRadius=60)
    if pupils1 is not None:
        pupils1 = np.round(pupils1[0, :]).astype("int")
    else:
        pupils1 = [[0,0,0]]
    if pupils2 is not None:
        pupils2 = np.round(pupils2[0, :]).astype("int")
    else:
        pupils2 = [[0,0,0]]
    # Compute areas for two selected pupils
    area1 = compute_circle_area(pupils1[0][2])
    area2 = compute_circle_area(pupils2[0][2])
    # Compare areas and select the largest one
    if area1 > area2:
        selected_pupil = pupils1
    else:
        selected_pupil = pupils2
    for (x, y, r) in selected_pupil:
        pupil_x,pupil_y,pupil_r = x,y,r
        cv2.circle(roi, (pupil_x, pupil_y), pupil_r, (0, 255, 0), 2)
    
    #find the precise contour for iris
    blurred2 = cv2.medianBlur(roi, 5)
    #Adaptive thresholding
    thresh2 = cv2.adaptiveThreshold(blurred2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    opened2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel2)

    # knowing pupil's radius from previous circle, make the boundary for iris radius be [pupil_r*2, pupil_r*10], avoiding select pupil again
    # #find the outer circles - iris : 1
    iriss = cv2.HoughCircles(opened2, cv2.HOUGH_GRADIENT, 1, 1000,
                                param1 = 150, param2 = 1, minRadius = pupil_r*2, maxRadius = pupil_r*10 )
    iriss = np.round(iriss[0, :]).astype("int")
    iris_x1, iris_y1, iris_r1 = iriss[0]

    #find the outer circles - iris : 2
    circles2 = cv2.HoughCircles(opened2, cv2.HOUGH_GRADIENT, 2, 1000,
                        param1 = 20, param2 = 1, minRadius = 0, maxRadius = pupil_r*10 )
    circles2 = np.round(circles2[0, :]).astype("int")
    iris_x2, iris_y2, iris_r2 = circles2[0]
    #compare iris1 and iris2, select the one the center is more close to pupil with larger area
    if compute_circle_area(iris_r1) < compute_circle_area(iris_r2) and abs(iris_x1-pupil_x) + abs(iris_y1 - pupil_y) > abs(iris_x2-pupil_x) + abs(iris_y2 - pupil_y):
        iris_x, iris_y, iris_r = circles2[0]
    else: 
        iris_x, iris_y, iris_r = iriss[0]
        
    #check bound of selected iris area
    #if out of boundary, update x and y
    if iris_x - iris_r < 0:
        iris_x = iris_x + abs(iris_x - iris_r)
    elif iris_x + iris_r > roi.shape[1]:
        iris_x = iris_x - abs(iris_x - iris_r)
    if iris_y - pupil_x < 0:
        iris_y = iris_y + abs(iris_y - pupil_x)
    elif iris_y + pupil_x > roi.shape[0]:
        iris_y = iris_y - abs(iris_y - pupil_x)
    #check if circled iris is too small
    if iris_r < pupil_r *1.5:
        iris_r = int(pupil_r*2)
    if iris_x - iris_r < 1:
        iris_r -= (abs(iris_x - iris_r))
    return (pupil_x, pupil_y, pupil_r), (iris_x, iris_y, iris_r), roi
