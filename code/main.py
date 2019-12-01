import cv2 as cv
import numpy as np

def Display(dat,la):
	for number in dat:
		tmp = np.reshape(number,(28,28))
		tmp = cv.resize(tmp,(100,100))
		cv.imshow('number',tmp)
		k = cv.waitKey(0) & 0xFF
		if k == 27:
			break

from joblib import load
neigh = load('../trained_model/knn.joblib')

# Read the input image 
im = cv.imread("../numbers.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_gray = cv.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv.threshold(im_gray, 90, 255, cv.THRESH_BINARY_INV)

# Find contours in the image
im2, ctrs, hier = cv.findContours(im_th.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv.boundingRect(ctr) for ctr in ctrs]
# for rect in rects:
# 	cv.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
# 	cv.imshow('',im)
# 	cv.waitKey(0)

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
im_out = np.zeros((8,28*28))
index = 0
for rect in rects:
    # Draw the rectangles
    cv.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    # leng = int(rect[3] * 1.6)
    # pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    # pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    # roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    (x,y,w,h) = rect
    x = x - 22
    y = y - 22
    w = w + 44
    h = h + 44
    roi = np.float32(im_th[y:y+h,x:x+w])
    # Resize the image
    roi = cv.resize(roi, (28, 28), interpolation=cv.INTER_AREA)
    roi = cv.dilate(roi, (5, 5))
    cv.imshow('number',roi)
    cv.waitKey(0)
    # Calculate the HOG features
    roi = np.reshape(roi,(28*28))
    im_out[index] = roi
    index = index + 1
    # im_out = np.concatenate((im_out, roi), axis=0)
    nbr = neigh.predict([roi])
    print(nbr)
    cv.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    
np.save('numbers.npy',im_out)

cv.imshow("Resulting Image with Rectangular ROIs", im)
cv.waitKey()





