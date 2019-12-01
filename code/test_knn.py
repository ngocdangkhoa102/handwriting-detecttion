import cv2 as cv
import numpy as np

data = np.load('../user/data.npy')
labels = np.load('../user/labels.npy')
t_data = np.load('../user/t-data.npy')
t_labels = np.load('../user/t-labels.npy')

from joblib import load
neigh = load('../trained_model/knn.joblib')

data = np.float32(data)
n = 6000
print(neigh.predict(data[n:n+10]))
print(labels[n:n+10])





