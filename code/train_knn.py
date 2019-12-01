import cv2 as cv
import numpy as np

data = np.load('../user/data.npy')
labels = np.load('../user/labels.npy')
t_data = np.load('../user/t-data.npy')
t_labels = np.load('../user/t-labels.npy')

data = np.float32(data)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
n = 5000
neigh.fit(data[:n],labels[:n])

from joblib import dump
dump(neigh,'../trained_model/knn.joblib')