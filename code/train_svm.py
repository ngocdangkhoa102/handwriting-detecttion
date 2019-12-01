import cv2 as cv
import numpy as np

data = np.load('../user/data.npy')
labels = np.load('../user/labels.npy')
t_data = np.load('../user/t-data.npy')
t_labels = np.load('../user/t-labels.npy')

data = np.float32(data)
from sklearn import LinearSVC
clf = LinearSVC()
n = 5000
clf.fit(data[:n],labels[:n])

from joblib import dump
dump(clf,'../trained_model/svm_linear.joblib')