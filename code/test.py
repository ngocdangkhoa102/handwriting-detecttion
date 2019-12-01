import cv2 as cv
import numpy as np 

data = np.load('../user/data.npy')
labels = np.load('../user/labels.npy')

from joblib import load
mlp = load('../trained_model/hwd_svm.joblib')

n = 3000
X = np.float32(data)
print(mlp.predict(X[n:n+10]))
print(labels[n:n+10])