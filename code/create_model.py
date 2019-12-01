import cv2 as cv
import numpy as np

data = np.load('../user/data.npy')
labels = np.load('../user/labels.npy')

from sklearn import svm
clf = svm.SVC(gamma = 0.001, C = 100.)
clf.fit(data[:2000],labels[:2000])

from joblib import dump
name = input('file name is: ')
dump(clf,'../trained_model/'+str(name)+'.joblib')

