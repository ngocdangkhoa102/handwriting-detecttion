import cv2 as cv
import numpy as np
from mlxtend.data import loadlocal_mnist

X, y = loadlocal_mnist(
	images_path = 't10k-images.idx3-ubyte',
	labels_path = 't10k-labels.idx1-ubyte' )

np.save('../user/t-data.npy',X)
np.save('../user/t-labels.npy',y)

X, y = loadlocal_mnist(
	images_path = 'train-images.idx3-ubyte',
	labels_path = 'train-labels.idx1-ubyte' )

np.save('../user/data.npy',X)
np.save('../user/labels.npy',y)