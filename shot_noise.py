import cv2
from numpy.fft import fft2, fftshift, ifftshift
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('output.tif')
# print(np.unravel_index(np.argmax(image, axis=None), image.shape))

imageplusshotnoise = np.random.poisson(lam=image, size=None)
plt.figure(figsize=(10, 5))
plt.imshow(imageplusshotnoise)
plt.show()
