import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def lg_filter(sd=1.6):
    total = 0
    k_size = 1 + 2*math.ceil(3*sd)
    a = 1/(2*math.pi*sd*sd)
    g_filter = np.zeros([k_size, k_size])
    for i in range(k_size):
        for j in range(k_size):
            g_filter[i, j] = a*math.e**(-((i-int(k_size/2))**2+(j-int(k_size/2))**2)/(2*sd**2))
            total += g_filter[i, j]
    b = total/(k_size*k_size)
    lg_filter = g_filter - b
    return lg_filter

image = cv2.imread("output.tif")
filtered = cv2.filter2D(image, -1, lg_filter())

plt.imshow(filtered)
plt.show()
# print(lg_filter())
