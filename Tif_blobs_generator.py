import numpy as np
from tifffile import imwrite
import csv


def gauss(x, y, x0, y0, i, sigma_x, sigma_y, p):
    # (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - p ** 2))) * INSTEAD OF i
    return i*np.exp(-(((x - x0)/sigma_x) ** 2 + ((y - y0)/sigma_y) ** 2 - 2*p*((x - x0)*(y - y0) / (sigma_x*sigma_y)) ) / (2 * (1-p**2)))


spots = []
settings = {"Include Shot Noise": ''}

with open('plot.csv', 'r', newline='') as csvfile:
    csvFile = csv.reader(csvfile)
    next(csvFile)
    for row in csvFile:
        if row[-2] != '':
            settings[row[-2]] = row[-1]
        print(row[0:6])
        spots.append([float(i) for i in row[:6]])

# with open('spots.json', 'r') as file:
#     data = json.load(file)

image = np.ndarray(shape=(512, 512), dtype=np.uint8)

image[()] = np.arange(512)
image[:, :] = 0

for spot in spots:
    for i in range(512):
        for j in range(512):
            image[i, j] += gauss(i, j, spot[0], spot[1], spot[2], spot[3], spot[4], spot[5])

imwrite('output.tif', image)
    
if settings["Include Shot Noise"].lower() == "yes":
    image = np.random.poisson(lam=image, size=None)
    print(image.max())
    print(image.min())
    for i in image:
        print(i)
    image = image.astype(np.uint8)
    imwrite('output_with_noise.tif', image)