import math
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

max_size = 100 # needs calibartion
radius = 3.5

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


def find_spots(image, image_unit16depth,  title):
    # plt.figure(figsize=(10, 5))

    # fig, ax = plt.subplots()

    inverted = cv2.bitwise_not(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marked_image = image

    # plt.imshow(grayscale, cmap='gray')
    # plt.imshow(covid_probe, cmap='gray')

    plt.title(title)

    mean_val = np.mean(grayscale)
    std_val = np.std(grayscale)
    # print(title)
    # print(mean_val)
    # print(std_val)
    # print(grayscale.shape)
    # print()
    threshold_value = mean_val + std_val*1.6
    # print(threshold_value)
    # Adjust factor if needed

    # Apply threshold to find bright spots
    binary_mask = (grayscale >= threshold_value).astype(np.uint8) * 255  # Convert to binary (0 or 255)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # print(num_labels, labels, stats, centroids)

    # Visualization (keeping original pixel values)
    output_img = np.stack([grayscale] * 3, axis=-1)  # Convert to 3-channel for drawing

    # Draw bounding boxes and centroids

    spots_coordinates = []

    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        # cv2.rectangle(inverted, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Bounding box (green)
        rect = patches.Rectangle((x, y), w, h, edgecolor='g', facecolor='none')
        # ax.add_patch(rect)
        if radius < int(cx) < 512-radius and radius < int(cy) < 512-radius:
            if  area <= max_size:
                intensity=0
                for i in range(-int(radius), int(radius)+1):
                    for j in range(-int(radius), int(radius)+1):
                        if i*i + j*j <= radius*radius:
                            marked_image[int(cx+i)][int(cy+j)] = [0, 255, 0]
                            intensity += image_unit16depth[int(cx+i)][int(cy+j)]
                ## FOR SCATERING
                # plt.imshow(marked_image)
                spots_coordinates.append([cx, cy, area, intensity])

    # print(spots_coordinates)
    return spots_coordinates

max_probe_intensity = 0
probe_intensities = []

for file in os.listdir("Probe"):
    if file.endswith(".tif"):
        image = cv2.imread("Probe/"+file)
        image_unit16depth = cv2.imread("Probe/"+file, cv2.IMREAD_ANYDEPTH)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.filter2D(image, -1, lg_filter())
        data = find_spots(filtered, image_unit16depth, "Lowered Gaussian")
        print(file)
        for single_spot in data:
            intensity = single_spot[3]
            grayscale_intensity = grayscale[int(single_spot[1])][int(single_spot[0])]
            # coordinates area accessed as y, x. Pasas these to the gray scale iamge to get
            # that pixel value
            probe_intensities.append(intensity)
            # print(intensity)
            # print(coordinate)
            if  intensity > max_probe_intensity:
                max_probe_intensity = intensity
                cv2.imwrite("unit16.tif", image_unit16depth)
                print(file)
                print(int(single_spot[0]), int(single_spot[1]), int(single_spot[2]))
                print(max_probe_intensity, end='\n\n')

plt.title("Probe Intensities")
plt.hist(probe_intensities, int(len(probe_intensities)/10))
plt.show()

values = []
sample_intensities = []
count = 0

for file in os.listdir("Sample"):
    if file.endswith(".tif"):
        image = cv2.imread("Sample/"+file)
        image_unit16depth = cv2.imread("Sample/"+file, cv2.IMREAD_ANYDEPTH)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.filter2D(image, -1, lg_filter())
        data = find_spots(filtered, image_unit16depth, "Lowered Gaussian")
        for single_spot in data:
            intensity = single_spot[3]
            grayscale_intensity = grayscale[int(single_spot[1])][int(single_spot[0])]
            # print(str(int(coordinate[0]))+","+str(int(coordinate[1])))
            sample_intensities.append(intensity)
            if intensity > max_probe_intensity:
                count+=1
                values.append(intensity/max_probe_intensity)


print("count is ", count)
# for i in find_spots(filtered, "Lowered Gaussian"):  # Skip background (label 0)
#     plt.scatter(i[0], i[1], 5, c="r")
plt.title("Sample Intensities")
plt.hist(sample_intensities, int(len(sample_intensities)/10))
plt.show()

plt.title("Ratio")
plt.hist(values)
plt.show()

# print(lg_filter())
