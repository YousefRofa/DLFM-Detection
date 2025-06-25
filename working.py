import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog
from tkinter import *
import os

# root = Tk()
# root.withdraw()
# Probe_folder_root = filedialog.askdirectory(initialdir="/",title="Select Probe Folder")
# Covid_folder_root = filedialog.askdirectory(initialdir="/",title="Select Covid+Probe Folder")
#
# Probe_files = os.listdir(Probe_folder_root)
# Covid_Probe_files = os.listdir(Covid_folder_root)

# files_array = [[Probe_files[i], Covid_Probe_files[i]] for i in range(len(Probe_files))]

Probe_src = r"D:\Uni\Research\Prof. George Shubeita\DNA_ImgDet\control.tif"
Covid_Probe_src = r"D:\Uni\Research\Prof. George Shubeita\DNA_ImgDet\covid.tif"

max_size = 20# This sizing needs calibration

probe = cv2.imread(Probe_src, cv2.IMREAD_COLOR)

probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
filtered_probe = cv2.Laplacian(probe_gray, cv2.CV_16S, ksize=9)  # CV_16S is 16 bits image signed

covid_probe = cv2.imread(Covid_Probe_src, cv2.IMREAD_COLOR)
covid_probe_gray = cv2.cvtColor(covid_probe, cv2.COLOR_BGR2GRAY)
filtered_covid_probe = cv2.Laplacian(covid_probe_gray, cv2.CV_16S, ksize=9)  # CV_16S is 16 bits image signed

# probe_gray = cv2.resize(probe_gray, (0, 0), fx=0.5, fy=0.5)


def find_spots(image, title):
    plt.figure(figsize=(10, 5))

    plt.subplot(111)

    inverted = cv2.bitwise_not(image)

    plt.imshow(inverted, cmap='gray')
    plt.title(title)

    mean_val = np.mean(inverted)
    std_val = np.std(inverted)
    threshold_value = mean_val + std_val + 4000  # Adjust factor if needed

    # Apply threshold to find bright spots
    binary_mask = (inverted >= threshold_value).astype(np.uint8) * 255  # Convert to binary (0 or 255)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Visualization (keeping original pixel values)
    output_img = np.stack([inverted] * 3, axis=-1)  # Convert to 3-channel for drawing

    # Draw bounding boxes and centroids

    spots_coordinates = []

    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Bounding box (green)
        if area <= max_size:
            plt.scatter(cx, cy, 5, c="r")
            # cv2.circle(output_img, (int(cx), int(cy)), 2, (0, 0, 255), -1)  # Centroid (red)
            # print(covid_probe[int(x + w/2)][int(y + h/2)])
            # print(f"X: {cx} Y: {cy} Value: {inverted[int(cx)][int(cy)]} Area: {area}")
            spots_coordinates.append([cx, cy])
            # for i in range(max_size):
            #     if (int(x + w/2)+i) < 512:
            #         print(f"{inverted[int(x + w/2)+i][int(y + h/2)]}")

    return spots_coordinates


def spot_real_intensity(image, x, y):
    try:
        probe_intensity_pixels_avg = 0
        for i in range(-3, 4):
            for j in range(-3, 4):
                probe_intensity_pixels_avg += image[x + i][y + j]
        probe_intensity_pixels_avg /= 49  # averaging out the
        return probe_intensity_pixels_avg
    except:
        return image[x][y]


covid_probe_coordinates = find_spots(filtered_covid_probe, "Laplican Covid+Probe")
probe_coordinates = find_spots(filtered_probe, "Laplican Probe")


covid_probe_16bit = cv2.imread(Covid_Probe_src, cv2.IMREAD_ANYDEPTH)

probe_16bit = cv2.imread(Probe_src, cv2.IMREAD_ANYDEPTH)
print(probe_16bit[35][125])
max_val = probe_16bit.max()
max_idx = np.unravel_index(np.argmax(probe_16bit, axis=None), probe_16bit.shape)
print("Maximum value:", max_val)
print("Located at (row, col):", max_idx)

plt.figure(figsize=(10, 5))

plt.subplot(111)

plt.imshow(covid_probe_16bit, cmap='gray')
plt.title('Covid Probe')

max_probe_intensity = -1
for i in probe_coordinates:
    if spot_real_intensity(probe_16bit, int(i[1]), int(i[0])) > max_probe_intensity:
        max_probe_intensity = spot_real_intensity(probe_16bit, int(i[1]), int(i[0]))


# writer = pd.ExcelWriter('Results.xlsx')
# dataframe = pd.DataFrame({'X-Coordinates': [], 'Y-Coordinates': [], 'GS-Intensity': []})
# print("Max Intensity: ", max_probe_intensity)
# print("Max just value: ", max)
for i in covid_probe_coordinates:  # Skip background (label 0)
    if spot_real_intensity(covid_probe_16bit, int(i[1]), int(i[0])) >= max_probe_intensity:
        plt.scatter(i[0], i[1], 5, c="r")
        plt.imsave('image.jpg', covid_probe_16bit)
        # print(covid_probe_16bit[int(i[0])][int(i[1])])
        # print(f"X: {i[0]} Y: {i[1]} Value: {spot_real_intensity(covid_probe_16bit, int(i[0]), int(i[1]))}")

        # dataframe['X-Coordinates']._append(i[0])
        # dataframe['Y-Coordinates']._append(i[1])
        # dataframe['GS-Intensity']._append(covid_probe_16bit[int(i[0])][int(i[1])])


# df = pd.DataFrame(dataframe)
# print(os.path.splitext(Probe_src[:-4])[0])
# df.to_excel(writer)
# writer._save()

# yx_coords = np.column_stack(np.where(filtered_covid_probe < -21000))
# print(len(yx_coords))
# for i in yx_coords:
#     plt.scatter(i[1], i[0], 1, c="r")





# PLOTTING
# plt.subplot(111)
# plt.figure(figsize=(10, 5))
# res, keypoints = blobize(inverted)

# plt.imshow(res, cmap='gray')
# plt.title('Detection')



# hist, bins, _ = plt.hist(covid_probe.ravel(), 64, (0, 100))

# print(filtered_covid_probe.ravel().max())
# print(filtered_probe.ravel().max())

# for i in (probe.ravel()):
#     print(points)

# for i in (probe.ravel()):
#     print(i)

# for i in range(len(filtered_probe.ravel())):
#     print(i)


# print(covid_probe_gray.ravel().max())
# print(probe_gray.ravel().max())

# plt.hist(probe.ravel(), 64, (0, 100))
# plt.show()


#
# # Plot the original and filtered images
# plt.subplot(141)
# plt.imshow(probe, cmap='gray')
# plt.title('Probe')
#
# plt.subplot(142)
# plt.imshow(filtered_probe, cmap='gray')
# plt.title('LoG Filtered Probe')
#
# plt.subplot(143)
# plt.imshow(covid_probe, cmap='gray')
# plt.title('Covid Probe')
#


# yx_coords = np.column_stack(np.where(filtered_covid_probe > filtered_probe.ravel().max()))
# print(yx_coords)
# for i in yx_coords:
#     plt.scatter(i[1], i[0], 1, c="r")


# hist1 = cv2.calcHist([probe_gray], [0], None, [64], [0, 36])
# plt.subplot(121)
# plt.plot(hist1)
# # label the x-axis
# plt.xlabel('Pixel Intensity')
# # label the y-axis
# plt.ylabel('Number of Pixels')
# # display the title
# plt.title('Probe Grayscale Histogram')
#
# hist2 = cv2.calcHist([covid_probe_gray], [0], None, [64], [0, 36])
# plt.subplot(122)
# plt.plot(hist2)
# # label the x-axis
# plt.xlabel('Pixel Intensity')
# # label the y-axis
# plt.ylabel('Number of Pixels')
# # display the title
# plt.title('Covid Probe Grayscale Histogram')
#
#
plt.show()


# https://medium.com/@rajilini/laplacian-of-gaussian-filter-log-for-image-processing-c2d1659d5d2
# https://medium.com/@meiyee715/do-you-know-python-grayscale-histogram-analysis-b1f8825aef7d