import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.patches as patches

from tkinter import *
import os

root = Tk()
root.withdraw()
# Probe_folder_root = filedialog.askdirectory(initialdir="/",title="Select Probe Folder")
# Covid_folder_root = filedialog.askdirectory(initialdir="/",title="Select Covid+Probe Folder")

Probe_folder_root = "Z:\Yousef(yma9167)\DNA_ImgDet\ssDNA50 1 fM"
Covid_folder_root = "Z:\Yousef(yma9167)\DNA_ImgDet\dsDNA50 1 fM"

Probe_files = os.listdir(Probe_folder_root)
Covid_Probe_files = os.listdir(Covid_folder_root)

files_array = [[os.path.join(Probe_folder_root, Probe_files[i]), os.path.join(Covid_folder_root, Covid_Probe_files[i])] for i in range(len(Probe_files))]


# def multiple_dfs(df_list, sheets, file_name, spaces):
#     writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
#     row = 0
#     for dataframe in df_list:
#         dataframe.to_excel(writer,sheet_name=sheets,startrow=row , startcol=0)
#         row = row + len(dataframe.index) + spaces + 1
#     writer.save()
#

# with pd.ExcelWriter('results.xlsx', mode='w') as writer:

file = []
x_coo = []
y_coo = []
intensity = []

for files in files_array:
    Probe_src = files[0]
    Covid_Probe_src = files[1]

    print(Covid_Probe_src)

    # print(Probe_src)

    max_size = 40# This sizing needs calibration

    # print(Probe_src)
    probe = cv2.imread(Probe_src, cv2.IMREAD_COLOR)

    probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    filtered_probe = cv2.Laplacian(probe_gray, cv2.CV_16S, ksize=5)  # CV_16S is 16 bits image signed

    covid_probe = cv2.imread(Covid_Probe_src, cv2.IMREAD_COLOR)
    covid_probe_gray = cv2.cvtColor(covid_probe, cv2.COLOR_BGR2GRAY)
    filtered_covid_probe = cv2.Laplacian(covid_probe_gray, cv2.CV_16S, ksize=5)  # CV_16S is 16 bits image signed

    # probe_gray = cv2.resize(probe_gray, (0, 0), fx=0.5, fy=0.5)


    def find_spots(image, title):
        plt.figure(figsize=(10, 5))

        fig, ax = plt.subplots()

        inverted = cv2.bitwise_not(image)

        plt.imshow(inverted, cmap='gray')
        # plt.imshow(covid_probe, cmap='gray')

        plt.title(title)

        mean_val = np.mean(inverted)
        std_val = np.std(inverted)
        print(title)
        print(mean_val)
        print(std_val)
        threshold_value = mean_val + std_val*3
        # Adjust factor if needed

        # Apply threshold to find bright spots
        binary_mask = (inverted >= threshold_value).astype(np.uint8) * 255  # Convert to binary (0 or 255)

        # Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        # print(num_labels, labels, stats, centroids)

        # Visualization (keeping original pixel values)
        output_img = np.stack([inverted] * 3, axis=-1)  # Convert to 3-channel for drawing

        # Draw bounding boxes and centroids

        spots_coordinates = []

        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            # cv2.rectangle(inverted, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Bounding box (green)
            rect = patches.Rectangle((x, y), w, h, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            if area <= max_size:
                plt.scatter(cx, cy, 5, c="r")
                ## FOR SCATERING
                spots_coordinates.append([cx, cy])

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


    covid_probe_coordinates = find_spots(filtered_covid_probe, "Laplacian Covid+Probe")
    probe_coordinates = find_spots(filtered_probe, "Laplacian Probe")

    covid_probe_16bit = cv2.imread(Covid_Probe_src, cv2.IMREAD_ANYDEPTH)

    probe_16bit = cv2.imread(Probe_src, cv2.IMREAD_ANYDEPTH)
    max_val = probe_16bit.max()
    max_idx = np.unravel_index(np.argmax(probe_16bit, axis=None), probe_16bit.shape)
    # print("Maximum value:", max_val)
    # print("Located at (row, col):", max_idx)

    plt.figure(figsize=(10, 5))

    plt.subplot(111)

    plt.imshow(covid_probe_16bit, cmap='gray')
    plt.title('Covid Probe')

    max_probe_intensity = -1
    for i in probe_coordinates:
        if spot_real_intensity(probe_16bit, int(i[1]), int(i[0])) > max_probe_intensity:
            max_probe_intensity = spot_real_intensity(probe_16bit, int(i[1]), int(i[0]))

    file.append(files[1].split('\\')[-1][:-3])
    x_coo.append('-')
    y_coo.append('-')
    intensity.append('-')

    print("Max intensity is :", max_probe_intensity)

    for i in covid_probe_coordinates:  # Skip background (label 0)
        if spot_real_intensity(covid_probe_16bit, int(i[1]), int(i[0])) > max_probe_intensity:
            plt.scatter(i[0], i[1], 5, c="r")
            file.append('-')
            x_coo.append(int(i[0]))
            y_coo.append(int(i[1]))
            print("Intensity is : ", spot_real_intensity(covid_probe_16bit, int(i[1]), int(i[0])))
            intensity.append(spot_real_intensity(covid_probe_16bit, int(i[1]), int(i[0])))

            plt.savefig('results/'+files[1].split('\\')[-1][:-3]+'jpeg')

    plt.show()

    # dataframe = pd.DataFrame({'File': file, 'X-Coordinates': x_coo, 'Y-Coordinates': y_coo, 'GS-Intensity': intensity})
    # dataframe.to_excel(writer)


# https://medium.com/@rajilini/laplacian-of-gaussian-filter-log-for-image-processing-c2d1659d5d2
# https://medium.com/@meiyee715/do-you-know-python-grayscale-histogram-analysis-b1f8825aef7d