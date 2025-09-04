import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

radius = 3.5
dimension = 512
sd_radius = 7
min_circularity = .95
max_size = 60 # needs calibartion

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


import numpy as np


def sum_pixels_above_thresh(cx, cy, sd_radius, binary_mask, image_unit16depth, average_bg_intensity):
    h, w = binary_mask.shape

    # Bounding box with clipping to stay inside image
    x_min = max(int(cx) - sd_radius, 0)
    x_max = min(int(cx) + sd_radius + 1, w)
    y_min = max(int(cy) - sd_radius, 0)
    y_max = min(int(cy) + sd_radius + 1, h)

    # Extract ROI
    roi_mask = binary_mask[y_min:y_max, x_min:x_max]
    roi_img = image_unit16depth[y_min:y_max, x_min:x_max]

    # Make a circular mask within the ROI
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= sd_radius ** 2

    # Apply circular mask and binary mask
    mask = (roi_mask == 255) & circle

    # Compute sum
    return np.sum(roi_img[mask] - average_bg_intensity)



def find_spots(image, image_unit16depth,  title):
    # plt.figure(figsize=(10, 5))
    dimension = len(image)

    # fig, ax = plt.subplots()

    inverted = cv2.bitwise_not(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marked_image = image

    # plt.imshow(grayscale, cmap='gray')
    # plt.imshow(covid_probe, cmap='gray')

    # plt.title(title)

    mean_val = np.mean(grayscale)
    std_val = np.std(grayscale)
    # print(title)
    # print(mean_val)
    # print(std_val)
    # print(grayscale.shape)
    # print()
    # print(mean_val, std_val)
    threshold_value = mean_val + std_val*1.6
    # print(threshold_value)
    # Adjust factor if needed

    # Apply threshold to find bright spots
    binary_mask = (grayscale >= threshold_value).astype(np.uint8) * 255  # Convert to binary (0 or 255)
    # plt.imshow(binary_mask, cmap='gray')
    # plt.show()
    # i is y coordinates and j is the x coordinates
    bg_pixels_count = 0
    bg_pixels_total_intensity = 0
    for i in range(len(binary_mask)):
        for j in range(len(binary_mask[0])):
            if binary_mask[i][j] == 0:
                bg_pixels_count += 1
                bg_pixels_total_intensity += int(image_unit16depth[i][j])

    average_bg_intensity = bg_pixels_total_intensity/(bg_pixels_count+1) # We add 1 to avoid division by zero
    # print("average bg intensity", average_bg_intensity)
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
        predicted_radius = math.sqrt(area / math.pi)
        # cv2.rectangle(inverted, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Bounding box (green)
        rect = patches.Rectangle((x, y), w, h, edgecolor='g', facecolor='none')
        # ax.add_patch(rect)

        if radius+predicted_radius < int(cx) < dimension-predicted_radius-radius and predicted_radius+radius < int(cy) < dimension-predicted_radius-radius:
            if  area <= max_size :
                intensity=0
                for i in range(-int(radius), int(radius)+1):
                    for j in range(-int(radius), int(radius)+1):
                        if i*i + j*j <= radius*radius:
                            marked_image[int(cy+i)][int(cx+j)] = [0, 255, 0]
                            intensity += int(image_unit16depth[int(cy+i)][int(cx+j)])-average_bg_intensity


                # Thresholding

                # if sd_radius > int(cx):
                #     lower_x = int(cx)
                # else:
                #     lower_x = sd_radius
                #
                # if dimension - sd_radius < int(cx):
                #     upper_x = int(cx)
                # else:
                #     upper_x = dimension - sd_radius
                #
                # if sd_radius > int(cy):
                #     lower_y = int(cy)
                # else:
                #     lower_y = sd_radius
                #
                # if dimension - sd_radius < int(cy):
                #     upper_y = int(cy)
                # else:
                #     upper_y = dimension - sd_radius
                #
                # try:
                #     for i in range(-lower_x, upper_x + 1):
                #         for j in range(-lower_y, upper_y + 1):
                #             if i * i + j * j <= sd_radius * sd_radius:
                #                 if binary_mask[int(cy + i)][int(cx + j)] == 255:
                #                     pixels_above_thresh += image_unit16depth[int(cy + i)][
                #                                                int(cx + j)] - average_bg_intensity
                # except:
                #     print(cx, cy)

                pixels_above_thresh = sum_pixels_above_thresh(cx, cy, sd_radius, binary_mask, image_unit16depth, average_bg_intensity)

                ## FOR SCATERING
                # print(cx, cy)
                # print(area, predicted_radius)
                pixels_within_area = 0
                for i in range(-int(predicted_radius), int(predicted_radius)+1):
                    for j in range(-int(predicted_radius), int(predicted_radius)+1):
                        if binary_mask[int(cy+i)][int(cx+j)] == 255:
                            pixels_within_area += 1
                circularity= pixels_within_area/area
                # print("circularity", circularity)
                spots_coordinates.append([cx, cy, area, intensity, circularity, pixels_above_thresh])
    # print(spots_coordinates)
    return spots_coordinates

max_probe_intensity = 0
max_probe_intensity_coordinates = [0, 0]
max_probe_intensity_file = ''
max_probe_full_intensity = 0

probe_intensities = []
probe_good_intensities = []
count = 0
for file in os.listdir("Probe"):
    if file.endswith(".tif"):
        image = cv2.imread("Probe/"+file)
        image_unit16depth = cv2.imread("Probe/"+file, cv2.IMREAD_ANYDEPTH)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.filter2D(image, -1, lg_filter())
        data = find_spots(filtered, image_unit16depth, file)
        for single_spot in data:
            count+=1
            size = single_spot[2]
            intensity = single_spot[3]
            circularity = single_spot[4]
            total_intensity = single_spot[5]
            grayscale_intensity = grayscale[int(single_spot[1])][int(single_spot[0])]
            # coordinates area accessed as y, x. Pasas these to the gray scale iamge to get
            # that pixel value
            probe_intensities.append(intensity)
            # print(intensity)
            # print(coordinate)
            if circularity >= min_circularity and size <= max_size:
                probe_good_intensities.append(total_intensity)

            if  intensity > max_probe_intensity and circularity >= min_circularity:
                max_probe_intensity = intensity
                max_probe_intensity_file = file
                max_probe_intensity_coordinates[0]= single_spot[0]
                max_probe_intensity_coordinates[1]= single_spot[1]
                max_probe_full_intensity = single_spot[5]

                cv2.imwrite("unit16.tif", image_unit16depth)
                # print(file)
                # print(int(single_spot[0]), int(single_spot[1]), int(single_spot[2]), int(single_spot[3]), single_spot[4])
                # print(max_probe_intensity, end='\n\n')

plt.title(f"Probe Intensities, count = {count}")
plt.hist(probe_intensities, int(len(probe_intensities)/10))
plt.show()

values = []
sample_intensities = []
sample_good_intensities = []
count = 0
count_ratio = 0

samples_no = len(os.listdir("Sample"))
samples_done = 0

print('MAX SPOT DATA')
print(max_probe_intensity)
print(max_probe_intensity_coordinates)
print(max_probe_intensity_file)
print(max_probe_intensity/max_probe_full_intensity*100)
print('\n\n')


between_1_2 = 0
above_2 = 0

for file in os.listdir("Sample"):
    if file.endswith(".tif"):
        print('\n')
        image = cv2.imread("Sample/"+file)
        image_unit16depth = cv2.imread("Sample/"+file, cv2.IMREAD_ANYDEPTH)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.filter2D(image, -1, lg_filter())
        data = find_spots(filtered, image_unit16depth, file)

        # plt.figure(figsize=(10, 5))
        # plt.imshow(grayscale)
        # plt.title(file)

        for single_spot in data:
            count+=1
            size = single_spot[2]
            intensity = single_spot[3]
            circularity = single_spot[4]
            total_intensity = single_spot[5]
            grayscale_intensity = grayscale[int(single_spot[1])][int(single_spot[0])]
            # print(str(int(coordinate[0]))+","+str(int(coordinate[1])))
            sample_intensities.append(intensity)

            if circularity >= min_circularity and size <= max_size:
                sample_good_intensities.append(total_intensity)

            if intensity > max_probe_intensity and circularity >= min_circularity:
                # plt.scatter(int(single_spot[0]), int(single_spot[1]), s=1, c='r')
                # if (intensity/total_intensity*100 < 95):
                if intensity < 2*max_probe_intensity:
                    between_1_2 += 1
                    print(single_spot[0], single_spot[1])
                else:
                    above_2 += 1

                count_ratio+=1
                values.append(intensity/max_probe_intensity)
        samples_done+=1

        # plt.show()

        print(samples_done/samples_no*100, "% complete.")

print(between_1_2)
print(above_2)

# for i in find_spots(filtered, "Lowered Gaussian"):  # Skip background (label 0)
#     plt.scatter(i[0], i[1], 5, c="r")
plt.title(f"Sample Intensities, count = {count}")
plt.hist(sample_intensities, int(len(sample_intensities)/10))
plt.show()

probe_arr = np.array(probe_good_intensities)
sample_arr = np.array(sample_good_intensities)

# Define common bin edges so both histograms align
bins = np.linspace(
    min(probe_arr.min(), sample_arr.min()),
    max(probe_arr.max(), sample_arr.max()),
    2500  # adjust number of bins (intervals)
)

plt.figure(figsize=(8, 6))
plt.hist(probe_arr, bins=bins, alpha=0.5, color='r', label='Probe', density=False)
plt.hist(sample_arr, bins=bins, alpha=0.5, color='g', label='Sample', density=False)

plt.title("Probe vs Sample Intensities (Frequency Overlap)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.title("Ratio: "+ str(count_ratio)+" Spots")
plt.hist(values)
plt.show()

# print(lg_filter())
