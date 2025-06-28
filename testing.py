# # Create an ImageJ2 gateway with the newest available version of ImageJ2.
# import imagej
# import scyjava
# scyjava.config.add_option('-Xmx6g')
# ij = imagej.init()
#
# # Load an image.
# image_url = 'https://imagej.net/images/clown.png'
# jimage = ij.io().open(image_url)
#
# # Convert the image from ImageJ2 to xarray, a package that adds
# # labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
# image = ij.py.from_java(jimage)
#
# # Display the image (backed by matplotlib).
# ij.py.show(image, cmap='gray')

import imagej
import imagej.doctor
import scyjava
import os

# imagej.doctor.checkup()

scyjava.config.add_option('-Xmx6g')
# # os.system('open /Users/yousefrofa/Documents/Programs/ImageJ.app')
ij = imagej.init('/Applications/Fiji.app', headless=False)
ij.ui().showUI()
print(ij.getVersion())

# load test image
# dataset = ij.io().open('output.tif')
#
# # display test image (see the Working with Images for more info)
# ij.py.show(dataset)
input("hihih")