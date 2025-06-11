import fijibin
import imagej
import json
import os
import pandas as pd

fijibin.BIN = '/Users/yousefrofa/Documents/Programs/ImageJ.app/Contents/MacOS/ImageJ' # Specify the location of ImageJ
fijibin.FIJI_VERSION = '20141125'
folder = "shibeitadata/Yousef(yma9167)/DNA_ImgDet/dsDNA50 1 fM"
sigma = 1.6  # Default value recommended by library doc is 1.6

tif_files = []
for file in os.listdir():
    if file.endswith(".tif"):
        tif_files.append(file)

# for file in tif_files:
#     macro = f'''
#     open("{folder}");
# selectImage("{file});
# run("Run analysis", "filter=[Lowered Gaussian filter] sigma={sigma} detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
# selectImage("Averaged shifted histograms");
# run("Export results", "sigma=true intensity=true offset=true saveprotocol=true filepath=[{folder}\output.csv] x=true y=true bkgstd=true id=false uncertainty=true fileformat=[CSV (comma separated)] frame=true");
#     '''
#     print(file)
#     fijibin.macro.run(macro)


m = f'''
open("C:\\Users\\youse\\Desktop\\output.tif");
selectImage("output.tif");
run("Run analysis", "filter=[Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
run("Export results", "sigma=true intensity=true offset=true saveprotocol=true filepath=C:\\Users\\youse\\Desktop\\output.csv x=true y=true bkgstd=true id=false uncertainty=true fileformat=[CSV (comma separated)] frame=true");
close();
'''
fijibin.macro.run(m)