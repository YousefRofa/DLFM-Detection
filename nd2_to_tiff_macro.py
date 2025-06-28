import imagej
from imagej.doctor import checkup
import os
import scyjava
from pathlib import Path

ij = imagej.init("/Applications/Fiji.app", mode="interactive")
ij.thread().queue(lambda: ij.ui().showUI())

print(ij.getVersion())
# ij.ui().showUI()
#
plugins_dir = Path('D:\Programs\Fiji.app\plugins')
scyjava.config.add_option(f'-Dplugins.dir={str(plugins_dir)}')
# ij_path = Path('D:\Programs\Fiji.app')
#
# folder = "shibeitadata/Yousef(yma9167)/DNA_ImgDet/dsDNA50 1 fM"
# sigma = 1.6  # Default value recommended by library doc is 1.6
#
nd2_files = []
for file in os.listdir("Probe"):
    if file.endswith(".nd2"):
        nd2_files.append(file)

for file in nd2_files:
    macro = f'''
        open("/Users/yousefrofa/Documents/Code/DLFM_Detection/Probe/{file}");
        selectImage("{file}");
        saveAs("Tiff", "/Users/yousefrofa/Documents/Code/DLFM_Detection/Probe/{Path(file).stem}.tiff")
        close();
    '''
    print(file)
    ij.py.run_macro(macro)

