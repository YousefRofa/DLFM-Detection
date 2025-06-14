import imagej
import scyjava
from scyjava import jimport
import imagej.doctor
from pathlib import Path

scyjava.config.add_option('-Xmx12g')
plugins_dir = Path('D:\Programs\Fiji.app\plugins')
# scyjava.config.add_option(f'-Dplugins.dir={str(plugins_dir)}')
ij_path = Path('D:\Programs\Fiji.app')

# imagej.doctor.checkup()
# ij = imagej.init('sc.fiji:fiji+com.github.zitmen:thunderstorm-algorithms:257978741c', headless=False)
ij = imagej.init(ij_path, headless=False)
ij.ui().showUI()
print(ij.getVersion())

# image_iterable = ij.op().transform().flatIterableView(ij.py.to_java("Z:\Yousef(yma9167)\DNA_ImgDet\dsDNA50 1 fM\output.tif"))

# image = ij.io().open("Z:\Yousef(yma9167)\DNA_ImgDet\dsDNA50 1 fM\output.tif")
#
# image_iterable = ij.op().transform().flatIterableView(ij.py.to_java(image))
#
# ij.ui().show(image_iterable)
# WindowManager = jimport('ij.WindowManager')
# current_image = WindowManager.getCurrentImage()

# macro = """
# rename("active")
# run("Duplicate...", "duplicate")
# selectWindow("active")
# run("Close")
# selectWindow("active-1")
# run("Re-order Hyperstack ...", "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]")
# """
# ij.py.run_macro(macro)
# dataset = ij.io().open("Z:\Yousef(yma9167)\DNA_ImgDet\dsDNA50 1 fM\output.tif")

# IJ.py.show(dataset)
# macro = '''
# open("D:/Uni/Research/Prof. George Shubeita/DNA_ImgDet/output.tif");
# run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
# run("Export results", "sigma=true intensity=true offset=true saveprotocol=true filepath=C:\\Users\\youse\\Desktop\\output.csv x=true y=true bkgstd=true id=false uncertainty=true fileformat=[CSV (comma separated)] frame=true");
# '''
plugin = "Run analysis"
args = {}
result = ij.py.run_plugin(plugin, args)
input()
# print(result)
# ij.IJ.run(dataset, "Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
# ij.IJ.run("Export results", "sigma=true intensity=true offset=true saveprotocol=true filepath=C:\\Users\\youse\\Desktop\\output.csv x=true y=true bkgstd=true id=false uncertainty=true fileformat=[CSV (comma separated)] frame=true");