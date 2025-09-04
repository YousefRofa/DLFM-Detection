# pip install python-bioformats javabridge opencv-python imageio-ffmpeg
import javabridge, bioformats as bf            # frame access
import cv2                           # filtering + re-encode
import tifffile as tiff
import numpy as np
import math

PATH = "file.cxd"
OUT  = "filtered.avi"
FPS  = 60                                      # fallback if metadata absent

# --- spin up the Bio-Formats JVM ----------
javabridge.start_vm(class_path=bf.JARS)
rdr = bf.ImageReader(PATH)

frame_count = rdr.rdr.getImageCount()
w, h = rdr.rdr.getSizeX(), rdr.rdr.getSizeY()

# --- set up AVI writer --------------------
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
video  = cv2.VideoWriter(OUT, fourcc, 1, (w, h))

# --- streaming loop (no huge RAM spike) ---
frames = []
for t in range(frame_count):
    frame = rdr.read(t=t, z=0, c=0, rescale=False)      # NumPy array
    frames.append(frame)
    # cv2.imwrite("filtered_test.tif", frame)
    # Bio-Formats gives monochrome planes → convert to BGR so VideoWriter is happy
    # video.write(frame.astype("uint8"))

stack = np.stack(frames)

tiff.imwrite(
    "output.ome.tiff",
    stack,
    photometric='minisblack',
    metadata={'axes': 'TYX'},  # T: time, YX: spatial
    ome=True
)

# video.release()
javabridge.kill_vm()
print("hi")