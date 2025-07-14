#!/usr/bin/env python3
"""
nd2_to_tiff_converter.py

Convert ND2 files (Nikon ND2 format) to TIFF format.

Dependencies:
    pip install nd2reader tifffile

Usage:
    python nd2_to_tiff_converter.py input.nd2
    python nd2_to_tiff_converter.py input_directory/ --output output_dir/
"""
import os
import argparse
import glob
import pathlib

from nd2reader import ND2Reader
import tifffile

def convert_nd2_to_tiff(multi_page: bool = True):
    """
    Reads an ND2 file and writes out TIFF files.

    Args:
        input_path: Path to the .nd2 file.
        output_dir: Directory where .tiff files will be saved.
        multi_page: If True, writes a single multi-page TIFF; otherwise, writes individual TIFF per frame.
    """
    input_path = "/Volumes/shubeitadata/Anand(akd6)/Data for software analysis/Short DNA samples/1 aM short DNA 50 mer/samples incubated at high conc and then diluted/1_aM experiment 1/ssDNA 50"
    output_dir = "/Users/yousefrofa/Documents/Code/DLFM_Detection/Probe"
    for file in os.listdir(input_path):
        if file.endswith(".nd2"):
            with ND2Reader(input_path+"/"+file) as images:
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                filename_base = os.path.splitext(os.path.basename(file))[0]

                if multi_page:
                    frames = [frame for frame in images]
                    out_file = os.path.join(output_dir, f"{filename_base}.tif")
                    tifffile.imwrite(out_file, frames[0])
                    print(f"Wrote multi-page TIFF: {out_file}")
                else:
                    for idx, frame in enumerate(images):
                        out_file = os.path.join(output_dir, f"{filename_base}_frame{idx:04d}.tiff")
                        tifffile.imwrite(out_file, frame)
                    print(f"Wrote {idx+1} TIFF files to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert ND2 files to TIFF format."
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Generate single-frame TIFFs instead of multi-page TIFF"
    )
    args = parser.parse_args()

    input_path = "/Volumes/shubeitadata/Anand(akd6)/Data for software analysis/Short DNA samples/1 aM short DNA 50 mer/samples incubated at high conc and then diluted/1_aM experiment 1/dsDNA50"
    output_dir = "/Users/yousefrofa/Documents/Code/DLFM_Detection/Sample"
    multi_page = not args.single

    # If input is a directory, process all .nd2 files within
    if os.path.isdir(input_path):
        nd2_files = glob.glob(os.path.join(input_path, "*.nd2"))
        if not nd2_files:
            print(f"No .nd2 files found in directory: {input_path}")
            return
        for nd2_file in nd2_files:
            convert_nd2_to_tiff()
    else:
        if not os.path.isfile(input_path):
            print(f"Input file not found: {input_path}")
            return
        if not input_path.lower().endswith('.nd2'):
            print(f"Warning: Input file does not have .nd2 extension: {input_path}")
        convert_nd2_to_tiff(input_path, output_dir, multi_page)

if __name__ == '__main__':
    main()
