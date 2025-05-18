"""Black & White Scan enhancer"""

import os
import shutil
import numpy as np
import argparse
from scipy.signal import find_peaks, find_peaks_cwt
from PIL import Image
import matplotlib.pyplot as plt


def plot(histogram: list[int], limit: int = 5000) -> None:
    """plot a histogram"""
    plt.bar(range(len(histogram)), histogram)
    plt.yticks(range(0, max(histogram) + 1, limit))
    plt.show()


def load_histogram(path: str) -> tuple[list[int], list[int], list[int], int]:
    """load the histogram of an image"""
    with Image.open(path) as img:
        histogram = img.histogram()
    smooth_histogram = get_histogram(img)
    minv1 = 0
    for i, v in enumerate(histogram):
        if v >= 100:
            minv1 = i
            break
    inverted = list(-np.array(smooth_histogram))
    peaks, _ = find_peaks(smooth_histogram)
    valleys, _ = find_peaks(inverted)
    peaks2, _ = find_peaks(smooth_histogram, height=1000)
    minv2 = 0
    for i in range(peaks2[0], -1, -1):
        if smooth_histogram[i] < 100:
            minv2 = i + 1
            break
    print("peaks", peaks, [smooth_histogram[p] for p in peaks])
    print("valleys", valleys, [smooth_histogram[v] for v in valleys])
    print("minv", minv1, minv2)
    print("black", find_black_level(smooth_histogram))
    print("white", find_white_level(smooth_histogram))
    return histogram, smooth_histogram, inverted, minv1


def create_destination(destination: str) -> None:
    """remove (if it exists) and create the destination folder"""
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)


def improve_pixel(p: int, wl: int, bl: int) -> int:
    """function to improve the W&B pixel"""
    if p >= wl:
        return 255
    if p <= bl:
        return 0
    newv = int((p - bl) / (wl - bl) * 255)
    return newv


def find_black_level(histogram: list[int]) -> int:
    """find the correct black level of this histogram"""
    med = len(histogram) // 2
    minv = 0
    median = np.median(histogram)
    peaks, _ = find_peaks(histogram, height=median)
    for i, v in enumerate(histogram):
        if v >= 100:
            minv = i
            break
    if len(peaks):
        for i in range(peaks[0], -1, -1):
            if histogram[i] < 100:
                minv = i + 1
                break
    peakscw = find_peaks_cwt(histogram, widths=10)
    peak = peakscw[0] if len(peakscw) else 0
    values = [(histogram[i], i) for i in peaks]
    peak = values[0][1] if values else med
    print(
        minv,
        [int(p) for p in peaks],
        [int(p) for p in peakscw],
        peak,
        ">",
        end=" ",
    )
    if peak < minv + (255 - minv) / 4:
        return peak
    return minv + 15


def find_white_level(histogram: list[int]) -> int:
    """find the correct white level of this histogram"""
    med = len(histogram) // 2
    halve = histogram[med:]
    median = np.median(histogram)
    mean = (
        np.mean([x - median for x in histogram if x > median]) * 1.5 + median
    )
    peaks, _ = find_peaks(halve, height=mean)
    peak = peaks[-1] + med if len(peaks) else 255
    valleys, _ = find_peaks(-np.array(histogram))
    start = peak - 20
    if len(valleys):
        for i in range(len(valleys) - 1, -1, -1):
            valley = valleys[i]
            if valley < peak:
                start = valley
                break
    threshold = histogram[peak] * 0.2
    print(threshold, [int(i + med) for i in peaks], start, peak, ">", end=" ")
    for i in range(start, peak):
        if histogram[i] > threshold:
            return i
    return peak - 15


def get_histogram(img) -> list[int]:
    """get the smooth histogram of image"""
    histogram = img.histogram()
    smooth_histogram = [
        int(x) for x in np.convolve(histogram, np.ones(5) / 5, mode="same")
    ]
    min_value = min(smooth_histogram)
    if min_value:
        smooth_histogram = [x - min_value for x in smooth_histogram]
    return smooth_histogram


def check_bw(pixels: list[tuple[int, int, int]]) -> bool:
    """check if RGB image is black and white"""
    diffs = []
    for p in pixels:
        r, g, b = p
        diffs.append(max(abs(r - g), abs(r - b), abs(g - b)))
    return bool(np.mean(diffs) < 1)


def process_image(
    input_path: str, output_path: str, split: bool = False
) -> None:
    """process image to correct B&W levels and split if required"""
    with Image.open(input_path) as img:
        filename = img.filename
        # Check if the image is color or black and white
        if img.mode == "RGB":
            if check_bw(img.getdata()):
                print(input_path, "is BW")
                img = img.convert("L")
        if img.mode == "L":  # Mode 'L' is black and white
            # Calculate the black and white levels
            print(filename[-7:], end=" ")
            smooth_histogram = get_histogram(img)
            black_level = find_black_level(smooth_histogram)
            print(black_level, "|", end=" ")
            white_level = find_white_level(smooth_histogram)
            print(white_level, end=" ")
            if black_level == 0 or white_level == 255:
                print(smooth_histogram, end=" ")
                exit(1)
            print()

            # Create a new adjusted black and white image
            img = img.point(
                lambda p: improve_pixel(p, white_level, black_level)  # type: ignore
            )

        # Save the processed image
        img.save(output_path)

        # If the --split flag is passed and the image is landscape
        if split and img.width > img.height:
            left_box = (0, 0, img.width // 2, img.height)
            right_box = (img.width // 2, 0, img.width, img.height)
            left_img = img.crop(left_box)
            right_img = img.crop(right_box)
            left_img.save(output_path.replace(".jpg", "-1.jpg"))
            right_img.save(output_path.replace(".jpg", "-0.jpg"))


def main(source_dir: str, dest_dir: str, split: bool) -> None:
    """main function"""
    # Create the destination directory if it does not exist
    create_destination(dest_dir)

    # Process each image in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        ):
            input_path = os.path.join(source_dir, filename)
            output_path = os.path.join(dest_dir, filename)
            process_image(input_path, output_path, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images from a directory."
    )
    parser.add_argument(
        "source_dir", type=str, help="Source directory of the images."
    )
    parser.add_argument(
        "dest_dir",
        type=str,
        help="Destination directory for the processed images.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split landscape images into two.",
    )

    args = parser.parse_args()

    main(args.source_dir, args.dest_dir, args.split)
