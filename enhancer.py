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


def load_histogram(path: str) -> list[int]:
    """load the histogram of an image"""
    with Image.open(path) as img:
        histogram = img.histogram()
    return histogram


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
    halve = histogram[:med]
    threshold = np.percentile(halve, 95)
    mean = np.mean(histogram) * 0.75
    peaks0, _ = find_peaks(halve, height=mean, width=10)
    peaks = find_peaks_cwt(halve, widths=10)
    peak = peaks[0] if len(peaks) else 0
    values = [(halve[i], i) for i in peaks0 if i < 100]
    peak = max(values)[1] if values else med
    print(
        threshold,
        [int(p) for p in peaks0],
        [int(p) for p in peaks],
        peak,
        ">",
        end=" ",
    )
    for i in range(peak, med):
        current = histogram[i]
        if current < threshold:
            return i
    return 15


def find_white_start(
    med: int, peak: int, halve: list[int], peaks: list[int]
) -> int:
    """find the first to point to start looking for the correct white level
    from the medium point between peaks (current peak and previous peak)"""
    rest_peaks = [p for p in peaks if p + 128 != peak]
    # print(rest_peaks, [(halve[i], (peak + i + med) // 2) for i in rest_peaks])
    if rest_peaks:
        return max([(halve[i], (peak + i + med) // 2) for i in rest_peaks])[1]
    return med


def find_white_level(histogram: list[int]) -> int:
    """find the correct white level of this histogram"""
    med = len(histogram) // 2
    halve = histogram[med:]
    threshold = np.percentile(halve, 85)
    mean = np.mean(histogram) * 0.75
    peaks, _ = find_peaks(halve, height=mean, width=10)
    peak = max([(halve[i], i + med) for i in peaks])[1]
    start = find_white_start(med, peak, halve, peaks)
    print(threshold, [int(i + med) for i in peaks], start, peak, ">", end=" ")
    for i in range(start, peak):
        if histogram[i] > threshold:
            return i
    return 240


def process_image(
    input_path: str, output_path: str, split: bool = False
) -> None:
    """process image to correct B&W levels and split if required"""
    with Image.open(input_path) as img:
        # Check if the image is color or black and white
        if img.mode == "L":  # Mode 'L' is black and white
            # Calculate the black and white levels
            print(img.filename[-7:], end=" ")
            histogram = img.histogram()
            smooth_histogram = [
                int(x)
                for x in np.convolve(histogram, np.ones(5) / 5, mode="same")
            ]
            black_level = find_black_level(smooth_histogram)
            print(black_level, "|", end=" ")
            white_level = find_white_level(smooth_histogram)
            print(white_level, end=" ")
            if black_level == 0 or white_level == 255:
                print(histogram, end=" ")
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
