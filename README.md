# Black & White Scan Enhancer

## Overview

The **Black & White Scan Enhancer** is a Python script designed to improve the
quality of black and white images by adjusting their pixel values based on
calculated black and white levels. The script processes images from a
specified source directory, enhances them, and saves the results in a
destination directory. Additionally, it offers an option to split landscape
images into two halves.

## Features

- Loads images in various formats (PNG, JPG, JPEG, BMP, GIF).
- Calculates and adjusts the black and white levels of the images.
- Optionally splits landscape images into two separate images.
- Visualizes histograms of image pixel values.

## Requirements

To run this script, you need to have the following Python packages installed:

- `numpy`
- `Pillow` (PIL)
- `matplotlib`
- `scipy`

You can install the required packages using pip:

```bash
pip install numpy Pillow matplotlib scipy
```

## Usage

To use the Black & White Scan Enhancer, run the script from the command line
with the following syntax:

```bash
python enhancer.py <source_dir> <dest_dir> [--split]
```

### Parameters

- `<source_dir>`: The directory containing the images you want to process.
- `<dest_dir>`: The directory where the processed images will be saved.
- `--split`: (Optional) If provided, landscape images will be split into two
separate images.

### Example

```bash
python enhancer.py ./input_images ./output_images --split
```

## License

This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for more details.

## Acknowledgments

This script utilizes various libraries such as NumPy, Pillow, Matplotlib, and
SciPy for image processing and data visualization.
