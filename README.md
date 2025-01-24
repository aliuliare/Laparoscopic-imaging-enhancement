# Laparoscopic Image Enhancement

This Python script is designed to process laparoscopic video imagery by enhancing visibility and removing haze and smoke through advanced image processing techniques. Utilizing methods such as Dark Channel Prior, Guided Filtering, and Atmospheric Light Estimation, the script significantly improves the clarity and contrast of the surgical field in laparoscopic videos.

## Features

- **Dark Channel Prior**: Identifies the darkest pixel in image neighborhoods, indicative of areas with minimal haze.
- **Guided Filter**: Provides edge-preserving smoothing to refine the transmission map.
- **Atmospheric Light Estimation**: Estimates the light intensity influenced by haze or smoke.
- **Transmission Estimate**: Calculates the extent of light scattering before reaching the camera.
- **Recover**: Restores the true appearance of the scene by adjusting the influence of haze.
- **Enhance Contrast**: Applies contrast enhancement in areas significantly affected by haze or smoke.

The script is designed to process video frames in real-time, making it suitable for live surgical environments or for use with pre-recorded surgical training materials.

## Installation

To run this project, you need Python and several dependencies:

    pip install numpy opencv-python

## Usage

To use this script, you can simply run it from the command line.

## Contributions

Contributions are welcome! If you have improvements or bug fixes, please fork the repository and submit a pull request.

## Authors

- Alicia Ulierte Arévalo, for Final Project of Medical Image Analysis subject at Biomedical Engineering degree, Rey Juan Carlos University.



