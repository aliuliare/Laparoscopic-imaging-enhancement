"""
This Python script is designed for processing laparoscopic video imagery to enhance visibility by removing haze and smoke.
It utilizes advanced image processing techniques such as Dark Channel Prior, Guided Filtering, and Atmospheric Light Estimation.
These methods collectively help to clarify the image by estimating and correcting for the light absorbed or scattered by particles,
thus significantly improving the clarity and contrast of the surgical field in laparoscopic videos.

Key functions include:
- dark_channel: Identifies the darkest pixel in image neighborhoods, indicating areas with less haze.
- guided_filter: Provides edge-preserving smoothing to refine the transmission map.
- atmospheric_light: Estimates the light intensity influenced by haze or smoke.
- transmission_estimate: Calculates how much of the light has been scattered before reaching the camera.
- recover: Restores the true appearance of the scene by adjusting the influence of haze.
- enhance_contrast: Applies contrast enhancement in areas with significant haze or smoke.

The script processes video frames in real-time, making it suitable for live surgical environments or pre-recorded surgical training materials.
Medical Image Analysis - Biomedical engineering - URJC
Final project 2024-25 - Group C - Alicia Ulierte Arévalo
"""



import cv2  # Importing the OpenCV library to handle image processing tasks.
import numpy as np  # Importing numpy for handling numerical operations and arrays.

# Dark Channel Prior Calculation
def dark_channel(img, size=15):
    '''
    Calculates the dark channel prior for an image, which is an important component in haze removal techniques.
    This function assumes haze has a higher intensity in the lighter parts of the image and aims to identify
    the darkest pixel in the local patches of each channel, which corresponds to the least amount of haze.

    Parameters:
        img (numpy.ndarray): The input image in BGR format.
        size (int): The size of the local patch (kernel size) used to calculate the dark channel.

    Returns:
        numpy.ndarray: The dark channel of the image.
    '''
    # Split the image into its blue, green, and red components
    b, g, r = cv2.split(img)
    # Find the minimum color value for each pixel across color channels
    min_img = cv2.min(cv2.min(r, g), b)
    # Create a structural element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    # Erode the minimum image to find the darkest value in the given patch size
    dark = cv2.erode(min_img, kernel)
    return dark

# Guided Image Filtering
def guided_filter(I, p, r, eps):
    '''
    Performs guided filtering on an image. This helps in refining the transmission map, providing
    edge-preserving smoothing. It is useful in laparoscopic imagery where maintaining edge
    clarity is very important while removing haze or smoke.

    Parameters:
        I (numpy.ndarray): The guidance image (usually grayscale version of the input image).
        p (numpy.ndarray): The input image to be filtered.
        r (int): The radius of the square filter window.
        eps (float): Regularization parameter to avoid division by zero.

    Returns:
        numpy.ndarray: The output filtered image.
    '''
    # Calculate means of I, p, and their product within the window
    I_mean = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    p_mean = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    Ip_mean = cv2.boxFilter(I*p, cv2.CV_64F, (r, r))
    # Calculate covariances and variances needed for the guided filter
    cov_Ip = Ip_mean - I_mean * p_mean
    I_var = cv2.boxFilter(I*I, cv2.CV_64F, (r, r)) - I_mean * I_mean
    # Compute the coefficients 'a' and 'b'
    a = cov_Ip / (I_var + eps)
    b = p_mean - a * I_mean
    # Calculate mean of coefficients over the window
    a_mean = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    b_mean = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # Compute the output filtered image
    q = a_mean * I + b_mean
    return q

# Atmospheric Light Estimation
def atmospheric_light(img, dark):
    '''
    Estimates the atmospheric light by averaging the brightest pixels in the dark channel.
    It represents the amount of ambient light and is used to better
    adjust the transmission and restoration of the image colors in dehazing or smoke removal processes.

    Parameters:
        img (numpy.ndarray): The original image.
        dark (numpy.ndarray): The dark channel of the image.

    Returns:
        numpy.ndarray: The estimated atmospheric light.
    '''
    # Get the number of pixels in the image
    h, w = img.shape[:2]
    n_pixels = h * w
    # Select the top 0.1% brightest pixels in the dark channel
    top_count = int(max(np.floor(n_pixels * 0.001), 1))
    dark_vec = dark.reshape(n_pixels)
    img_vec = img.reshape(n_pixels, 3)
    # Find the indices of the brightest pixels
    indices = dark_vec.argsort()[::-1][:top_count]
    # Average the pixel values at these indices
    atmo = np.mean(img_vec[indices], axis=0)
    return atmo

# Transmission Estimation
def transmission_estimate(img, A, omega=0.95, size=15):
    '''
    Estimates the transmission map of an image. The transmission map represents areas where light has not
    been significantly scattered and is crucial for restoring the image. 

    Parameters:
        img (numpy.ndarray): The input image.
        A (numpy.ndarray): The estimated atmospheric light.
        omega (float): The scattering coefficient, usually less than 1.
        size (int): The size of the patch for calculating the dark channel.

    Returns:
        numpy.ndarray: The estimated transmission map.
    '''
    # Normalize the image by atmospheric light
    norm_img = img / A
    # Estimate transmission by subtracting the product of omega and dark channel
    t = 1 - omega * dark_channel(norm_img, size)
    return t

# Scene Radiance Recovery
def recover(img, t, A, t0=0.1):
    '''
    Recovers the scene radiance from an image given the transmission map and atmospheric light.
    This function restores the true colors and contrast of the image by adjusting for the effect of haze or smoke.

    Parameters:
        img (numpy.ndarray): The input image.
        t (numpy.ndarray): The transmission map.
        A (numpy.ndarray): The atmospheric light.
        t0 (float): A lower bound for transmission to prevent division by zero.

    Returns:
        numpy.ndarray: The dehazed or smoke-free image.
    '''
    # Clip the transmission to avoid division by zero
    t = cv2.max(t, t0)
    # Recover the scene radiance using the transmission and atmospheric light
    J = (img - A) / t[:, :, np.newaxis] + A
    # Clip values to proper range
    J = np.clip(J, 0, 255)
    return J.astype(np.uint8)

# Contrast Enhancement in Low Transmission Areas
def enhance_contrast(img, t, threshold=1):
    '''
    Enhances the contrast in areas with low transmission. This function applies adaptive histogram equalization
    to areas identified as having low transmission, which typically represent thicker haze or more dense smoke.

    Parameters:
        img (numpy.ndarray): The dehazed or smoke-free image.
        t (numpy.ndarray): The transmission map.
        threshold (float): The cutoff point for identifying low transmission areas.

    Returns:
        numpy.ndarray: The contrast-enhanced image.
    '''
    # Create a mask for low transmission areas
    low_t_mask = (t < threshold).astype(np.uint8)
    low_t_mask = cv2.merge([low_t_mask]*3)  # Convert to 3 channels

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Combine enhanced regions with original image
    result = np.where(low_t_mask == 1, enhanced_img, img)
    return result

# Path and capture setup for video or camera input
video_path = '../code/input_videos_low_res/video7.mp4'
cam_path = 0

# Initialize video capture
cap = cv2.VideoCapture(cam_path)
if not cap.isOpened():
    print("No se pudo abrir el video.")
else:
    print("Video abierto correctamente.")

# Main loop for frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame
    dark = dark_channel(img)
    A = atmospheric_light(img, dark)
    t = transmission_estimate(img, A)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    t_refined = guided_filter(gray_img, t.astype(np.float64), r=40, eps=1e-3)
    J = recover(img.astype(np.float64), t_refined, A)
    J_contrast_enhanced = enhance_contrast(J, t_refined)

    # Display original vs processed frame and stop if q is pressed
    combined = np.hstack((frame, J_contrast_enhanced))
    cv2.imshow("Video Original vs Enhanced", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
