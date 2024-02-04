import gradio as gr
import cv2
import numpy as np

def find_skyline(mask):
    h, w = mask.shape

    # Iterate over each column
    for column in range(w):
        # Get each one column
        single_Column = mask[:, column]
        try:
            # Find the first zero index which is the sky
            first_zero_index = np.where(single_Column == 0)[0][0]
            # Find the first one index which is the non-sky
            first_one_index = np.where(single_Column == 1)[0][0]
            if first_zero_index > 20:
                # Set the sky to 1 and the non-sky to 0(black)
                mask[first_one_index:first_zero_index, column] = 1
                mask[first_zero_index:, column] = 0
                mask[:first_one_index, column] = 0
        except:
            continue
    return mask

def detect_sky_by_gradient(image):
    magnitude_threshold = 6

    # Convert the image to grayscale to calculate the gradient
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  Apply blur to remove basic noise
    grayscale_image = cv2.blur(grayscale_image, (10, 5))
    # Apply median filter to remove salt-and-pepper noise further
    # Choose kernel size 3 to remove small noise, same as the Sobel operation
    cv2.medianBlur(grayscale_image, 3)

    # Apply sobel filter to calculate the gradient
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize the gradient to 0-255
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    # Apply threshold to get the gradient mask
    gradient_mask = (sobel_mag < magnitude_threshold).astype(np.uint8)

    # Apply morphological openning(erosion followed by dilation) to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_OPEN, kernel)
    
    # Create a mask to mark the sky with helper function
    mask = find_skyline(mask)
    # Apply the mask to the original image
    detected_image = cv2.bitwise_and(image, image, mask=mask)

    return detected_image

demo = gr.Interface(
    fn=detect_sky_by_gradient,
    inputs='image',
    outputs='image',
    title="Detect Sky",
)

# Launch the interface
# When using Hugging Face, no need to specify share=True
demo.launch()