import math
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Input folder containing dataset images
input_folder = 'dataset'

# Get the list of image files in the dataset folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Function to apply edge detection using selected operator with Non-Maximum Suppression
def custom_edge_detection(image, operator='sobel', low_threshold=50, high_threshold=150):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute gradients based on the selected operator
    if operator == 'sobel':
        # Compute gradients using Sobel operator
        grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
    elif operator == 'roberts':
        # Define Roberts operator kernels
        kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)

        # Apply the kernels to get gradients
        grad_x = cv2.filter2D(blurred, cv2.CV_16S, kernel_x)
        grad_y = cv2.filter2D(blurred, cv2.CV_16S, kernel_y)
    elif operator == 'prewitt':
        # Define Prewitt operator kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

        # Apply the kernels to get gradients
        grad_x = cv2.filter2D(blurred, cv2.CV_16S, kernel_x)
        grad_y = cv2.filter2D(blurred, cv2.CV_16S, kernel_y)

    # Convert gradients to absolute values and then to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Combine the gradients to get the edge magnitude and direction
    gradient_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    gradient_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    gradient_direction[gradient_direction < 0] += 180

    # Non-Maximum Suppression with Interpolation
    nms_result = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    rows, cols = gradient_magnitude.shape

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            angle = gradient_direction[r, c]
            mag = gradient_magnitude[r, c]

            # Determine the neighboring pixels to interpolate based on the gradient direction
            if (0 <= angle < 45):
                weight = math.tan(math.radians(angle))
                neighbor_1 = (1 - weight) * gradient_magnitude[r + 1, c] + weight * gradient_magnitude[r + 1, c + 1]
                neighbor_2 = (1 - weight) * gradient_magnitude[r - 1, c] + weight * gradient_magnitude[r - 1, c - 1]
            elif 45 <= angle < 90:
                weight = 1 / math.tan(math.radians(angle))
                neighbor_1 = (1 - weight) * gradient_magnitude[r, c + 1] + weight * gradient_magnitude[r + 1, c + 1]
                neighbor_2 = (1 - weight) * gradient_magnitude[r, c - 1] + weight * gradient_magnitude[r - 1, c - 1]
            elif 90 <= angle < 135:
                weight = math.tan(math.radians(angle - 90))
                neighbor_1 = (1 - weight) * gradient_magnitude[r, c + 1] + weight * gradient_magnitude[r - 1, c + 1]
                neighbor_2 = (1 - weight) * gradient_magnitude[r, c - 1] + weight * gradient_magnitude[r + 1, c - 1]
            elif 135 <= angle < 180:
                weight = math.tan(math.radians(180 - angle))
                neighbor_1 = (1 - weight) * gradient_magnitude[r - 1, c] + weight * gradient_magnitude[r - 1, c + 1]
                neighbor_2 = (1 - weight) * gradient_magnitude[r + 1, c] + weight * gradient_magnitude[r + 1, c - 1]

            # Suppress non-maximum values using interpolation
            if mag >= neighbor_1 and mag >= neighbor_2:
                nms_result[r, c] = mag
            else:
                nms_result[r, c] = 0

    # Apply double threshold to get edges
    _, strong_edges = cv2.threshold(nms_result, high_threshold, 255, cv2.THRESH_BINARY)
    _, weak_edges = cv2.threshold(nms_result, low_threshold, 255, cv2.THRESH_BINARY)

    # Combine strong and weak edges
    edges = cv2.bitwise_or(strong_edges, weak_edges)

    return edges

output_folder = 'operator_comparison'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Additional output folders for different comparisons
canny_detection_folder = 'canny_detection'
threshold_comparison_folder = 'threshold_comparison'

if not os.path.exists(canny_detection_folder):
    os.makedirs(canny_detection_folder)

if not os.path.exists(threshold_comparison_folder):
    os.makedirs(threshold_comparison_folder)

# Process each image in the dataset
for i, image_file in enumerate(image_files):
    # Read the image
    img = cv2.imread(os.path.join(input_folder, image_file))

    # Check if the image was successfully loaded
    if img is None:
        print(f"Unable to load image {image_file}")
        continue

    # Apply custom Sobel edge detection
    custom_sobel_edges = custom_edge_detection(img, operator='sobel', low_threshold=50, high_threshold=150)

    # Apply custom Roberts edge detection
    custom_roberts_edges = custom_edge_detection(img, operator='roberts', low_threshold=50, high_threshold=150)

    # Apply custom Prewitt edge detection
    custom_prewitt_edges = custom_edge_detection(img, operator='prewitt', low_threshold=50, high_threshold=150)

    # Apply OpenCV's built-in Canny edge detection
    canny_edges = cv2.Canny(img, 50, 150)

    # Save the results for comparison in the main output folder
    sobel_output_path = os.path.join(output_folder, f'sobel_edges_{i + 1}.png')
    roberts_output_path = os.path.join(output_folder, f'roberts_edges_{i + 1}.png')
    prewitt_output_path = os.path.join(output_folder, f'prewitt_edges_{i + 1}.png')
    canny_output_path = os.path.join(output_folder, f'canny_edges_{i + 1}.png')
    cv2.imwrite(sobel_output_path, custom_sobel_edges)
    cv2.imwrite(roberts_output_path, custom_roberts_edges)
    cv2.imwrite(prewitt_output_path, custom_prewitt_edges)
    cv2.imwrite(canny_output_path, canny_edges)

    # Plot the original image, Sobel, Roberts, Prewitt, and OpenCV Canny results
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(custom_sobel_edges, cmap='gray')
    plt.title('Custom Sobel')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(custom_roberts_edges, cmap='gray')
    plt.title('Custom Roberts')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(custom_prewitt_edges, cmap='gray')
    plt.title('Custom Prewitt')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('OpenCV Canny')
    plt.axis('off')

    # Save the comparison plot
    comparison_output_path = os.path.join(output_folder, f'comparison_{i + 1}.png')
    plt.savefig(comparison_output_path)
    plt.close()

    print(f"Processed image saved: {comparison_output_path}")

    # Additional comparison: Sobel vs Canny vs Original
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(custom_sobel_edges, cmap='gray')
    plt.title('Custom Sobel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('OpenCV Canny')
    plt.axis('off')

    canny_comparison_output_path = os.path.join(canny_detection_folder, f'canny_comparison_{i + 1}.png')
    plt.savefig(canny_comparison_output_path)
    plt.close()

    print(f"Canny comparison saved: {canny_comparison_output_path}")

    # Additional comparison: Different thresholds for Sobel vs Original
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    for j, (low_thresh, high_thresh) in enumerate([(40, 100), (50, 150), (80, 200)]):
        sobel_edges = custom_edge_detection(img, operator='sobel', low_threshold=low_thresh, high_threshold=high_thresh)
        plt.subplot(1, 4, j + 2)
        plt.imshow(sobel_edges, cmap='gray')
        plt.title(f'Sobel (Low={low_thresh}, High={high_thresh})')
        plt.axis('off')

    threshold_comparison_output_path = os.path.join(threshold_comparison_folder, f'threshold_comparison_{i + 1}.png')
    plt.savefig(threshold_comparison_output_path)
    plt.close()

    print(f"Threshold comparison saved: {threshold_comparison_output_path}")

print("Processing complete!")
