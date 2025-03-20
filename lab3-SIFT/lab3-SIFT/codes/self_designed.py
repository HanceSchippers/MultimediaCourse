import cv2
import numpy as np
import os

def build_image_pyramid(image, levels=3):
    """
    Construct an image pyramid by resizing the image to multiple scales.
    """
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)  # Downscale the image
        pyramid.append(image)
    return pyramid

def compute_harris_corners(image, block_size=2, ksize=3, k=0.04):
    """
    Use Harris corner detection to find keypoints with an adaptive threshold.
    """
    corners = cv2.cornerHarris(image, blockSize=block_size, ksize=ksize, k=k)
    corners = cv2.dilate(corners, None)  # Enhance corner points
    adaptive_thresh = 0.25 * corners.max()  # Compute adaptive threshold based on maximum response
    keypoints = np.argwhere(corners > adaptive_thresh)
    keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in keypoints]
    return keypoints


def compute_sift_descriptor(image, keypoints, patch_size=16):
    """
    Accurately compute SIFT-like descriptors with bilinear interpolation for keypoints.
    Each descriptor uses a 16x16 neighborhood divided into 4x4 cells,
    with an 8-bin histogram for each cell (128-dimensional descriptor).
    """
    descriptors = []
    image = cv2.GaussianBlur(image, (5, 5), 1.6)  # Gaussian smoothing to reduce noise.

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])  # Keypoint coordinates.

        # Extract the 16x16 neighborhood around the keypoint.
        patch = image[max(0, y - patch_size // 2):y + patch_size // 2,
                      max(0, x - patch_size // 2):x + patch_size // 2]

        # Skip incomplete patches near image boundaries.
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue

        # Compute gradients using Sobel operators.
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Adjust angles relative to the dominant keypoint orientation.
        main_orientation = kp.angle if kp.angle else 0
        adjusted_angle = (angle - main_orientation) % 360

        # Initialize a 4x4 grid with 8 bins per cell (128-dimensional descriptor).
        cell_size = patch_size // 4  # Each cell is 4x4 pixels.
        descriptor = np.zeros((4, 4, 8), dtype=np.float32)

        # Iterate over all pixels in the 16x16 patch.
        for i in range(patch_size):
            for j in range(patch_size):
                # Compute relative coordinates within the grid.
                patch_x = j + 0.5  # Pixel center adjustment
                patch_y = i + 0.5

                cell_x = patch_x / cell_size  # Grid row index
                cell_y = patch_y / cell_size  # Grid column index

                # Identify the 4 nearest grid cells.
                x0, y0 = int(cell_x), int(cell_y)
                x1, y1 = min(x0 + 1, 3), min(y0 + 1, 3)  # Ensure within 4x4 grid boundaries.

                # Compute interpolation weights.
                dx1, dy1 = cell_x - x0, cell_y - y0
                dx2, dy2 = 1 - dx1, 1 - dy1

                # Quantize angle into 8 bins (each 45 degrees).
                bin_idx = int(adjusted_angle[i, j] // 45) % 8
                magnitude_value = magnitude[i, j]

                # Bilinearly interpolate the gradient magnitude into the 4 nearest cells.
                descriptor[y0, x0, bin_idx] += magnitude_value * dx2 * dy2
                descriptor[y0, x1, bin_idx] += magnitude_value * dx1 * dy2
                descriptor[y1, x0, bin_idx] += magnitude_value * dx2 * dy1
                descriptor[y1, x1, bin_idx] += magnitude_value * dx1 * dy1

        # Flatten the 4x4x8 grid into a 128-dimensional vector.
        descriptor = descriptor.flatten()

        # Normalize the descriptor to unit length (L2 normalization).
        descriptor /= (np.linalg.norm(descriptor) + 1e-7)

        # Clip values to 0.2 to suppress extreme gradients and re-normalize.
        descriptor = np.clip(descriptor, 0, 0.2)
        descriptor /= (np.linalg.norm(descriptor) + 1e-7)

        descriptors.append(descriptor)

    return np.array(descriptors, dtype=np.float32)


def match_features(descriptors1, descriptors2):
    """
    Match features between two sets of descriptors using a brute-force matcher.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def main():
    target_path = "target.jpg"
    dataset_folder = "dataset"
    output_folder = "output_matches"

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load target image and create image pyramid
    target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    target_pyramid = build_image_pyramid(target_image, levels=3)

    # Compute Harris corners and SIFT descriptors for target image
    target_keypoints = compute_harris_corners(target_pyramid[0])
    target_descriptors = compute_sift_descriptor(target_pyramid[0], target_keypoints)

    for image_name in os.listdir(dataset_folder):
        image_path = os.path.join(dataset_folder, image_name)
        dataset_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dataset_pyramid = build_image_pyramid(dataset_image, levels=3)

        # Compute Harris corners and SIFT descriptors for dataset image
        dataset_keypoints = compute_harris_corners(dataset_pyramid[0])
        dataset_descriptors = compute_sift_descriptor(dataset_pyramid[0], dataset_keypoints)

        # Match features
        matches = match_features(target_descriptors, dataset_descriptors)


        print(f"{image_name}: {len(matches)} matches")

        # Draw matches
        result_image = cv2.drawMatches(target_pyramid[0], target_keypoints,
                                       dataset_pyramid[0], dataset_keypoints,
                                       matches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        output_path = os.path.join(output_folder, f"matches_{image_name}.png")
        cv2.imwrite(output_path, result_image)

if __name__ == "__main__":
    main()

