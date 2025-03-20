import cv2
import os
import numpy as np

# Path to the dataset folder and target image
dataset_path = "dataset"
target_image_path = "target.jpg"
output_folder = "system_shift"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the target image
target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
if target_image is None:
    print(f"Error: Could not load target image '{target_image_path}'")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints in the target image
target_keypoints = sift.detect(target_image, None)

# Compute descriptors for the detected keypoints
_, target_descriptors = sift.compute(target_image, target_keypoints)

# Draw keypoints for the target image (optional visualization)
target_keypoint_image = cv2.drawKeypoints(target_image, target_keypoints, None)
cv2.imwrite(os.path.join(output_folder, "target_keypoints.jpg"), target_keypoint_image)

# Set up FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

best_match_count = 0
best_match_image = None
best_match_keypoints = None
best_match_good_matches = None

# Iterate through all images in the dataset folder
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)

    # Load the current image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not load image '{image_path}'")
        continue

    # Detect keypoints in the current image
    keypoints = sift.detect(image, None)

    # Compute descriptors for the detected keypoints
    _, descriptors = sift.compute(image, keypoints)

    # Use FLANN matcher to find matches between the target and current image
    if descriptors is not None and target_descriptors is not None:
        matches = flann.knnMatch(target_descriptors, descriptors, k=2)

        # Apply Lowe's ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Output the number of good matches for the current image
        print(f"Image '{image_name}' has {len(good_matches)} good matches.")

        # Draw keypoints for the current image (optional visualization)
        keypoint_image = cv2.drawKeypoints(image, keypoints, None)
        cv2.imwrite(os.path.join(output_folder, f"keypoints_{image_name}"), keypoint_image)

        # Draw matches for the current image and save
        result_image = cv2.drawMatches(target_image, target_keypoints, image, keypoints, good_matches,
                                       None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        output_path = os.path.join(output_folder, f"matches_{image_name}")
        cv2.imwrite(output_path, result_image)

        # Update the best match if the current image has more good matches
        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_match_image = image
            best_match_keypoints = keypoints
            best_match_good_matches = good_matches

# Draw and save the best match result if found
if best_match_image is not None:
    best_match_result = cv2.drawMatches(target_image, target_keypoints, best_match_image, best_match_keypoints,
                                        best_match_good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_folder, "best_match_result.jpg"), best_match_result)
    print(f"Best match saved as 'best_match_result.jpg' with {best_match_count} good matches.")
else:
    print("No good matches found in the dataset.")
