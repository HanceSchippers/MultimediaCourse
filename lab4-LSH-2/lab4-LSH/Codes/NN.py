import numpy as np
import cv2
import time
from typing import List


# Normalize a vector (scaling it to unit length)
def normalize_vector(v: List[float]) -> np.ndarray:
    """Normalize a vector to unit length."""
    return np.array(v) / np.linalg.norm(v)


# Compute the color histogram (RGB feature vector) for the image
def compute_color_histogram(img: np.ndarray) -> np.ndarray:
    """Compute a color histogram for the image and return a feature vector."""
    height, width, _ = img.shape
    quadrants = [img[:height // 2, :width // 2], img[:height // 2, width // 2:], img[height // 2:, :width // 2],
                 img[height // 2:, width // 2:]]

    # Sum the RGB values for each quadrant
    feature_vectors = [np.sum(q, axis=(0, 1)) for q in quadrants]

    # Normalize each quadrant's feature vector
    normalized_vectors = [normalize_vector(v) for v in feature_vectors]

    # Concatenate all normalized vectors into a single feature vector
    return np.concatenate(normalized_vectors)


# Euclidean distance function to compute similarity between two vectors
def compute_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the Euclidean distance between two feature vectors."""
    return np.linalg.norm(vec1 - vec2)


# Main function to process the dataset and find the most similar image using Nearest Neighbor
def find_nn_similar_images(target_img: np.ndarray, dataset_dir: str, num_images: int = 40, top_n: int = 6) -> List[int]:
    """Find the most similar images from the dataset to the target image using Nearest Neighbor (NN)."""
    vectors = []

    # Process all images in the dataset
    for i in range(1, num_images + 1):
        img = cv2.imread(f"{dataset_dir}/{i}.jpg")
        color_hist = compute_color_histogram(img)
        vectors.append(color_hist)

    # Process the target image
    target_hist = compute_color_histogram(target_img)

    # Calculate the Euclidean distances between the target image and all dataset images
    distances = []
    for i, vector in enumerate(vectors):
        distance = compute_euclidean_distance(vector, target_hist)
        distances.append((i, distance))

    # Sort the distances in ascending order and pick the top N closest images
    sorted_distances = sorted(distances, key=lambda x: x[1])

    # Extract the indices of the top N most similar images
    top_similar_images = sorted_distances[:top_n]

    return top_similar_images


if __name__ == "__main__":
    dataset_directory = './Dataset'  # Directory containing the dataset images
    target_image = cv2.imread("./target.jpg")  # Target image to compare

    start_time = time.time()
    top_similar_images = find_nn_similar_images(target_image, dataset_directory, top_n=6)  # Get top 6 similar images
    end_time = time.time()

    # Print the most similar image and the 5 other top images with their similarity scores
    best_image_index, best_image_score = top_similar_images[0]
    print(f"The most similar image is: {best_image_index + 1} with similarity score (distance) {best_image_score:.4f}")

    print("The other top 5 similar images are:")
    for i, (image_index, score) in enumerate(top_similar_images[1:], start=1):
        print(f"Rank {i + 1}: Image {image_index + 1} with similarity score (distance) {score:.4f}")

    print(f"Time taken: {end_time - start_time:.4f} seconds")
