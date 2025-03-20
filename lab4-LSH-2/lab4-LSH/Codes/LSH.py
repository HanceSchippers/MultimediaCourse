import numpy as np
import cv2
import time
from typing import List, Dict


# Normalize a vector (scaling it to unit length)
def normalize_vector(v: List[float]) -> np.ndarray:
    """Normalize a vector to unit length."""
    return np.array(v) / np.linalg.norm(v)


# Compute the color histogram (RGB feature vector) for the image
def compute_color_histogram(img: np.ndarray) -> np.ndarray:
    """Compute a color histogram for the image and return a feature vector."""
    # Split the image into 4 quadrants and calculate the sum of RGB values for each quadrant
    height, width, _ = img.shape
    quadrants = [img[:height // 2, :width // 2], img[:height // 2, width // 2:], img[height // 2:, :width // 2],
                 img[height // 2:, width // 2:]]

    # Sum the RGB values for each quadrant
    feature_vectors = [np.sum(q, axis=(0, 1)) for q in quadrants]

    # Normalize each quadrant's feature vector
    normalized_vectors = [normalize_vector(v) for v in feature_vectors]

    # Concatenate all normalized vectors into a single feature vector
    return np.concatenate(normalized_vectors)


# Normalize the color histogram to a set of discrete values (0, 1, 2)
def normalize_histogram(v: np.ndarray) -> np.ndarray:
    """Normalize the histogram into discrete values (0, 1, 2) based on thresholds."""
    v = v.copy()
    for i in range(len(v)):
        if v[i] < 0.3:
            v[i] = 0
        elif v[i] < 0.6:
            v[i] = 1
        else:
            v[i] = 2
    return v


# Generate the Hamming code from the normalized histogram
def generate_hamming_code(v: np.ndarray) -> str:
    """Generate a Hamming code string from the normalized histogram."""
    return ''.join(['00' if val == 0 else '10' if val == 1 else '11' for val in v])


# Select specific bits from the Hamming code based on the provided index list
def select_bits(hamming_code: str, indices: List[int]) -> str:
    """Select bits from the Hamming code based on the specified indices."""
    return ''.join(hamming_code[i] for i in indices)


# Compute the similarity between two feature vectors using cosine similarity
def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    return np.dot(vec1, vec2)


# Main function to process the dataset and find the most similar image
def find_similar_images(target_img: np.ndarray, dataset_dir: str, key_indices: List[int],
                        num_images: int = 40, top_n: int = 6) -> List[Dict[str, float]]:
    """Find the most similar images from the dataset to the target image."""
    vectors, dataset = [], []

    # Process all images in the dataset
    for i in range(1, num_images + 1):
        img = cv2.imread(f"{dataset_dir}/{i}.jpg")
        color_hist = compute_color_histogram(img)
        normalized_hist = normalize_histogram(color_hist)
        hamming_code = generate_hamming_code(normalized_hist)
        selected_bits = select_bits(hamming_code, key_indices)
        dataset.append(selected_bits)
        vectors.append(color_hist)

    # Process the target image
    target_hist = compute_color_histogram(target_img)
    target_normalized_hist = normalize_histogram(target_hist)
    target_hamming_code = generate_hamming_code(target_normalized_hist)
    target_selected_bits = select_bits(target_hamming_code, key_indices)

    # Find the similarities between the target image and dataset images
    similarities = {}
    for i, vector in enumerate(vectors):
        if dataset[i] == target_selected_bits:
            similarity = compute_similarity(vector, target_hist)
            similarities[i] = similarity

    # Sort the similarities in descending order and pick the top N
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_similar_images = sorted_similarities[:top_n]

    return top_similar_images


if __name__ == "__main__":
    dataset_directory = './Dataset'  # Directory containing the dataset images
    target_image = cv2.imread("./target.jpg")  # Target image to compare

    key_indices = list(range(0,24))  # Indices to select from the Hamming code (adjusted to 0-based)
    #key_indices = [2, 4, 8, 16]
    #key_indices = [2, 8, 18, 23]
    start_time = time.time()
    top_similar_images = find_similar_images(target_image, dataset_directory, key_indices, top_n=6)  # Get top 6 similar images
    end_time = time.time()

    # Print the most similar image and the 5 other top images
    most_similar_image_index, most_similar_image_score = top_similar_images[0]
    print(
        f"The most similar image is: {most_similar_image_index + 1} with similarity score {most_similar_image_score:.4f}")
    print("The other top 5 similar images are:")
    for i, (index, similarity) in enumerate(top_similar_images[1:], start=1):
        print(f"Rank {i+1}: Image {index + 1} with similarity score {similarity:.4f}")

    print(f"Time taken: {end_time - start_time:.4f} seconds")

