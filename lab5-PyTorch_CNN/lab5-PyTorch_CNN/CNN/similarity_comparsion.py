import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision import models
from itertools import combinations

def extract_feature(model, preprocess, image_path, device):
    """
    Extracts a feature vector from an image using the provided model.

    Args:
        model (torch.nn.Module): The feature extraction model.
        preprocess (torchvision.transforms.Compose): Preprocessing transforms.
        image_path (str): Path to the image.
        device (torch.device): Device to perform computation on.

    Returns:
        np.ndarray: Flattened feature vector.
    """
    try:
        image = default_loader(image_path)
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(input_tensor)
        feature = feature.cpu().numpy().flatten()
        return feature
    except Exception as e:
        print(f'Error extracting features from {image_path}: {e}')
        return None

def similarity_compare(image_paths, features_folder, temp_features_file='temp_features.npy'):
    """
    Computes pairwise Euclidean distances between selected images.

    Args:
        image_paths (list): List of image file paths to compare.
        features_folder (str): Path to the folder containing pre-extracted features.
        temp_features_file (str): Temporary file to store selected features.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load ResNet50 model and remove the final classification layer
    print('Loading ResNet50 model...')
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
    model = model.to(device)
    model.eval()

    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Extract features for each selected image
    features = []
    valid_image_names = []
    for img_path in image_paths:
        print(f'Extracting features from {img_path}...')
        feature = extract_feature(model, preprocess, img_path, device)
        if feature is not None:
            # Normalize the feature
            norm = np.linalg.norm(feature)
            if norm == 0:
                print(f'Feature vector for {img_path} is zero. Skipping.')
                continue
            feature_normalized = feature / norm
            features.append(feature_normalized)
            valid_image_names.append(os.path.basename(img_path))
        else:
            print(f'Failed to extract features from {img_path}.')

    if len(features) < 2:
        print('Need at least two valid images to compare.')
        return

    features_array = np.array(features)
    # Save temporary features if needed
    np.save(os.path.join(features_folder, temp_features_file), features_array)

    # Compute pairwise Euclidean distances
    print('Computing pairwise Euclidean distances...')
    distance_matrix = np.linalg.norm(features_array[:, np.newaxis] - features_array, axis=2)

    # Display the results
    print('\nPairwise Euclidean Distances:')
    for (i, j) in combinations(range(len(valid_image_names)), 2):
        print(f'{valid_image_names[i]} <-> {valid_image_names[j]}: {distance_matrix[i, j]:.4f}')

if __name__ == '__main__':
    image_paths = ["Dataset-1/Dataset/20.jpg", "Dataset-1/Dataset/35.jpg"]
    features_folder = "features1"

    similarity_compare(image_paths, features_folder)
