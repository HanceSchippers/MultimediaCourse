import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision import models

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

def image_search(query_image, features_folder, top_k=5):
    """
    Searches for the most similar images to the query image based on Euclidean distance.

    Args:
        query_image (str): Path to the query image.
        features_folder (str): Path to the folder containing pre-extracted features.
        top_k (int): Number of top similar images to return.
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

    # Load pre-extracted features
    feature_file = os.path.join(features_folder, 'resnet50_features.npy')
    image_names_file = os.path.join(features_folder, 'resnet50_image_names.npy')

    if not os.path.exists(feature_file) or not os.path.exists(image_names_file):
        print(f'Feature files not found in {features_folder}')
        return

    features = np.load(feature_file)
    image_names = np.load(image_names_file)
    print(f'Loaded features for {features.shape[0] + 1} images')

    # Extract features from the query image
    print('Extracting features from the query image...')
    query_feature = extract_feature(model, preprocess, query_image, device)
    if query_feature is None:
        print('Failed to extract features from the query image.')
        return

    # Normalize the query feature
    query_norm = np.linalg.norm(query_feature)
    if query_norm == 0:
        print('Query feature vector is zero.')
        return
    query_feature_normalized = query_feature / query_norm

    # Normalize all features
    features_norm = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm[features_norm == 0] = 1  # Prevent division by zero
    features_normalized = features / features_norm

    # Compute Euclidean distances
    print('Computing Euclidean distances...')
    distances = np.linalg.norm(features_normalized - query_feature_normalized, axis=1)

    # Get the top-K closest images
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:top_k]
    top_distances = distances[top_indices]
    top_image_names = image_names[top_indices]

    # Display the results
    print(f'\nTop-{top_k} similar images:')
    for i in range(top_k):
        print(f'{i+1}. {top_image_names[i]} - Euclidean Distance: {top_distances[i]:.4f}')

if __name__ == '__main__':
    query_image = "leopard.png"
    features_folder = "output"
    top_k = 5

    image_search(query_image, features_folder, top_k)
