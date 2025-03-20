# SJTU EE208

# import time
#
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torchvision.datasets.folder import default_loader
#
# print('Load model: ResNet50')
# model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
#
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# trans = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize,
# ])
#
# print('Prepare image data!')
# test_image = default_loader('panda.png')
# input_image = trans(test_image)
# input_image = torch.unsqueeze(input_image, 0)
#
#
# def features(x):
#     x = model.conv1(x)
#     x = model.bn1(x)
#     x = model.relu(x)
#     x = model.maxpool(x)
#     x = model.layer1(x)
#     x = model.layer2(x)
#     x = model.layer3(x)
#     x = model.layer4(x)
#     x = model.avgpool(x)
#
#     return x
#
#
# print('Extract features!')
# start = time.time()
# image_feature = features(input_image)
# image_feature = image_feature.detach().numpy()
# print('Time for extracting features: {:.2f}'.format(time.time() - start))
#
#
# print('Save features!')
# np.save('features.npy', image_feature)

import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision import models

def extract_features(image_folder, output_folder):
    """
    Extracts features from images using ResNet50 and saves them.

    Args:
        image_folder (str): Path to the folder containing images.
        output_folder (str): Path to the folder to save features.
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

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    print(len(image_files) + 1)

    features_list = []
    image_names = []
    print('Starting feature extraction...')
    start_time = time.time()

    with torch.no_grad():
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(image_folder, img_name)
            try:
                image = default_loader(img_path)
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                feature = model(input_tensor)
                feature = feature.cpu().numpy().flatten()
                features_list.append(feature)
                image_names.append(img_name)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
                    print(f'Processed {idx + 1}/{len(image_files)} images')
            except Exception as e:
                print(f'Error processing {img_name}: {e}')

    features_array = np.array(features_list)
    np.save(os.path.join(output_folder, 'resnet50_features.npy'), features_array)
    np.save(os.path.join(output_folder, 'resnet50_image_names.npy'), np.array(image_names))
    print(f'Feature extraction completed in {time.time() - start_time:.2f} seconds')
    print(f'Features saved to {output_folder}')

if __name__ == '__main__':

    image_folder = 'Dataset-1/Dataset'
    output_folder = 'output'

    extract_features(image_folder, output_folder)

