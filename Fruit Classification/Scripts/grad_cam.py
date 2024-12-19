import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import os
from torchvision import models
from model_builder import ResNet50Transfer, MobileNetV3Transfer
import matplotlib.pyplot as plt

model_path = "models/model_mobileNetV3.pth"
target_layer_name = "base_model.features.16.0"

# Initialize the custom model
model = MobileNetV3Transfer(2)

# Get the list of modules in the model
modules = list(model.named_modules())

# Reverse the list to go from the end to the beginning of the model
modules.reverse()

# Display the last 7 layers
print("The last 7 layers of MobileNetV3Large are:")
for i, (name, module) in enumerate(modules[:30]):
    print(f"{i+1}. {name}: {type(module).__name__}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def _compute_gradcam(self, input_tensor, class_idx):
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            print("Forward hook called! Output shape:", output.shape)
            activations['value'] = output

        def backward_hook(module, grad_input, grad_output):
            print("Backward hook called! Grad_output shape:", grad_output[0].shape)
            gradients['value'] = grad_output[0]

        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            outputs = self.model(input_tensor)

            # Check if activations have been captured
            if 'value' not in activations:
                print("No activation captured.")
                raise ValueError("Failed to capture activations for the target layer.")

            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(outputs)
            one_hot[0, class_idx] = 1
            outputs.backward(gradient=one_hot)

            # Check if gradients have been captured
            if 'value' not in gradients:
                print("No gradient captured.")
                raise ValueError("Failed to capture gradients for the target layer.")

            activation = activations['value'].detach().cpu().numpy()
            gradient = gradients['value'].detach().cpu().numpy()

            weights = np.mean(gradient, axis=(2, 3))[0, :]
            cam = np.zeros(activation.shape[2:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * activation[0, i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam

        finally:
            handle_forward.remove()
            handle_backward.remove()

def get_module_by_name(model, layer_name):
    module = model
    for name in layer_name.split('.'):
        if name.isdigit():
            module = module[int(name)]
        else:
            module = getattr(module, name)
    return module


def main():
    
    # Load weights securely
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        print("The version of PyTorch does not support 'weights_only'. Loading without this argument.")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print("Error loading state_dict:", e)
        print("Attempting to load with strict=False.")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Retrieve the target layer
    try:
        target_layer = get_module_by_name(model, target_layer_name)
        print(f"Grad-CAM target: {target_layer}")
    except AttributeError as e:
        print(f"Error: {e}")
        print(f"Check if the layer name '{target_layer_name}' is correct.")
        return

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess the image
    # Base path where the images are stored
    base_path = "Data/Rotten_or_Fresh_Test"

    # List of subdirectories (Rotten and Fresh)
    subdirectories = ['Rotten', 'Fresh']

    # Randomly choose a subdirectory
    chosen_directory = random.choice(subdirectories)

    # Construct the full path to the chosen directory
    directory_path = os.path.join(base_path, chosen_directory)

    # Get a list of all image files in the chosen directory
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    if not image_files:
        print(f"No images found in {directory_path}")
        return

    # Randomly select an image from the list
    image_path = os.path.join(directory_path, random.choice(image_files))

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"The specified image was not found: {image_path}")
        return

    preprocess = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Get the prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().detach().numpy()
    class_idx = np.argmax(probabilities)
    class_names = ["Good", "Rotten"]

    # Display classification results
    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[class_idx]} (Confidence: {probabilities[class_idx] * 100:.2f}%)")

    # Custom message based on confidence
    if class_idx == 0:
        prob_good = probabilities[0]
        if 0.5 <= prob_good <= 0.6:
            status = "Close to rotting"
        elif prob_good > 0.6:
            status = "Still very good"
        else:
            status = "Rotten"
    else:
        prob_bad = probabilities[1]
        if 0.5 <= prob_bad <= 0.6:
            status = "Close to rotting"
        elif prob_bad > 0.6:
            status = "Rotten"
        else:
            status = "Possibly still good"

    print(f"Status: {status}")

    # Calculate Grad-CAM
    try:
        cam = grad_cam._compute_gradcam(input_tensor.clone().detach().requires_grad_(True), class_idx)
    except ValueError as e:
        print(e)
        return

    # Apply the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize the heatmap to match the original image size
    heatmap = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
    heatmap = np.array(heatmap)

    # Overlay Grad-CAM on the original image
    overlay = Image.blend(image, Image.fromarray(heatmap), alpha=0.5)

    # Display the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()