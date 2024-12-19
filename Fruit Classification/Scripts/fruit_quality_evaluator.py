import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random
import os
from model_builder import MobileNetV2Transfer, ResNet50Transfer
from typing import Dict

# Initialize the model and evaluate fruits
model_name = "models/model_resNet50.pth"
model = ResNet50Transfer(2)

class FruitQualityEvaluator:
    def __init__(self, model: torch.nn.Module, dataset_path: str, class_labels: list):
        self.model = model
        self.dataset_path = dataset_path
        self.class_labels = class_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def load_weights(self, model_path: str):
        """Load weights into the model."""
        try:
            self.model.load_state_dict(torch.load(model_path))
            print(f"The model weights have been loaded from {model_path}.")
        except Exception as e:
            raise ValueError(f"Error while loading weights: {e}")

    def get_random_image(self) -> (Image.Image, str):
        """Load a random image from the dataset."""
        all_images = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(root, file))
        
        if not all_images:
            raise FileNotFoundError(f"No images found in {self.dataset_path}")
        
        image_path = random.choice(all_images)
        image = Image.open(image_path).convert("RGB")
        return image, image_path

    def classify_image(self, image: Image.Image) -> Dict[str, float]:
        """Classify an image and return probabilities and status."""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)  # Raw outputs
            print(f"Raw model output: {output}")  # Added for debugging
            probabilities = F.softmax(output, dim=1).squeeze()  # Probabilities

        prob_good = probabilities[0].item()  # Probability for "Good"
        prob_bad = probabilities[1].item()  # Probability for "Rotten"

        # Check thresholds
        if prob_good > 0.6:
            state = "Still very good"
        elif 0.5 <= prob_good <= 0.6:
            state = "Close to rotting"
        else:
            state = "Rotten"

        return {"state": state, "prob_good": prob_good, "prob_bad": prob_bad}

    def run(self):
        """Run evaluation on a random image."""
        try:
            image, image_path = self.get_random_image()
            result = self.classify_image(image)
            print(f"Analyzed image: {image_path}")
            print(f"Probability 'Good': {result['prob_good']:.2f}")
            print(f"Probability 'Rotten': {result['prob_bad']:.2f}")
            print(f"Status: {result['state']}")
            image.show()
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# Usage
evaluator = FruitQualityEvaluator(model, "Data/Rotten_or_Fresh_Test", ["Good", "Rotten"])

# Load weights and run evaluation
try:
    evaluator.load_weights(model_name)
    evaluator.run()
except ValueError as e:
    print(e)