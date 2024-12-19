"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

from torchvision import models

import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.

    Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                        out_channels=hidden_units, 
                        kernel_size=3, 
                        stride=1, 
                        padding=0),  
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                            stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*13*13,
                        out_features=output_shape)
        )
    
    def forward(self, x=torch.Tensor):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)
            return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


#How to import
# import torch
# # Import model_builder.py
# from going_modular import model_builder
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Instantiate an instance of the model from the "model_builder.py" script
# torch.manual_seed(42)
# model = model_builder.TinyVGG(input_shape=3,
#                               hidden_units=10, 
#                               output_shape=len(class_names)).to(device)




class MobileNetV2Transfer(nn.Module):
    """MobileNetV2 model with transfer learning for custom classification tasks.
    
    Args:
        num_classes: Number of output classes for the classification task.
    """
    def __init__(self, output_shape: int):
        super().__init__()
        # Load the pre-trained MobileNetV2 model
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze the base model's layers (optional: set to True if fine-tuning is not required)
        for param in self.base_model.features.parameters():
            param.requires_grad = False

        # Replace the classifier with a custom one
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, output_shape)  # MobileNetV2 uses `last_channel` for its output features
        )
    
    def forward(self, x: torch.Tensor):
        return self.base_model(x)


class ResNet50Transfer(nn.Module):
    """ResNet50 model with transfer learning for custom classification tasks.
    
    Args:
        output_shape (int): Number of output classes for the classification task.
    """
    def __init__(self, output_shape: int):
        super().__init__()
        # Load the pre-trained ResNet50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze the base model's layers (optional: set to True if fine-tuning is not required)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last block (layer4) of ResNet50
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # Replace the last fully connected layer with a custom one
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.base_model(x)
    





class MobileNetV3Transfer(nn.Module):
    """MobileNetV3 model with transfer learning for custom classification tasks."""
    
    def __init__(self, output_shape: int):
        super().__init__()
        # Load the pre-trained MobileNetV3 model
        self.base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # Freeze all layers by default
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 2 blocks
        for param in list(self.base_model.features[-2:].parameters()):
            param.requires_grad = True

        # Replace the classification head
        in_features = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.base_model(x)



class VGG16Transfer(nn.Module):
    """VGG16 model with transfer learning for custom classification tasks."""
    
    def __init__(self, output_shape: int):
        super().__init__()
        # Load the pre-trained VGG16 model
        self.base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Freeze all layers by default
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 3 convolutional layers along with their associated layers
        # The indices for the last 3 conv layers and their associated layers:
        # - conv5_1 (index 24), followed by relu5_1 and maxpool5 (index 25)
        # - conv5_2 (index 26), followed by relu5_2
        # - conv5_3 (index 28), followed by relu5_3
        for i in range(24, 29):  # This range includes conv5_1, relu5_1, maxpool5, conv5_2, relu5_2, conv5_3, relu5_3
            for param in self.base_model.features[i].parameters():
                param.requires_grad = True

        # Replace the classification head
        in_features = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.base_model(x)

