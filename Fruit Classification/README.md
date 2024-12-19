Project: Fruit Classification - Rotten or Fresh

This project aims to classify fruits to determine if they are fresh or rotten using computer vision techniques and deep learning models. The training and test data come from the study "An extensive dataset for successful recognition of fresh and rotten fruits" by Nusrat Sultana, Musfika Jahan, and Mohammad Shorif Uddin, available at https://data.mendeley.com/datasets/bdd69gyhv8/1. This dataset offers a variety of fruit images in different states of freshness. Additional features like model interpretability through Grad-CAM are included to provide visual explanations of the model's decisions.

The inspiration for this project came from my experience working as a restocker at the Cora store. Every morning, employees must check if all fruits are rotten or not, which is very important for consumers and takes a significant amount of time daily. This AI could allow for quick analysis of stock to indicate where and if fruits are rotten, thereby improving efficiency.

Files:
data_setup.py
Contains functions for data preparation, including loading fruit images from the Mendeley dataset, creating DataLoaders for training and testing, and applying transformations to images.
engine.py
Manages the training and validation lifecycle of the model, including training loops, calculation, and recording of metrics.
fruit_quality_evaluator.py
Implements logic to evaluate fruit quality based on model outputs, providing states like "fresh", "close to rotting", "rotten".
grad_cam.py
Implements the Grad-CAM (Gradient-weighted Class Activation Mapping) technique to visualize which parts of an image most influence the model's decision, providing interpretability through generated heatmaps.
main.py
The entry point script that orchestrates the project's execution, including parameter selection and calls to training, testing, and evaluation processes.
model_builder.py
Contains definitions of model architectures (like MobileNetV3 or ResNet50) adapted for fruit classification.
optimisation.py
Uses libraries like Optuna to optimize the model's hyperparameters, aiming to enhance classification performance on fruits.
predict.py
Provides functionality for making predictions on specific images using a pre-trained model, with options for specifying image and model paths via command line.
predictions.py
Includes tools for making and visualizing predictions.
train.py
Dedicated to training the model, including training loops and interactions with the optimizer and loss function.
utils.py
Contains the method for saving a PyTorch model.
