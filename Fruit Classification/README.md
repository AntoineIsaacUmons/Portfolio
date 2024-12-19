# Fruit Classification - Rotten or Fresh

**Project Overview:**  
This project endeavors to classify fruits as either fresh or rotten using computer vision and deep learning techniques. The dataset used for training and testing is sourced from the study **"An extensive dataset for successful recognition of fresh and rotten fruits"** by Nusrat Sultana, Musfika Jahan, and Mohammad Shorif Uddin, available at [https://data.mendeley.com/datasets/bdd69gyhv8/1](https://data.mendeley.com/datasets/bdd69gyhv8/1). It provides a diverse set of fruit images in varying states of freshness. Additionally, the project incorporates model interpretability through Grad-CAM for visual explanation of decisions made by the model.

**Project Inspiration:**  
The idea for this project was born from my experience working as a restocker at the Cora store. Every morning, employees must check if all fruits are rotten or not, which is very important for consumers and takes a significant amount of time daily. This AI could allow for quick analysis of stock to indicate where and if fruits are rotten, thereby improving efficiency.

## Project Structure

- **data_setup.py**
  - Contains functions for data preparation, including:
    - Loading fruit images from the Mendeley dataset.
    - Creating DataLoaders for training and testing.
    - Applying transformations to images.

- **engine.py**
  - Manages the training and validation lifecycle of the model:
    - Handles training loops.
    - Calculates and records metrics.

- **fruit_quality_evaluator.py**
  - Implements logic to evaluate fruit quality based on model outputs, providing states like "fresh", "close to rotting", "rotten".

- **grad_cam.py**
  - Implements the Grad-CAM (Gradient-weighted Class Activation Mapping) technique to visualize which parts of an image most influence the model's decision, providing interpretability through generated heatmaps.

- **main.py**
  - The entry point script that orchestrates the project's execution:
    - Parameter selection.
    - Calls to training, testing, and evaluation processes.

- **model_builder.py**
  - Contains definitions of model architectures (like MobileNetV3 or ResNet50) adapted for fruit classification.

- **optimisation.py**
  - Uses libraries like Optuna to optimize the model's hyperparameters, aiming to enhance classification performance on fruits.

- **predict.py**
  - Provides functionality for making predictions on specific images using a pre-trained model:
    - Options for specifying image and model paths via command line.

- **predictions.py**
  - Includes tools for making and visualizing predictions.

- **train.py**
  - Dedicated to training the model:
    - Manages training loops.
    - Interacts with the optimizer and loss function.

- **utils.py**
  - Contains the method for saving a PyTorch model.
