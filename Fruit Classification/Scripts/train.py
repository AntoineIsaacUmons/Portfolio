"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if __name__ == '__main__':
  # Setup hyperparameters
  NUM_EPOCHS = 15
  BATCH_SIZE = 64
  HIDDEN_UNITS = 15
  LEARNING_RATE = 0.01

  # Setup directories
  train_dir = "Data/Rotten_or_Fresh_Train_Small"
  test_dir = "Data/Rotten_or_Fresh_Test"

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)
  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )
  

  model = model_builder.MobileNetV3Transfer(
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Start training with help from engine.py
  results = engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)
  
  # Afficher les résultats finaux après l'entraînement
  final_train_loss = results["train_loss"][-1]
  final_train_acc = results["train_acc"][-1]
  final_test_loss = results["test_loss"][-1]
  final_test_acc = results["test_acc"][-1]

  print("\nRésultats finaux après l'entraînement :")
  print(f"Train Loss: {final_train_loss:.4f}")
  print(f"Train Accuracy: {final_train_acc:.4f}%")
  print(f"Test Loss: {final_test_loss:.4f}")
  print(f"Test Accuracy: {final_test_acc:.4f}%")


  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="model_mobileNetV3.pth")
