import optuna
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import data_setup
from model_builder import ResNet50Transfer, MobileNetV3Transfer

# Define constants
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Setup directories
train_dir = "Data/Rotten_or_Fresh_Train_Small"
test_dir = "Data/Rotten_or_Fresh_Test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Define a function for Optuna optimization
def objective(trial):
    # Suggest hyperparameters
    model_name = trial.suggest_categorical("model", ["MobileNetV3", "ResNet50"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)


    # Select the model
    if model_name == "MobileNetV3":
        model = MobileNetV3Transfer(output_shape=2).to(device)
    elif model_name == "ResNet50":
        model = ResNet50Transfer(output_shape=2).to(device)

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Main function
if __name__ == "__main__":
    print(device)
    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    # Print best parameters
    print("Best hyperparameters:", study.best_params)