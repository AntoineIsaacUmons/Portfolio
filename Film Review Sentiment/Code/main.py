import torch
import torch.nn as nn
from transformers import AutoTokenizer
from data_loader import IMDBDataset, load_data, get_data_loaders
from model_builder import ModelBuilder
from train_eval import train, eval_model, predict
import pandas as pd
from torch.utils.data import DataLoader

def main():
    # Define hyperparameters
    BATCH_SIZE = 64
    MAX_LENGTH = 256
    EPOCHS = 3
    LEARNING_RATE = 0.0001
    MODEL_NAME = 'bert'  # or 'distilbert'

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased' if MODEL_NAME == 'bert' else 'distilbert-base-uncased')

    # Load and prepare data
    train_dataset, test_dataset = load_data('Data/IMDB Dataset.csv', tokenizer, max_length=MAX_LENGTH)
    train_loader, val_loader = get_data_loaders(train_dataset, test_dataset, batch_size=BATCH_SIZE)

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.get_model(MODEL_NAME, n_classes=2)
    model = model.to(device)

    # Train the model
    trained_model = train(model, train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)

    # Save the model
    torch.save(trained_model.state_dict(), 'Models/sentiment_model.pth')

    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    test_loss, test_accuracy = eval_model(trained_model, val_loader, nn.CrossEntropyLoss().to(device), device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predict on new data
    print("\nMaking Predictions on New Data...")
    new_data = pd.DataFrame({
        'review': [
            "This movie was fantastic! I loved every minute of it.",
            "I was very disappointed with the film. It was boring and predictable."
        ]
    })

    # Prepare new data for prediction
    new_dataset = IMDBDataset(
        new_data['review'].values, 
        [0] * len(new_data),  # Dummy labels, we don't need them for prediction
        tokenizer, 
        MAX_LENGTH
    )
    new_loader = DataLoader(new_dataset, batch_size=1, shuffle=False)

    # Use the predict function from train_eval
    predictions = predict(trained_model, new_loader, device)

    # Print predictions
    for review, prediction in zip(new_data['review'], predictions):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Review: '{review}'\nPredicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()