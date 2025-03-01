import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from resnet import ResNet, Bottleneck
from utils import *

# Load a trained model from the given path
def load_model(model_path, num_classes=2):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

# Evaluate the model on the given dataset
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)[1]  # Extract classification output
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)  # Get predicted class index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute average loss and accuracy
    loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Run model evaluation on the test dataset
def test_model(data_path, model_save_folder):
    dataset = NiftiDataset(root_dir=data_path, transform=None)  # Load test dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    model = load_model(os.path.join(model_save_folder, 'best_model.pth'), 2)  # Load trained model
    criterion = nn.CrossEntropyLoss()  # Define loss function
    
    evaluate_model(model, dataloader, criterion)  # Perform evaluation

if __name__ == '__main__':
    set_random_seed(42)  # Ensure reproducibility
    
    model_save_folder = 'model_save_folder'  # Path to saved model weights
    data_path = 'test_data_set_folder'  # Path to test dataset
    
    test_model(data_path, model_save_folder)  # Run model testing