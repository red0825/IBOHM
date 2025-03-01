import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from resnet import ResNet, Bottleneck
from utils import *

# Model training function
# Trains a ResNet classifier using domain-separated NIfTI slice images.
def train_model(model_save_folder, model, dataloader, criterion, optimizer, num_epochs=25, patience=100):
    best_loss = float('inf')
    best_model_weights = None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            loss = criterion(model(inputs)[1], labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch} with best loss: {best_loss:.4f}')
            break
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), os.path.join(model_save_folder, 'best_model.pth'))
        print(f'Best model saved with loss: {best_loss:.4f}')

# Main function to execute training pipeline
def main(data_path, model_save_folder):
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = NiftiDataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model_save_folder, model, dataloader, criterion, optimizer, num_epochs=100)

if __name__ == '__main__':
    set_random_seed(42)
    model_save_folder = 'model_save_folder'  # Directory to save trained model weights
    data_path = 'data_folder'  # Directory containing domain-separated NIfTI datasets
    main(data_path, model_save_folder)
