import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from minist_dataset import MNISTDataset, collate_fn
from model.LeNet import LeNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataloader: DataLoader) -> list:
    model.train()
    total_loss = 0
    for item in dataloader:
        images: torch.Tensor = item['image'].to(device).float()
        labels: torch.Tensor = item['label'].to(device).long()
        
        optimizer.zero_grad()
        outputs: torch.Tensor = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()  
    
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(dataloader: DataLoader) -> dict:
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for item in dataloader:
            images: torch.Tensor = item['image'].to(device).float()
            labels: torch.Tensor = item['label'].to(device).long()
            outputs: torch.Tensor = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())
            
        return {
            'f1_score': f1_score(true_labels, predictions, average='macro'),
            'precision': precision_score(true_labels, predictions, average='macro'),
            'recall': recall_score(true_labels, predictions, average='macro')
        }
        
if __name__ == "__main__":
    train_dataset = MNISTDataset(
        image_path="D:/DS201/LAB_2/MNIST_dataset/train-images.idx3-ubyte",
        label_path="D:/DS201/LAB_2/MNIST_dataset/train-labels.idx1-ubyte"
    )
    
    test_dataset = MNISTDataset(
        image_path="D:/DS201/LAB_2/MNIST_dataset/t10k-images.idx3-ubyte",
        label_path="D:/DS201/LAB_2/MNIST_dataset/t10k-labels.idx1-ubyte"
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(train_dataloader)
        scores = evaluate(test_dataloader)
        print(f"F1 Score: {scores['f1_score']:.4f}, Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}") 