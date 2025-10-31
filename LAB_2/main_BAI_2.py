import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from vinafood21_dataset import Vinafood21Dataset, collate_fn
from model.GoogleNet import GoogleNet

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
    train_dataset = Vinafood21Dataset(
        image_path="D:/DS201/LAB_2/VinaFood21/train",
        label_path="D:/DS201/LAB_2/VinaFood21/train"
    )
    
    test_dataset = Vinafood21Dataset(
        image_path="D:/DS201/LAB_2/Vinafood21/test",
        label_path="D:/DS201/LAB_2/Vinafood21/test"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model = GoogleNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(train_dataloader)
        metrics = evaluate(test_dataloader)
        print(f"Test F1 Score: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}\n")