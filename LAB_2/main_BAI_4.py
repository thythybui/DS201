import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models 
from sklearn.metrics import f1_score, precision_score, recall_score
import os
from vinafood21_dataset import Vinafood21Dataset, collate_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_resnet50_finetune(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model.to(device)


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer) -> float:
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
        
        total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
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
            'f1_score': f1_score(true_labels, predictions, average='macro', zero_division=0),
            'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='macro', zero_division=0)
        }

if __name__ == "__main__":
    
    train_dataset = Vinafood21Dataset(
        path="D:/DS201/LAB_2/VinaFood21/train"
    )

    print("Số lớp dự kiến:", len(set([item['label'] for item in train_dataset])))
    print("Giá trị label duy nhất:", set([item['label'] for item in train_dataset])))

    test_dataset = Vinafood21Dataset(
        path="D:/DS201/LAB_2/Vinafood21/test",
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
    
    model = setup_resnet50_finetune(num_classes=num_classes)
    
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        
        metrics = evaluate(test_dataloader, model)
        
        print(f"Test F1 Score (Macro): {metrics['f1_score']:.4f}, Precision (Macro): {metrics['precision']:.4f}, Recall (Macro): {metrics['recall']:.4f}\n")