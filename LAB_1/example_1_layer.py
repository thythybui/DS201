import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from mnist_dataset import MNISTDataset, collate_fn
from perceptron_1_layer import Perceptron1Layer
import numpy as np

def train(dataloader: DataLoader) -> list:
    model.train()
    total_loss = 0
    for batch in dataloader:
        images = batch['image'].to(device).view(-1, 28*28).float()
        labels = batch['label'].to(device).long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(dataloader: DataLoader) -> dict:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device).view(-1, 28*28).float()
            labels = batch['label'].to(device).long()
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def compute_score(dataloader: DataLoader, score_name: str) -> float:
    if score_name == 'accuracy':
        return evaluate(dataloader)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_dataset = MNISTDataset(
        image_path="D:/DS201/train-images.idx3-ubyte",
        label_path="D:/DS201/train-labels.idx1-ubyte"
    )
    
    test_dataset = MNISTDataset(
        image_path="D:/DS201/t10k-images.idx3-ubyte",   
        label_path="D:/DS201/t10k-labels.idx1-ubyte"
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle = True,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle = False,
        collate_fn=collate_fn
    )
    
    model = Perceptron1Layer(image_size=(28,28), num_labels=10).to(device)
    loss_fn = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    EPOCHS = 1
    best_score = 0
    best_score_name ='accuracy'
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        losses = train(train_dataloader)
        print(f"Loss: {np.array(losses).mean():.4f}")
       
    score = compute_score(test_dataloader, 'accuracy')

    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best model saved with accuracy: {best_score:.4f}")
        
        