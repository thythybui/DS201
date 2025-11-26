import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from VSFC_UIT_dataset import UIT_VSFC_Dataset
from LSTM import LSTM 
import numpy as np
from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split
import pandas as pd

EMBEDDING_DIM = 100 
HIDDEN_SIZE = 256 
NUM_LAYERS = 5    
DROPOUT = 0.5
MAX_SEQ_LEN = 100
MIN_FREQ = 2

LEARNING_RATE = 1e-3 

def calculate_f1(preds: torch.Tensor, y: torch.Tensor) -> float:
    max_preds = preds.argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    return f1_score(y_np, max_preds, average='weighted')

def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:      
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        text_sequences = batch['sequence'].to(device).long()
        labels = batch['label'].to(device).long()
        
        optimizer.zero_grad()
        outputs = model(text_sequences)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            text_sequences = batch['sequence'].to(device).long()
            labels = batch['label'].to(device).long()
            
            outputs = model(text_sequences)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1-Score (Weighted): {f1:.4f}")
    return f1

def compute_score(dataloader: DataLoader, score_name: str, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> float:
    if score_name == 'f1_score':
        return evaluate(dataloader, model, loss_fn, device)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = 'uit_vsfc_data.tsv'


    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['text', 'label_str'])
    all_texts = df['text'].tolist()
    all_labels = df['label_str'].tolist() 
    
    X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)
    
    train_dataset = UIT_VSFC_Dataset(
        path=DATA_PATH, max_len=MAX_SEQ_LEN, min_freq=MIN_FREQ
    )
    
    train_dataset = UIT_VSFC_Dataset(path=DATA_PATH, max_len=MAX_SEQ_LEN, min_freq=MIN_FREQ)
    
    VOCAB_SIZE = len(train_dataset.vocab)
    NUM_CLASSES = len(train_dataset.label_map)
    
    test_dataset = UIT_VSFC_Dataset(
        path=DATA_PATH,
        max_len=MAX_SEQ_LEN, 
        min_freq=MIN_FREQ,
        vocab=train_dataset.vocab,
        label_map=train_dataset.label_map 
    )
        
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle = True,
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle = False,
    )
    
    model = LSTM( 
        vocab_size=VOCAB_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_labels=NUM_CLASSES, 
        dropout=DROPOUT
    ).to(device)
    
    loss_fn = nn.NLLLoss().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    EPOCHS = 10
    best_score = 0
    best_score_name ='f1_score' 

    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        
        score = compute_score(test_dataloader, best_score_name, model, loss_fn, device)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_lstm_vsfc.pth")
            print(f"New best model saved with F1-Score: {best_score:.4f}")