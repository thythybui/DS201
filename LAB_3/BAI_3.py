import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, classification_report
from BiLSTM import BiLSTMEncoder
from PhoNER_dataset import PhoNERDataset

VOCAB_SIZE = 30000  
EMBEDDING_DIM = 100 
NUM_NER_TAGS = 13   

HIDDEN_SIZE = 256
NUM_LAYERS = 5
DROPOUT = 0.5
PAD_IDX = 0
MAX_SEQ_LEN = 100 
MIN_FREQ = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:      
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        text_sequences = batch['sequence'].to(device).long()
        labels = batch['tags'].to(device).long()          
        
        optimizer.zero_grad()
        outputs = model(text_sequences)
        loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
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
            labels = batch['tags'].to(device).long()
            
            outputs = model(text_sequences)
        
            max_preds = outputs.argmax(dim=2) 
            
            non_pad_elements = (labels != PAD_IDX).nonzero(as_tuple=True)
            
            filtered_preds = max_preds[non_pad_elements].cpu().numpy()
            filtered_labels = labels[non_pad_elements].cpu().numpy()
            
            all_preds.extend(filtered_preds)
            all_labels.extend(filtered_labels)
            
    f1_score = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"F1-Score (Weighted): {f1_score:.4f}")
    return f1_score


if __name__ == "__main__":
    
    
    train_dataset = PhoNERDataset(
        path='D:\DS201\LAB_3\Dataset\PhoNER\word\train_word.json', 
        max_len=MAX_SEQ_LEN, 
        min_freq=MIN_FREQ
        )
        
    VOCAB_SIZE = len(train_dataset.vocab)
    NUM_NER_TAGS = len(train_dataset.tag_map)
    
    test_dataset = PhoNERDataset(
        path='D:\DS201\LAB_3\Dataset\PhoNER\word\test_word.json', 
        max_len=MAX_SEQ_LEN, 
        min_freq=MIN_FREQ,
        vocab=train_dataset.vocab,
        tag_map=train_dataset.tag_map 
    )
    
    train_dataloader = DataLoader(
        dataset=[{'sequence': torch.zeros(100), 'tags': torch.zeros(100)}] * 64, 
        batch_size=32,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=[{'sequence': torch.zeros(100), 'tags': torch.zeros(100)}] * 32, 
        batch_size=32,
        shuffle=False
    )
    
    model = BiLSTMEncoder(
        vocab_size=VOCAB_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_ner_tags=NUM_NER_TAGS, 
        dropout=DROPOUT
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    
    EPOCHS = 10
    best_score = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        
        score = evaluate(test_dataloader, model, loss_fn, device)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_bilstm_phoner.pth")
            print(f"New best model saved with F1-Score (Macro): {best_score:.4f}")