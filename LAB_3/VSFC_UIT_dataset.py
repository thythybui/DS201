import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import re

MAX_SEQ_LEN = 100 
MIN_FREQ = 2
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

class UIT_VSFC_Dataset(Dataset):
    def __init__(self, path: str, max_len: int = MAX_SEQ_LEN, min_freq: int = MIN_FREQ, 
                 vocab: Optional[Dict[str, int]] = None, label_map: Optional[Dict[str, int]] = None):
        super().__init__()
        
        self.max_len = max_len
        
        raw_data, all_texts, self.label_map, self.idx_to_label = self._load_data(path, label_map)
        
        if vocab is None:
            self.vocab, self.idx_to_vocab = self._build_vocab(all_texts, min_freq)
        else:
            self.vocab = vocab
            self.idx_to_vocab = {idx: token for token, idx in vocab.items()}

        self.pad_id = self.vocab.get(PAD_TOKEN, 0)
        self.unk_id = self.vocab.get(UNK_TOKEN, 1)

        self._data: List[Dict[str, torch.Tensor]] = self._encode_data(raw_data)
        
    def _preprocess_text(self, text: str) -> str:
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text) 
            return text.strip()
        return ""

    def _build_vocab(self, texts: List[str], min_freq: int) -> Tuple[Dict[str, int], Dict[int, str]]:
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        
        for word, count in word_counts.items():
            if count >= min_freq: 
                vocab[word] = len(vocab)
                
        idx_to_vocab: Dict[int, str] = {idx: word for word, idx in vocab.items()}
        return vocab, idx_to_vocab

    def _load_data(self, path: str, existing_label_map: Optional[Dict[str, int]]) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int], Dict[int, str]]:
        data = []
        all_texts = []
        label_map: Dict[str, int] = existing_label_map if existing_label_map is not None else {}
        
        df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label_str']) 
        
        if existing_label_map is None:
            unique_labels = df['label_str'].unique()
            label_map = {label: i for i, label in enumerate(unique_labels)}
        
        for index, row in df.iterrows():
            text = self._preprocess_text(row['text'])
            label_str = row['label_str']
            
            if label_str in label_map:
                data.append(
                    {
                        'text': text,
                        'label_id': label_map[label_str]
                    }
                )
                all_texts.append(text)
            
        idx_to_label = {idx: label for label, idx in label_map.items()}
        return data, all_texts, label_map, idx_to_label

    def _encode_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        processed_data = []
        
        for item in raw_data:
            text = item['text']
            seq = [self.vocab.get(word, self.unk_id) for word in text.split()]
            
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            
            if len(seq) < self.max_len:
                seq += [self.pad_id] * (self.max_len - len(seq))

            processed_data.append(
                {
                    'sequence': torch.tensor(seq, dtype=torch.long),
                    'label': torch.tensor(item['label_id'], dtype=torch.long)
                }
            )
            
        return processed_data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._data[index]