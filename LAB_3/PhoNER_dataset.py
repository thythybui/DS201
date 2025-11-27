import torch
from torch.utils.data import Dataset
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import re

PAD_TOKEN = '<PAD>' 
UNK_TOKEN = '<UNK>' 

class PhoNERDataset(Dataset):

    def __init__(self, path: str, max_len: int = MAX_SEQ_LEN, min_freq: int = MIN_FREQ, 
                 vocab: Optional[Dict[str, int]] = None, tag_map: Optional[Dict[str, int]] = None):
        super().__init__()
        
        self.max_len = max_len
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN

        raw_data, all_words, all_tags = self._load_data(path)

        if vocab is None:
            self.vocab, self.idx_to_vocab = self._build_vocab(all_words, min_freq)
        else:
            self.vocab = vocab
            self.idx_to_vocab = {idx: token for token, idx in vocab.items()}

        self.pad_id = self.vocab.get(PAD_TOKEN, 0)
        self.unk_id = self.vocab.get(UNK_TOKEN, 1)

        if tag_map is None:
            self.tag_map, self.idx_to_tag = self._build_tag_map(all_tags)
        else:
            self.tag_map = tag_map
            self.idx_to_tag = {idx: tag for tag, idx in tag_map.items()}

        if self.pad_token not in self.tag_map:
            self.tag_pad_id = self.pad_id 
        else:
            self.tag_pad_id = self.tag_map[self.pad_token]

        self._data: List[Dict[str, torch.Tensor]] = self._encode_data(raw_data)

    def _load_data(self, path: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        data = []
        all_words = []
        all_tags = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                words = item.get('words', [])
                tags = item.get('tags', [])
                
                if len(words) != len(tags) or not words:
                    continue
                    
                data.append({'words': words, 'tags': tags})
                all_words.extend(words)
                all_tags.extend(tags)
        
        return data, list(set(all_words)), list(set(all_tags))


    def _build_vocab(self, all_words: List[str], min_freq: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    
        word_counts = Counter(all_words)
        
        vocab = {
            self.pad_token: 0, 
            self.unk_token: 1
        }
      
        idx = 2
        for word, count in word_counts.items():
            if count >= min_freq and word not in vocab:
                vocab[word] = idx
                idx += 1
                
        idx_to_vocab = {idx: token for token, idx in vocab.items()}
        print(f"Vocab size: {len(vocab)} (>= {min_freq} lần)")
        return vocab, idx_to_vocab
    
    def _build_tag_map(self, all_tags: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
     
        tag_map = {self.pad_token: 0}
        idx = 1
      
        for tag in sorted(list(set(all_tags))): 
            if tag not in tag_map:
                tag_map[tag] = idx
                idx += 1
        
        idx_to_tag = {idx: tag for tag, idx in tag_map.items()}
        print(f"Number of NER tags: {len(tag_map)}")
        return tag_map, idx_to_tag


    def _encode_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        processed_data = []
        
        for item in raw_data:
            words = item['words']
            tags = item['tags']
            
            seq_id = [self.vocab.get(word, self.unk_id) for word in words]
            
            tag_id = [self.tag_map.get(tag, self.tag_pad_id) for tag in tags]
            
            if len(seq_id) > self.max_len:
                seq_id = seq_id[:self.max_len]
                tag_id = tag_id[:self.max_len]
            
            current_len = len(seq_id)
            if current_len < self.max_len:
                pad_length = self.max_len - current_len
                seq_id += [self.pad_id] * pad_length
                tag_id += [self.tag_pad_id] * pad_length 
                
            processed_data.append(
                {
                    'sequence': torch.tensor(seq_id, dtype=torch.long),
                    'tags': torch.tensor(tag_id, dtype=torch.long) 
                }
            )
            
        return processed_data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]