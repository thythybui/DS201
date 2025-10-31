import numpy as np
import torch 
import torch.utils.data as Dataset
import os 
import cv2

def collate_fn(samples: list[dict]) -> torch.Tensor:
    
    samples = [{
        'image': np.expand_dims(sample['image'], axis=0),
        'label': np.array(sample['label'])
    } for sample in samples]
    
    samples = {
        'image': np.stack([sample['image'] for sample in samples], axis=0),
        'label': np.stack([sample['label'] for sample in samples], axis=0)
    }
    
    samples = {
        'image': torch.tensor(samples['image']),
        'label': torch.tensor(samples['label'])
    }
    
    return samples

class Vinafood21Dataset(Dataset):
    
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self._data: list[dict] = self.data_load(path)
        self.label2idx = {}
        self.idx2label = {}
        
    def data_load(self, path: str) -> list[dict]:
        data = []
        label_id = 0
        for folder in os.listdir(path):
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                self.idx2label[label_id] = label
                label_id += 1
            for image_path in os.listdir(os.path.join(path, folder)):
                image = cv2.imread(os.path.join(path, folder, image_path))
                image = cv2.resize(image, (224, 224))
                
                self._data.append(
                    {
                        'image': image,
                        'label': label_id
                    }
                )
        return self._data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> dict:
        return self._data[index]
                
                
        