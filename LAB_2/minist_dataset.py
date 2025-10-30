import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

def collate_fn(items: list[dict])-> dict[torch.Tensor]:
    items = [{
        'image': np.expand_dims(item['image'], axis=0),
        'label': np.array(item['label'])
    } for item in items]
    
    items = {
        'image': np.stack([item['image'] for item in items], axis=0),
        'label': np.stack([item['label'] for item in items], axis=0),
    }
    
    items = {
        'image': torch.tensor(items['image']),
        'label': torch.tensor(items['label'])
    }
    
    return items

class MNISTDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        self._data = self.load_data(image_path, label_path)
    
    def load_data(self, image_path: str, label_path: str):
        
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)
        
        self._data = [
            {
                'image': np.array(image),
                'label': label
            }
            for image, label in zip(images.tolist(), labels.tolist())
        ]
        return self._data
        
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> dict:
        return self._data[index]