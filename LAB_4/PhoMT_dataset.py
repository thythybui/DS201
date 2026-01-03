import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

from vocab import Vocab

def collate_fn (items: list[dict]) -> dict:

    vi_sents = [item['vietnamese'] for item in items]
    en_sents = [item['english'] for item in items]

    vi_sents = pad_sequence(vi_sents, batch_first=True, padding_value=2)
    en_sents = pad_sequence(en_sents, batch_first=True, padding_value=2)

    return {
        "vietnamese": vi_sents,
        "english": en_sents
    }
    

class PhoMTDataset(Dataset):
    
    def __init__(self, path: str, vocab: Vocab) :
        super().__init__()

        data = json.load(open(path, encoding='utf-8'))
        self.data = data
        self.vocab = Vocab

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int)
        
        item = self.data[index]

        vi_sentence = item['vietnamese']
        en_sentence = item['english']

        encoded_vi = self.vocab.encoder_sentences(vi_sentence, 'vietnamese')
        encoded_en = self.vocab.dencoder_sentences(en_sentence, 'english')

        return {
            'vietnamese': encoded_vi,
            'english': encoded_en
        }
