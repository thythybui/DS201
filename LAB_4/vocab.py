import torch
import re
import json
import os

class Vocab:
    
    def __init__(self, path: str, src_language: str, tgt_language: str):
        
        self.initialized_special_tokens()
        self.make_vocab(path, src_language, tgt_language)

        self.src_language = src_language
        self.tgt_language = tgt_language
    
    def initialize_special_tokens(self)-> None:
        
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.special_tokens = (self.pad_token, self.bos_token, self.eos_token, self.unk_token)
        
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.special_idxs = (self.pad_idx, self.bos_idx, self.eos_idx, self.unk_idx)

    def make_vocab(self, path:str, src_language:str, tgt_language:str):

        json_files = os.lisdir(path)

        src_words = set()
        tgt_words = set()
        
        for json_file in json_files:
            data = json.load(open(os.path.join(path, json_file), encoding='utf-8'))
            for item in data:
                src_sentence = item[src_language]
                tgt_sentence = item[tgt_language]

                src_tokens = self.preprocess_sentences(src_sentence)
                tgt_tokens = self.preprocess_sentences(tgt_sentence)

                src_words.update(src_tokens)
                tgt_words.update(tgt_tokens)

            src_stoi = list(self.special_tokens) + list(src_words)
            self.src_stoi = {i: tok for i, tok in enumerate(src_stoi)}
            self.src_itos = {tok: i for i, tok in enumerate(src_stoi)}

            tgt_stoi = list(self.special_tokens) + list(tgt_words)
            self.tgt_stoi = {i: tok for i, tok in enumerate(tgt_stoi)}
            self.tgt_itos = {tok: i for i, tok in enumerate(tgt_stoi)}

    @property
    def total_src_tokens(self)->int:
        return len(self.src_itos)
    
    @property
    def total_tgt_tokens(self)->int:
        return len(self.tgt_itos)
    
    def preprocess_sentences(self, sentences:str):
        sentence = sentences.lower()
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"\?", " ? ", sentence)
        sentence = re.sub(r";", " ; ", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"\"", " \" ", sentence)
        sentence = re.sub(r"'", " ' ", sentence)
        sentence = re.sub(r"\(", " ( ", sentence)
        sentence = re.sub(r"\)", " ) ", sentence)
        sentence = re.sub(r"\[", " [ ", sentence)
        sentence = re.sub(r"\]", " ] ", sentence)
        sentence = re.sub(r"/", " / ", sentence)

        sentence = " ".join(sentence.strip().split())
        tokens = sentence.strip().split()

        return tokens
    
    def encoder_sentences(self, sentence: str, language: str)-> torch.Tensor:
        
        tokens = self.preprocess_sentences(sentence)
        stoi = self.src_stoi if language == self.src_language else self.tgt_stoi
        vec = [stoi[token] if token in stoi else self.unk_idx for token in tokens]
        vec = [self.bos_idx] + vec + [self.eos_idx]
        vec = torch.Tensor(vec).long()

    def decoder_sentences(self, tensor: torch.Tensor, language:str)-> list[str]
        sentence_ids = tensor.tolist()
        sentences = []

        itos = self.src_itos if language == self.src_language else self.tgt_itos
        for sentence_id in sentence_ids:
            words = [itos[idx] for idx in sentence_id if idx not in self.special_idxs]
            sentence = " ".join(words)
            sentences.append(sentence)

            return sentences
    
        