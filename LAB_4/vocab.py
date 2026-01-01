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
        
        self.specials = (self.pad_token, self.bos_token, self.eos_token, self.unk_token)
        
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
    def make_vocab(self, path:str, src_language:str, tgt_language:str):
        json_files = os.lisdir(path)
        src_words = set()
        tgt_words = set()
        
        for json_file in json_files:
            data = json.load(open(os.path.join(path, json_file), encoding='utf-8'))
            for item in     
    @property
    def total_src_tokens(self)->int:
        pass
    
    @property
    def total_tgt_tokens(self)->int:
        pass
    
    def preprocess_sentences(self, sentences:str):
        pass
    
    def encoder_sentences(self, sentences:str, language:str)-> torch.Tensor:
        pass
    
    def decoder_sentences(self, tensor: torch.Tensor, language:str)-> list[str]
        pass
    
    
        