import json
import faiss
import numpy as np
from typing import List
class VectorDB:
    def __init__(self,emb_model):
        self.text2emb = {}
        self.id2text = []
        self.emb_model = emb_model
        self.index = None

    def load_emb(self,file_path):
        with open(file_path,'r') as f:
            self.text2emb = json.load(f)
        d = len(next(iter(self.text2emb.values())))
        self.index = faiss.IndexFlatL2(d)
        for text,emb in self.text2emb.items():
            self.id2text.append(text)
            self.index.add(np.array(emb).reshape(1,-1).astype('float32'))
    def query(self,text,k:int):
        emb = self.emb_model.get_embs([text]).astype('float32')
        D,I = self.index.search(emb.reshape(1,-1),k)
        return [self.id2text[i] for i  in I[0]]





