from typing import List
import streamlit as st
from .openai_emb import OpenaiEmb
from .hugging_face_emb import HuggingFaceEmb
class EmbModel:
    def __init__(self,config):
        self.config = config
        if self.config['model'] in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
            self.model = OpenaiEmb(self.config)
        else:
            self.model = HuggingFaceEmb(self.config)
    def extract_all_embs(self,paragraphs:List[str],batch_size: int = 8):
        return self.model.extract_all_embs(paragraphs,batch_size)

    def save_emb(self,paragraphs,text_emb,emb_file):
        self.model.save_emb(paragraphs,text_emb,emb_file)

    def get_embs(self,text):
        return self.model.get_embs(text)