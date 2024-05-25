import torch.nn.functional as f
from torch import Tensor
from transformers import AutoTokenizer,AutoModel
import torch
from tenacity import retry,stop_after_attempt,wait_fixed
import numpy as np
from utils import write_json_file
import streamlit as st
class HuggingFaceEmb:
    def __init__(self,config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.model = AutoModel.from_pretrained(self.config['model']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'])


    def get_embs(self,text):
        try:
            batch_dict = self.tokenizer(text,max_length = 512,padding = True,
                                        truncation = True,return_tensors = 'pt')
            batch_dict = {k:v.to(self.device) for k,v in batch_dict.items()}

            outputs = self.model(**batch_dict)

            embeddings = outputs.last_hidden_state[:,0]

            embeddings_normal = f.normalize(embeddings,p = 2,dim = 1)
            embeddings_np = embeddings_normal.cpu().detach().numpy()

            return embeddings_np
        except Exception as e:
            print(f"Error in get_embs: {str(e)}")
            raise


    def extract_all_embs(self,paragraphs,batch_size : int = 8):
        text_emb = []
        for i in range(0,len(paragraphs),batch_size):
            text_list = paragraphs[i:i + batch_size]
            temp_emb = self.get_embs(text_list)
            text_emb.append(temp_emb)
        text_emb = np.concatenate(text_emb)
        return text_emb

    def save_emb(self,paragraphs,text_emb,file_path):
        text2emb = {}
        for i in range(len(paragraphs)):
            text2emb[paragraphs[i]] = text_emb[i].tolist()
        write_json_file(text2emb,file_path)


