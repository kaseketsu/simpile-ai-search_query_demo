from tqdm import tqdm
import numpy as np
from tenacity import retry,stop_after_attempt,wait_fixed
import requests
import json
from tqdm import tqdm
from utils import write_json_file
from typing import List
class OpenaiEmb:
    def __init__(self,config):
        self.config = config
        self.api_key = self.config['api_key']
        self.model = self.config['model']
        self.url = self.config.get('url','https://api.openai.com/v1/embeddings')

    @retry(stop=stop_after_attempt(5),wait=wait_fixed(5))
    def get_embs(self,text:List[str]):
        headers = {'Content-Type':'application/json',
                'Authorization':f'Bearer{self.api_key}'
                }
        payload = {
                'model':self.model,
                'input':text
        }
        response = requests.post(url = self.url,headers = headers,json = payload,stream = False,timeout = 180)
        response = json.loads(response.text)
        emb_array = []
        for i in range(len(response['data'])):
            emb_array.append(response['data'][i]['embedding'])
        return np.array(emb_array)



    def extract_all_embs(self,paragraphs:List[str],batch_size: int = 256):
        text_embs = []
        for i in tqdm(range(0,len(paragraphs),batch_size)):
            text = paragraphs[i:i + batch_size]
            temp_embs = self.get_embs(text)
            text_embs.append(temp_embs)
        text_embs = np.concatenate(text_embs)
        return text_embs

    def save_emb(self,paragraphs,text_emb,emb_file):
        text2_emb = {}
        for i in tqdm(range(len(paragraphs))):
            text2_emb[paragraphs[i]] = text_emb[i].tolist()
        write_json_file(text2_emb,emb_file)
