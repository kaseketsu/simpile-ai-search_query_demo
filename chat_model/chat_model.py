import streamlit as st
from .openai_model import OpenAIModel
from .qwen_model import QwenModel
class ChatModel:
    def __init__(self,config,vector_db):
        self.config = config
        self.vector_db = vector_db
        if self.config['model'] in ['gpt-3.5-turbo-16k', "gpt-3.5-turbo", 'gpt-3.5-turbo-1106', "gpt-4", 'gpt-4-0125-preview',
         'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview','gpt-4o']:
            self.model = OpenAIModel(config,vector_db)
        else:
            self.model = QwenModel(config,vector_db)
    def ask(self,question):
        return self.model.ask(question)