import streamlit as st
from utils import save_uploaded_files,merge_files
from pdf_processer import extract_text
from emb_vector import EmbModel,VectorDB
from chat_model import ChatModel
import os
import numpy as np
import time
st.title('小花的论文问答小助手')
message = '你好呀！我是小花的论文问答小助手 ~ 我会根据你上传的文件来回答问题哦 ~ '
if 'message' not in st.session_state:
    st.session_state.message = [{'role':'assistant','content':message}]
for msg in st.session_state.message:
    st.chat_message(msg['role']).write(msg['content'])
with st.sidebar:
    emb_selected = st.selectbox('请选择embedding模型',['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large','thenlper/gte-large-zh'])
    model_selected = st.selectbox('请选择AI模型',['gpt-3.5-turbo-16k', "gpt-3.5-turbo", 'gpt-3.5-turbo-1106', "gpt-4", 'gpt-4-0125-preview',
         'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview','gpt-4o','qwen'])
    api_key = st.text_input('请输入您的api_key',type = 'password')
    uploaded_files = st.file_uploader('请上传您的pdf文件',accept_multiple_files = True)
    if st.button('提取文件'):
        if uploaded_files:
            saved_files = save_uploaded_files(uploaded_files)
            if saved_files:
                merge_files('merged_file.pdf',saved_files)
                for path in saved_files:
                    if os.path.exists(path):
                        os.remove(path)
                paragraphs = extract_text('merged_file.pdf')
                if os.path.exists('merged_file.pdf'):
                    os.remove('merged_file.pdf')
                st.info('段落提取成功')
                if paragraphs:
                    emb_model = EmbModel({'model':emb_selected,'api_key':api_key})
                    st.info('emb_model创建成功')
                    text_emb = emb_model.extract_all_embs(paragraphs)
                    emb_model.save_emb(paragraphs,text_emb,'./db/temp.json')
                    if os.path.exists('./db/temp.json'):
                        st.success('文本成功转变为嵌入式向量')
                    else:
                        st.error('文本转化失败，请重试')
                else:
                    st.error('小花没有提取到段落哦')
        else:
            st.stop()

if os.path.exists('./db/temp.json'):
    emb_model = EmbModel({'model': emb_selected, 'api_key': api_key})
    vector_db = VectorDB(emb_model)
    vector_db.load_emb('./db/temp.json')
    model = ChatModel({'api_key': api_key, 'model': model_selected}, vector_db)

    prompt = st.chat_input()
    if prompt:
        st.session_state.message.append({'role':'user','content':prompt})
        st.chat_message('user').write(prompt)
        res = model.ask(prompt)
        st.session_state.message.append({'role':'assistant','content':res['msg']})
        st.chat_message('assistant').write(res['msg'])

