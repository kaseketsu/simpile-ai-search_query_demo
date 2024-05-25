import streamlit as st
import json
import os
import PyPDF2
def save_uploaded_files(uploaded_files):
    paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join('uploads',uploaded_file.name)
        paths.append(file_path)
        try:
            with open(file_path,'wb') as file:
                file.write(uploaded_file.getbuffer())
            st.success(f'所有文件均已保存成功')
            return paths
        except Exception as e:
            st.error(f'文件保存失败:{e}')
            return False

def merge_files(output_path,input_path):
    merger = PyPDF2.PdfMerger()
    for path in input_path:
        merger.append(path)
    merger.write(output_path)
    merger.close()


def write_json_file(text_emb,file_path):
    with open(file_path,'w') as f:
        json.dump(text_emb,f)
    st.success('文本与向量映射保存成功')
