import ollama
from tenacity import stop_after_attempt,wait_fixed,retry
class QwenModel:
    def __init__(self,config,vector_db):
        self.config = config
        self.vector_db = vector_db


    def ask(self,question):
        text_list = self.vector_db.query(question,2)
        messages = [{'role':'user',
                     'content':f"下面请你根据我提供的参考资料来回答问题，注意，这里只允许你根据参考资料来回答，如果参考资料提供的内容无法回答问题，则回复不知道。已知内容:{text_list} \n 我的问题:{question} "
                     }]
        response = ollama.chat(model = 'qwen',messages = messages)
        res = response['message']['content']
        return{
            'msg':res,
            'knowledge':text_list
        }

