import openai
from tenacity import stop_after_attempt,wait_fixed,retry
class OpenAIModel:
    def __init__(self,config,vector_db):
        self.config = config
        self.vector_db = vector_db


    def ask(self,question):
        openai.api_key = self.config['api_key']
        text_list = self.vector_db.query(question,2)
        messages = [{'role':'user',
                     'content':f"下面请你根据我提供的参考资料来回答问题，注意，这里只允许你根据参考资料来回答，如果参考资料提供的内容无法回答问题，则回复不知道。已知内容:{text_list} \n 我的问题:{question} "}]
        response = openai.ChatCompletion.create(model = self.config.get('model','gpt-3.5-turbo'),messages = messages)
        msg = response['choices'][0]['message']['content']
        return{
            'msg':msg,
            'knowledge':text_list
        }
