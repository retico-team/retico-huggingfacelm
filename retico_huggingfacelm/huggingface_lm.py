import os
import sys

# os.environ['RETICO'] = 'path/to/retico-core'
# os.environ['WASR'] = 'path/to/retico-whisperasr'

# sys.path.append(os.environ['RETICO'])
# sys.path.append(os.environ['WASR'])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
import retico_core
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU
from retico_whisperasr.whisperasr import WhisperASRModule

class HuggingfaceLM(abstract.AbstractModule):
    def __init__(self, asr, device, tokenizer, model, streamer):
        super().__init__()
        self.asr = asr
        self.sentence = ""

        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.streamer = streamer
        
    @staticmethod
    def name():
        return "Hugging Face LM Module"

    @staticmethod
    def description():
        return "A module running Hugging Face language model for real-time dialogue."

    @staticmethod
    def input_ius():
        return SpeechRecognitionIU

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update_message):
        text, _ = self.asr.acr.recognize()
        if text is not None:
            self.sentence = str(text)

        last_commit_sentence = None 

        for iu, um in update_message:  
            if um == abstract.UpdateType.COMMIT: 
                last_commit_sentence = self.sentence

        if last_commit_sentence:
            self.process_iu(last_commit_sentence, iu)
        

    def process_iu(self, last_commit_sentence, iu):

        messages = [
            {"role": "system",
            "content": "You are a friendly chatbot who responds to questions"},
            {"role": "user", 
            "content": last_commit_sentence},
        ]
        
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        input_length = tokenized_chat.shape[1]

        with torch.no_grad():
            output_tokens = self.model.generate(
                tokenized_chat, 
                max_new_tokens=500,
                temperature=0.2, 
                top_p=0.9, 
                do_sample=True,
                streamer=self.streamer        
        )
            
        response = self.tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True)
        words = response.split()  

        for word in words:
            current_iu = self.create_iu(iu)
            current_iu.payload = word
            self.current_output.append(current_iu)
            update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.ADD)
            self.append(update_message)

    def process_revoke(self, iu):
        pass

        
